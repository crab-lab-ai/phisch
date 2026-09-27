use std::path::Path;

use burn::prelude::*;
use burn::record::CompactRecorder;
use burn::tensor::activation::softmax;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::backend::{Device, Inference};
use crate::config::train::TrainConfig;
use crate::model::Phisch;
use crate::model::kv_cache::KVCache;
use crate::tokenizer::{END, Tokenizer};

fn config_for(dir: &Path, checkpoint: Option<&str>) -> anyhow::Result<TrainConfig> {
    let specific = checkpoint
        .map(|name| {
            ["-best", "-latest", "-step"]
                .iter()
                .fold(name, |name, suffix| {
                    name.strip_suffix(suffix).unwrap_or(name)
                })
        })
        .map(|artifact| dir.join(format!("{artifact}-config.json")))
        .filter(|path| path.exists());

    Ok(TrainConfig::load(
        specific.unwrap_or_else(|| dir.join("config.json")),
    )?)
}

#[allow(clippy::too_many_arguments)]
pub fn generate(
    dir: &Path,
    checkpoint: Option<&str>,
    prompt: &str,
    tokens: usize,
    top_k: usize,
    temperature: f64,
    seed: u64,
    max_context: usize,
) -> anyhow::Result<()> {
    log::set_max_level(log::LevelFilter::Off);
    let device = Device::default();
    let config = config_for(dir, checkpoint)?;

    let checkpoint = match checkpoint {
        Some(name) => name.to_string(),
        None => format!("{}-best", config.artifact),
    };

    let tokenizer = Tokenizer::load(&config.tokenizer)?;
    let rope_config = config.model.clone().with_seq_length(max_context);
    let model = Phisch::<Inference>::new(&rope_config, tokenizer.vocab_size(), &device).load_file(
        dir.join(&checkpoint),
        &CompactRecorder::new(),
        &device,
    )?;

    let mut ids = tokenizer.encode(prompt)?;
    let end = tokenizer.id(END)?;
    let mut rng = StdRng::seed_from_u64(seed);

    let mut kv_cache =
        vec![KVCache::<Inference>::empty(config.model.seq_length); config.model.layers];

    let context = &ids[ids.len().saturating_sub(config.model.seq_length)..];

    anyhow::ensure!(
        max_context > context.len(),
        "--max-context ({max_context}) must be larger than the prompt ({} tokens)",
        context.len()
    );

    let input = Tensor::from_data(
        TensorData::new(context.to_vec(), [1, context.len()]),
        &device,
    );
    let mut logits = model.forward(input, Some(&mut kv_cache));

    for _ in 0..tokens {
        let [_, last, vocab] = logits.dims();
        let next = sample(
            logits.slice([0..1, last - 1..last]).reshape([vocab]),
            top_k,
            temperature,
            &mut rng,
        );

        if next == end {
            break;
        }

        ids.push(next);

        if kv_cache[0].seen >= max_context {
            break;
        }

        let input = Tensor::from_data(TensorData::new(vec![next], [1, 1]), &device);
        logits = model.forward(input, Some(&mut kv_cache));
    }

    println!("{}", tokenizer.decode(&ids[..], false)?);

    Ok(())
}

fn sample<B: Backend>(
    logits: Tensor<B, 1>,
    top_k: usize,
    temperature: f64,
    rng: &mut StdRng,
) -> u32 {
    if temperature <= 0.0 {
        return logits.argmax(0).into_scalar().elem::<u32>();
    }

    let [vocab] = logits.dims();
    let (values, indices) = logits
        .div_scalar(temperature)
        .topk_with_indices(top_k.clamp(1, vocab), 0);

    let probabilities = softmax(values, 0).into_data();
    let indices = indices.into_data();

    let threshold: f32 = rng.random();
    let mut cumulative = 0.0;
    let mut last = 0;

    for (probability, index) in probabilities.iter::<f32>().zip(indices.iter::<u32>()) {
        cumulative += probability;
        last = index;
        if cumulative >= threshold {
            break;
        }
    }

    last
}
