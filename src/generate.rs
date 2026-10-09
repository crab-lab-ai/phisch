use std::path::Path;

use anyhow::Result;
use burn::prelude::*;
use burn::tensor::activation::softmax;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

use crate::backend::{Device, Inference};
use crate::config::model::ModelConfig;
use crate::model::Phisch;
use crate::model::kv_cache::KVCache;
use crate::store::load_model;
use crate::tokenizer::END;

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
) -> Result<()> {
    log::set_max_level(log::LevelFilter::Off);
    let device = Device::default();
    let (model, tokenizer, config) = load_model(dir, checkpoint, max_context, &device)?;

    let mut ids = tokenizer.encode(prompt)?;
    let end = tokenizer.id(END)?;
    let sampling = Sampling {
        tokens,
        top_k,
        temperature,
        seed,
        max_context,
    };

    ids.extend(complete(&model, &config, &ids, end, &sampling)?);

    println!("{}", tokenizer.decode(&ids[..], false)?);

    Ok(())
}

#[derive(serde::Deserialize)]
pub struct Sampling {
    pub tokens: usize,
    pub top_k: usize,
    pub temperature: f64,
    pub seed: u64,
    pub max_context: usize,
}

pub fn complete(
    model: &Phisch<Inference>,
    config: &ModelConfig,
    ids: &[u32],
    end: u32,
    sampling: &Sampling,
) -> Result<Vec<u32>> {
    let device = model.embedding.weight.val().device();
    let mut rng = StdRng::seed_from_u64(sampling.seed);
    let mut kv_cache = vec![KVCache::<Inference>::empty(config.seq_length); config.layers];

    let context = &ids[ids.len().saturating_sub(config.seq_length)..];

    anyhow::ensure!(
        sampling.max_context > context.len(),
        "max-context ({}) must be larger than the prompt ({} tokens)",
        sampling.max_context,
        context.len()
    );

    let input = Tensor::from_data(
        TensorData::new(context.to_vec(), [1, context.len()]),
        &device,
    );
    let mut logits = model.forward(input, Some(&mut kv_cache));
    let mut generated = Vec::new();

    for _ in 0..sampling.tokens {
        let [_, last, vocab] = logits.dims();
        let next = sample(
            logits.slice([0..1, last - 1..last]).reshape([vocab]),
            sampling.top_k,
            sampling.temperature,
            &mut rng,
        );

        if next == end {
            break;
        }

        generated.push(next);

        if kv_cache[0].seen >= sampling.max_context {
            break;
        }

        let input = Tensor::from_data(TensorData::new(vec![next], [1, 1]), &device);
        logits = model.forward(input, Some(&mut kv_cache));
    }

    Ok(generated)
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
