use std::path::Path;

use anyhow::Result;
use burn::prelude::*;
use rand::SeedableRng;
use rand::rngs::StdRng;

use crate::backend::{Device, Inference};
use crate::config::model::ModelConfig;
use crate::model::Phisch;
use crate::model::kv_cache::KVCache;
use crate::sampling::{Sampling, sample};
use crate::store::load_model;
use crate::tokenizer::{END, Tokenizer};

const REPETITION_PENALTY_WINDOW: usize = 32;

#[allow(clippy::too_many_arguments)]
pub fn generate(
    dir: &Path,
    checkpoint: Option<&str>,
    prompt: &str,
    tokens: usize,
    top_k: usize,
    temperature: f64,
    repetition_penalty: f64,
    seed: u64,
    max_context: usize,
) -> Result<()> {
    log::set_max_level(log::LevelFilter::Off);
    let device = Device::default();
    let (model, tokenizer, config, _) = load_model(dir, checkpoint, max_context, &device)?;

    let mut ids = tokenizer.encode(prompt)?;
    let end = tokenizer.id(END)?;
    let sampling = Sampling {
        tokens,
        repetition_penalty,
        top_k,
        temperature,
        seed,
        max_context,
    };

    ids.extend(complete(&model, &config, &tokenizer, &ids, end, &sampling)?);

    println!("{}", tokenizer.decode(&ids[..], false)?);

    Ok(())
}

pub fn complete(
    model: &Phisch<Inference>,
    config: &ModelConfig,
    tokenizer: &Tokenizer,
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

    anyhow::ensure!(
        sampling.repetition_penalty > 0.0,
        "repetition penalty must be greater than 0"
    );

    let input = Tensor::from_data(
        TensorData::new(context.to_vec(), [1, context.len()]),
        &device,
    );
    let mut logits = model.forward(input, Some(&mut kv_cache));
    let mut generated = Vec::new();
    let mut history = Vec::new();

    for _ in 0..sampling.tokens {
        let [_, last, vocab] = logits.dims();

        let next = sample(
            logits.slice([0..1, last - 1..last]).reshape([vocab]),
            &history[history.len().saturating_sub(REPETITION_PENALTY_WINDOW)..],
            sampling,
            &mut rng,
        );

        if next == end {
            break;
        }

        generated.push(next);
        if !tokenizer.is_special_token(next) {
            history.push(next);
        }

        if kv_cache[0].seen >= sampling.max_context {
            break;
        }

        let input = Tensor::from_data(TensorData::new(vec![next], [1, 1]), &device);
        logits = model.forward(input, Some(&mut kv_cache));
    }

    Ok(generated)
}
