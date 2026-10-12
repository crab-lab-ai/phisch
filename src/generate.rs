use std::io::{Write, stdout};
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
use crate::tokenizer::{END, TextStream, Tokenizer};

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
    stream: bool,
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

    if stream {
        print!("{prompt}");

        let completion = Completion::new(&model, &config, &tokenizer, &ids, end, &sampling)?;
        let mut text_stream = TextStream::new(false);
        for id in completion {
            if let Some(decoded) = text_stream.push(&tokenizer, id)? {
                print!("{decoded}");
                stdout().flush()?;
            }
        }
        println!();
    } else {
        ids.extend(complete(&model, &config, &tokenizer, &ids, end, &sampling)?);

        println!("{}", tokenizer.decode(&ids[..], false)?);
    }

    Ok(())
}

pub struct Completion<'a> {
    model: &'a Phisch<Inference>,
    tokenizer: &'a Tokenizer,
    sampling: &'a Sampling,
    end: u32,
    device: Device,
    rng: StdRng,
    kv_cache: Vec<KVCache<Inference>>,
    input: Tensor<Inference, 2, Int>,
    history: Vec<u32>,
    count: usize,
}

impl<'a> Completion<'a> {
    pub fn new(
        model: &'a Phisch<Inference>,
        config: &'a ModelConfig,
        tokenizer: &'a Tokenizer,
        ids: &'a [u32],
        end: u32,
        sampling: &'a Sampling,
    ) -> Result<Self> {
        let device = model.embedding.weight.val().device();
        let rng = StdRng::seed_from_u64(sampling.seed);
        let kv_cache = vec![KVCache::<Inference>::empty(config.seq_length); config.layers];

        let context = &ids[ids.len().saturating_sub(config.seq_length)..];

        sampling.check(context.len())?;

        let input = Tensor::from_data(
            TensorData::new(context.to_vec(), [1, context.len()]),
            &device,
        );

        Ok(Self {
            model,
            tokenizer,
            sampling,
            end,
            device,
            rng,
            kv_cache,
            input,
            history: Vec::new(),
            count: 0,
        })
    }
}

impl Iterator for Completion<'_> {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count >= self.sampling.tokens {
            return None;
        }

        if self.kv_cache[0].seen >= self.sampling.max_context {
            return None;
        }

        let logits = self
            .model
            .forward(self.input.clone(), Some(&mut self.kv_cache));

        let [_, last, vocab] = logits.dims();

        let next = sample(
            logits.slice([0..1, last - 1..last]).reshape([vocab]),
            &self.history[self.history.len().saturating_sub(REPETITION_PENALTY_WINDOW)..],
            self.sampling,
            &mut self.rng,
        );

        if next == self.end {
            return None;
        }

        if !self.tokenizer.is_special_token(next) {
            self.history.push(next);
        }

        self.input = Tensor::from_data(TensorData::new(vec![next], [1, 1]), &self.device);

        self.count += 1;
        Some(next)
    }
}

pub fn complete(
    model: &Phisch<Inference>,
    config: &ModelConfig,
    tokenizer: &Tokenizer,
    ids: &[u32],
    end: u32,
    sampling: &Sampling,
) -> Result<Vec<u32>> {
    Ok(Completion::new(model, config, tokenizer, ids, end, sampling)?.collect())
}
