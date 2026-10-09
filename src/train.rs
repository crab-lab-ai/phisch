use std::fs::{File, create_dir_all};
use std::io::{BufRead, BufReader};
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::data::dataloader::{DataLoader, DataLoaderBuilder};
use burn::module::AutodiffModule;
use burn::nn::loss::{CrossEntropyLoss, CrossEntropyLossConfig};
use burn::optim::grad_clipping::GradientClippingConfig;
use burn::optim::{AdamWConfig, GradientsParams, Optimizer};
use burn::prelude::*;
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::AutodiffBackend;
use log::{info, warn};

use crate::backend::{Device, Training};
use crate::config::train::{Progress, TrainConfig};
use crate::data::{Batch, TokenBatcher, TokenDataset};
use crate::model::Phisch;
use crate::tokenizer::Tokenizer;

type Inner = <Training as AutodiffBackend>::InnerBackend;

const LOG_EVERY: usize = 25;

fn corpus_tokens(tokenizer: &Tokenizer, path: &str) -> anyhow::Result<Vec<u32>> {
    let mut tokens = Vec::new();

    for line in BufReader::new(File::open(path)?).lines() {
        tokens.extend(tokenizer.encode(&format!("{}\n", line?))?);
    }

    Ok(tokens)
}

fn split(
    mut tokens: Vec<u32>,
    valid: f64,
    seq_length: usize,
) -> anyhow::Result<(Vec<u32>, Vec<u32>)> {
    let pivot = (tokens.len() as f64 * (1.0 - valid)) as usize;
    let valid_tokens = tokens.split_off(pivot);

    anyhow::ensure!(
        valid_tokens.len() > seq_length,
        "validation split has {} tokens but one window needs {}; raise `valid` or add data",
        valid_tokens.len(),
        seq_length + 1
    );

    Ok((tokens, valid_tokens))
}

fn loss<B: Backend>(
    model: &Phisch<B>,
    batch: Batch<B>,
    loss_fn: &CrossEntropyLoss<B>,
) -> Tensor<B, 1> {
    let logits = model.forward(batch.inputs, None);
    let [batch_size, context, vocab] = logits.dims();

    loss_fn.forward(
        logits.reshape([batch_size * context, vocab]),
        batch.targets.reshape([batch_size * context]),
    )
}

fn valid_loss(
    model: &Phisch<Inner>,
    loader: &Arc<dyn DataLoader<Inner, Batch<Inner>>>,
    loss_fn: &CrossEntropyLoss<Inner>,
) -> Option<f32> {
    let mut total = 0.0;
    let mut count = 0;

    for batch in loader.iter() {
        total += loss(model, batch, loss_fn).into_scalar().elem::<f32>();
        count += 1;
    }

    (count > 0).then(|| total / count as f32)
}

fn lr_at(config: &TrainConfig, global: usize, total: usize) -> f64 {
    if global < config.warmup {
        return config.lr * (global + 1) as f64 / config.warmup as f64;
    }

    let progress = (global - config.warmup) as f64 / (total - config.warmup).max(1) as f64;
    config.min_lr
        + 0.5
            * (config.lr - config.min_lr)
            * (1.0 + (std::f64::consts::PI * progress.min(1.0)).cos())
}

pub fn train(path: &Path, resume: bool, artifacts: &Path) -> anyhow::Result<()> {
    let config = TrainConfig::load(path)?;
    let device = Device::default();

    Training::seed(&device, config.seed);

    let tokenizer = Tokenizer::load(&config.tokenizer)?;
    let corpus = corpus_tokens(&tokenizer, &config.dataset)?;
    let (train_tokens, valid_tokens) = split(corpus, config.valid, config.model.seq_length)?;

    info!(
        "{} train tokens, {} valid tokens, vocab {}",
        train_tokens.len(),
        valid_tokens.len(),
        tokenizer.vocab_size()
    );

    let loader_train = DataLoaderBuilder::new(TokenBatcher)
        .batch_size(config.batch)
        .shuffle(config.seed)
        .build(TokenDataset::new(train_tokens, config.model.seq_length));

    let loader_valid = DataLoaderBuilder::<Inner, _, _>::new(TokenBatcher)
        .batch_size(config.valid_batch)
        .build(TokenDataset::new(valid_tokens, config.model.seq_length));

    let mut model = Phisch::<Training>::new(&config.model, tokenizer.vocab_size(), &device);
    info!("Training with {} parameters", model.num_params());

    if let Some(ref init) = config.init
        && !init.is_empty()
    {
        model = model.load_file(init, &CompactRecorder::new(), &device)?;
        info!("Initialised from checkpoint {init}");
    }

    let mut optimizer = AdamWConfig::new()
        .with_grad_clipping(Some(GradientClippingConfig::Norm(config.clip)))
        .init();

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let loss_fn_valid = CrossEntropyLossConfig::new().init(&device);

    let base = artifacts.join(&config.artifact).display().to_string();
    create_dir_all(artifacts)?;
    config.save(format!("{base}-config.json"))?;
    config.save(artifacts.join("config.json"))?;

    let [latest, best, checkpoint] =
        ["latest", "best", "step"].map(|kind| format!("{base}-{kind}"));

    let mut start = Progress::new(1, 0, 0, f32::INFINITY);

    if resume {
        (model, optimizer, start) = load_step(model, optimizer, &checkpoint, &device)?;
        info!("Resumed from epoch {} step {}", start.epoch, start.step);
    }

    let interval = Duration::from_secs(config.checkpoint_minutes * 60);
    let mut saved = Instant::now();
    let mut global = start.global;
    let mut best_loss = start.best.unwrap_or(f32::INFINITY);

    let total = config.epochs * loader_train.num_items().div_ceil(config.batch);

    for epoch in 1..=config.epochs {
        let batches = loader_train.iter();

        if epoch < start.epoch {
            continue;
        }

        let skip = if epoch == start.epoch { start.step } else { 0 };
        let mut last_loss = 0.;
        for (step, batch) in batches.enumerate().skip(skip) {
            let lr = lr_at(&config, global, total);
            let value = loss(&model, batch, &loss_fn);
            last_loss = value.clone().into_scalar().elem::<f32>();

            anyhow::ensure!(
                last_loss.is_finite(),
                "loss is {last_loss} at epoch {epoch} step {step}, last good checkpoint is {checkpoint}"
            );

            if step % LOG_EVERY == 0 {
                info!(
                    "Progress:\n- Epoch {epoch}\n- Step {step}\n- lr {lr:.2e}\n- loss {last_loss:.4}"
                );

                if saved.elapsed() >= interval {
                    save_step(
                        &model,
                        &optimizer,
                        &checkpoint,
                        &Progress::new(epoch, step, global, best_loss),
                    )?;
                    saved = Instant::now();
                    info!("Time save: checkpoint at epoch {epoch}, step {step}");
                }
            }

            let grads = GradientsParams::from_grads(value.backward(), &model);
            model = optimizer.step(lr, model, grads);
            global += 1;
        }

        let average = valid_loss(&model.valid(), &loader_valid, &loss_fn_valid);

        model.clone().save_file(&latest, &CompactRecorder::new())?;

        match average {
            Some(average) if average.is_finite() => {
                info!(
                    "Epoch finished:\n- epoch {epoch}\n- last loss {last_loss:.4}\n- valid loss {average:.4}"
                );

                if average < best_loss {
                    best_loss = average;
                    model.clone().save_file(&best, &CompactRecorder::new())?;
                    info!("Epoch {epoch} - new best");
                }
            }
            Some(average) => warn!(
                "Average is not finite:\n- epoch {epoch}\n- valid loss {average}\nbest left unchanged"
            ),
            None => {}
        }
    }

    Ok(())
}

fn save_step<O: Optimizer<Phisch<Training>, Training>>(
    model: &Phisch<Training>,
    optimizer: &O,
    checkpoint: &str,
    progress: &Progress,
) -> anyhow::Result<()> {
    model
        .clone()
        .save_file(checkpoint, &CompactRecorder::new())?;
    CompactRecorder::new().record(optimizer.to_record(), format!("{checkpoint}-optim").into())?;
    progress.save(format!("{checkpoint}-progress.json"))?;

    Ok(())
}

fn load_step<O: Optimizer<Phisch<Training>, Training>>(
    model: Phisch<Training>,
    optimizer: O,
    checkpoint: &str,
    device: &Device,
) -> anyhow::Result<(Phisch<Training>, O, Progress)> {
    let progress = Progress::load(format!("{checkpoint}-progress.json"))?;
    let model = model.load_file(checkpoint, &CompactRecorder::new(), device)?;
    let optimizer = optimizer
        .load_record(CompactRecorder::new().load(format!("{checkpoint}-optim").into(), device)?);

    Ok((model, optimizer, progress))
}
