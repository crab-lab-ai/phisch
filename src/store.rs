use std::{fs::create_dir_all, path::Path};

use anyhow::{Result, anyhow};
use burn::{config::Config, module::Module, record::CompactRecorder};
use burn_store::{ModuleSnapshot, SafetensorsStore};
use log::info;

use crate::{
    backend::{Device, Inference},
    config::train::TrainConfig,
    model::Phisch,
    tokenizer::Tokenizer,
};

pub fn save_safetensor(checkpoint: Option<&str>, dir: &Path, out: &Path) -> Result<()> {
    create_dir_all(out)?;

    let safetensor_out = out.join(format!("{}.safetensors", checkpoint.unwrap_or("model")));
    let tokenizer_out = out.join("tokenizer.json");
    let config_out = out.join("config.json");

    let mut store = SafetensorsStore::from_file(&safetensor_out);
    let device = Device::default();
    let (config, checkpoint) = TrainConfig::config_for(dir, checkpoint)?;

    let tokenizer = Tokenizer::load(&config.tokenizer)?;
    let model = Phisch::<Inference>::new(&config.model, tokenizer.vocab_size(), &device)
        .load_file(dir.join(&checkpoint), &CompactRecorder::new(), &device)?;
    model.save_into(&mut store)?;

    tokenizer
        .into_inner()
        .save(&tokenizer_out, true)
        .map_err(|err| anyhow!("{err}"))?;

    config.model.save(&config_out)?;

    info!(
        "Safetensors, tokenizer and config written:\n- {}\n- {}\n-{}",
        safetensor_out.display(),
        tokenizer_out.display(),
        config_out.display()
    );

    Ok(())
}
