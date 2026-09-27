use std::{
    fs::{OpenOptions, create_dir_all},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use anyhow::Result;
use burn::config::Config;
use log::info;
use serde::{Deserialize, Serialize};

use crate::config::model::ModelConfig;

#[derive(Config, Debug)]
pub struct TrainConfig {
    pub model: ModelConfig,
    pub tokenizer: String,
    pub dataset: String,
    #[config(default = 4)]
    pub batch: usize,
    #[config(default = 5)]
    pub epochs: usize,
    #[config(default = 1e-3)]
    pub lr: f64,
    #[config(default = 0.05)]
    pub valid: f64,
    #[config(default = "String::from(\"model\")")]
    pub artifact: String,
    #[config(default = "None")]
    pub init: Option<String>,
    #[config(default = 50)]
    pub seed: u64,
    #[config(default = 200)]
    pub warmup: usize,
    #[config(default = 1.0)]
    pub clip: f32,
    #[config(default = 30)]
    pub checkpoint_minutes: u64,
    #[config(default = 8)]
    pub valid_batch: usize,
}

impl TrainConfig {
    pub fn generate_file(mut path: PathBuf) -> Result<()> {
        create_dir_all(&path)?;
        path.push("default-config.json");
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(&path)?;

        file.write_all(
            serde_json::to_string_pretty(&Self::new(
                ModelConfig::new(),
                "your-tokenizer.json".to_string(),
                "your-corpus.txt".to_string(),
            ))?
            .as_bytes(),
        )?;
        info!("Config generated at {}", path.display());

        Ok(())
    }

    pub fn config_for(dir: &Path, checkpoint: Option<&str>) -> Result<(Self, String)> {
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

        let config = Self::load(specific.unwrap_or_else(|| dir.join("config.json")))?;
        let checkpoint =
            checkpoint.map_or_else(|| format!("{}-best", config.artifact), str::to_string);

        Ok((config, checkpoint))
    }
}

#[derive(Serialize, Deserialize)]
pub struct Progress {
    pub epoch: usize,
    pub step: usize,
    pub global: usize,
    #[serde(default)]
    pub best: Option<f32>,
}

impl Progress {
    pub fn new(epoch: usize, step: usize, global: usize, best: f32) -> Self {
        Self {
            epoch,
            step,
            global,
            best: best.is_finite().then_some(best),
        }
    }

    pub fn save(&self, path: String) -> Result<()> {
        let mut file = OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)?;

        file.write_all(serde_json::to_string_pretty(self)?.as_bytes())?;

        Ok(())
    }

    pub fn load(path: String) -> Result<Self> {
        let mut file = OpenOptions::new().read(true).open(path)?;
        let mut buf = Vec::new();
        file.read_to_end(&mut buf)?;
        let progress = serde_json::from_slice(&buf)?;

        Ok(progress)
    }
}
