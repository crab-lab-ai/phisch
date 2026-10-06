use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use anyhow::Result;
use burn::config::Config;
use burn::module::Module;
use log::info;

use crate::backend::{Device, Training};
use crate::config::train::TrainConfig;
use crate::model::Phisch;
use crate::tokenizer::Tokenizer;

struct Corpus {
    bytes: usize,
    lines: usize,
    words: usize,
    chars: usize,
    tokens: usize,
}

impl Corpus {
    fn count(path: &Path, tokenizer: &Tokenizer) -> anyhow::Result<Self> {
        let mut bytes = 0;
        let mut lines = 0;
        let mut words = 0;
        let mut chars = 0;
        let mut tokens = 0;

        for line in BufReader::new(File::open(path)?).lines() {
            let line = format!("{}\n", line?);
            bytes += line.len();
            lines += 1;
            words += line.split_whitespace().count();
            chars += line.chars().count();
            tokens += tokenizer.encode(&line)?.len();
        }

        Ok(Self {
            bytes,
            lines,
            words,
            chars,
            tokens,
        })
    }
}

#[allow(clippy::cast_precision_loss)]
pub fn corpus_info(path: &Path, tokenizer_path: &Path) -> Result<()> {
    let tokenizer = Tokenizer::load(tokenizer_path)?;
    let corpus = Corpus::count(path, &tokenizer)?;

    info!(
        "Corpus {} with Tokenizer {}\n- size {}\n- lines {}\n- words {}\n- chars {}\n- tokens {}\n- chars per token {:.2}",
        path.display(),
        tokenizer_path.display(),
        bytes(corpus.bytes as f64),
        human(corpus.lines as f64),
        human(corpus.words as f64),
        human(corpus.chars as f64),
        human(corpus.tokens as f64),
        corpus.chars as f64 / corpus.tokens as f64,
    );

    Ok(())
}

#[allow(clippy::cast_precision_loss)]
pub fn config_info(path: &Path) -> Result<()> {
    let config = TrainConfig::load(path)?;
    let device = Device::default();

    let tokenizer = Tokenizer::load(&config.tokenizer)?;

    let model = Phisch::<Training>::new(&config.model, tokenizer.vocab_size(), &device);

    info!(
        "Config {}\n- tokenizer {}\n- parameters {}",
        path.display(),
        config.tokenizer,
        human(model.num_params() as f64)
    );

    drop(model);
    drop(tokenizer);

    Ok(())
}

fn human(value: f64) -> String {
    let k = 1000.;
    let m = 1_000_000.;
    let b = 1_000_000_000.;
    match value {
        v if v >= b => format!("{:.2}B", v / b),
        v if v >= m => format!("{:.2}M", v / m),
        v if v >= k => format!("{:.2}K", v / k),
        v => format!("{v}"),
    }
}

fn bytes(value: f64) -> String {
    let kbs = 1024.;
    let mbs = kbs * kbs;
    let gbs = mbs * kbs;
    match value {
        v if v >= gbs => format!("{:.2} GB", v / gbs),
        v if v >= mbs => format!("{:.2} MB", v / mbs),
        v if v >= kbs => format!("{:.2} KB", v / kbs),
        v => format!("{v} B"),
    }
}
