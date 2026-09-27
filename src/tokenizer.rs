use std::fs::create_dir_all;
use std::path::{Path, PathBuf};

use anyhow::anyhow;
use tokenizers::models::TrainerWrapper;
use tokenizers::models::bpe::{BPE, BpeTrainerBuilder};
use tokenizers::pre_tokenizers::byte_level::ByteLevel;
use tokenizers::{AddedToken, Tokenizer as Inner};

pub const END: &str = "<|end|>";
pub const SYSTEM: &str = "<|system|>";
pub const USER: &str = "<|user|>";
pub const ASSISTANT: &str = "<|assistant|>";

const RESERVED: usize = 8;

pub struct Tokenizer {
    inner: Inner,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let inner = Inner::from_file(path).map_err(|err| anyhow!("{err}"))?;

        Ok(Self { inner })
    }

    pub fn encode(&self, text: &str) -> anyhow::Result<Vec<u32>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|err| anyhow!("{err}"))?;

        Ok(encoding.get_ids().to_vec())
    }

    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> anyhow::Result<String> {
        self.inner
            .decode(ids, skip_special_tokens)
            .map_err(|err| anyhow!("{err}"))
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    pub fn id(&self, token: &str) -> anyhow::Result<u32> {
        self.inner
            .token_to_id(token)
            .ok_or_else(|| anyhow!("{token} is missing from the tokenizer"))
    }
}

fn special_tokens() -> Vec<AddedToken> {
    let mut tokens: Vec<AddedToken> = [END, SYSTEM, USER, ASSISTANT]
        .into_iter()
        .map(|token| AddedToken::from(token, true))
        .collect();

    tokens.extend((0..RESERVED).map(|i| AddedToken::from(format!("<|reserved_{i}|>"), true)));

    tokens
}

pub fn train(files: &[PathBuf], vocab_size: usize, output: &Path) -> anyhow::Result<usize> {
    let byte_level = ByteLevel::new(false, true, true);

    let mut tokenizer = Inner::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(byte_level));
    tokenizer.with_post_processor(Some(byte_level));
    tokenizer.with_decoder(Some(byte_level));

    let mut trainer: TrainerWrapper = BpeTrainerBuilder::new()
        .vocab_size(vocab_size)
        .min_frequency(2)
        .special_tokens(special_tokens())
        .initial_alphabet(ByteLevel::alphabet().into_iter().collect())
        .show_progress(false)
        .build()
        .into();

    let paths = files.iter().map(|p| p.display().to_string()).collect();

    tokenizer
        .train_from_files(&mut trainer, paths)
        .map_err(|err| anyhow!("{err}"))?;

    if let Some(parent) = output.parent()
        && !parent.exists()
    {
        create_dir_all(parent)?;
    }

    tokenizer
        .save(output, true)
        .map_err(|err| anyhow!("{err}"))?;

    Ok(tokenizer.get_vocab_size(true))
}
