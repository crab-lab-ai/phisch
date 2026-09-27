use anyhow::Result;
use clap::Parser;
use log::{LevelFilter, info};

use crate::{
    cli::{Cli, Commands},
    config::train::TrainConfig,
    logger::Logger,
    store::save_safetensor,
};

mod backend;
mod cli;
mod config;
mod data;
mod generate;
mod logger;
mod model;
mod store;
mod tokenizer;
mod train;

static LOGGER: Logger = Logger;

fn main() -> Result<()> {
    eprintln!("Welcome to Phisch! \x1b[0;32m><>\x1b[0m");
    let cli = Cli::parse();
    if let Ok(()) = log::set_logger(&LOGGER) {
        log::set_max_level(LevelFilter::Info);
    }

    match cli.command {
        Commands::Config { path } => TrainConfig::generate_file(path),
        Commands::Tokenizer { files, out, vocab } => {
            let size = tokenizer::train(&files, vocab, &out)?;
            info!("trained {size} tokens -> {}", out.display());
            Ok(())
        }
        Commands::Train {
            config,
            resume,
            artifacts,
        } => train::train(&config, resume, &artifacts),
        Commands::Generate {
            prompt,
            artifacts,
            checkpoint,
            tokens,
            top_k,
            temperature,
            seed,
            max_context,
        } => generate::generate(
            &artifacts,
            checkpoint.as_deref(),
            &prompt,
            tokens,
            top_k,
            temperature,
            seed,
            max_context,
        ),
        Commands::Export {
            checkpoint,
            artifacts,
            output,
        } => save_safetensor(checkpoint.as_deref(), &artifacts, &output),
    }
}
