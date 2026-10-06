use anyhow::Result;
use clap::Parser;
use colored::Colorize;
use log::{LevelFilter, info};

use crate::{
    cli::{Cli, Commands},
    config::train::TrainConfig,
    info::{config_info, corpus_info},
    logger::Logger,
    store::save_safetensor,
};

mod backend;
mod cli;
mod config;
mod data;
mod generate;
mod info;
mod logger;
mod model;
mod store;
mod tokenizer;
mod train;

static LOGGER: Logger = Logger;

fn main() -> Result<()> {
    eprintln!("Phisch {}", "><>".blue().bold());
    if let Ok(()) = log::set_logger(&LOGGER) {
        log::set_max_level(LevelFilter::Info);
    }
    let cli = Cli::parse();

    match cli.command {
        Commands::Config { path } => TrainConfig::generate_file(path),
        Commands::Tokenizer { files, out, vocab } => {
            let size = tokenizer::train(&files, vocab, &out)?;
            info!("Trained {size} tokens -> {}", out.display());
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
        Commands::Info { command } => match command {
            cli::InfoCommand::Config { path } => config_info(&path),
            cli::InfoCommand::Corpus { path, tokenizer } => corpus_info(&path, &tokenizer),
        },
    }
}
