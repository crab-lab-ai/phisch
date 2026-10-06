use std::path::PathBuf;

use clap::{Parser, Subcommand};

const CONFIG: &str = "configs/";
const ARTIFACTS: &str = "artifacts/";
const OUTPUT: &str = "out/";

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand)]
pub enum Commands {
    #[command(about = "Generate a default config")]
    Config {
        #[arg(short, long, default_value = CONFIG, help = "Directory where the config file will generate")]
        path: PathBuf,
    },
    #[command(about = "Train the tokenizer")]
    Tokenizer {
        #[arg(index = 1, required = true, help = "Corpus files to train on")]
        files: Vec<PathBuf>,
        #[arg(
            short,
            long,
            default_value = "data/tokenizer.json",
            help = "Path of the resulting tokenizer"
        )]
        out: PathBuf,
        #[arg(short, long, default_value_t = 4096, help = "Size of the vocabulary")]
        vocab: usize,
    },
    #[command(about = "Train a Phisch")]
    Train {
        #[arg(index = 1, help = "Path of a training config")]
        config: PathBuf,
        #[arg(long, help = "Continue from the last step checkpoint of this config")]
        resume: bool,
        #[arg(short, long, default_value = ARTIFACTS, help = "Directory where checkpoints and configs are written")]
        artifacts: PathBuf,
    },
    #[command(about = "Generate a sequence of text")]
    Generate {
        #[arg(index = 1, help = "Prompt to complete")]
        prompt: String,
        #[arg(
            short,
            long,
            default_value = ARTIFACTS,
            help = "Directory of a trained model"
        )]
        artifacts: PathBuf,
        #[arg(
            long,
            help = "Checkpoint to load, defaults to the config artifact plus -best"
        )]
        checkpoint: Option<String>,
        #[arg(
            short,
            long,
            default_value_t = 64,
            help = "Amount of tokens to generate"
        )]
        tokens: usize,
        #[arg(
            short = 'k',
            long,
            default_value_t = 40,
            help = "Amount of candidates to sample from"
        )]
        top_k: usize,
        #[arg(
            long,
            default_value_t = 0.8,
            help = "Sampling temperature, 0 for greedy"
        )]
        temperature: f64,
        #[arg(short, long, default_value_t = 50, help = "Seed of the sampler")]
        seed: u64,
        #[arg(
            long,
            default_value_t = 4096,
            help = "Most tokens a single generation can reach, prompt included"
        )]
        max_context: usize,
    },
    #[command(about = "Export a checkpoint as safetensors")]
    Export {
        #[arg(
            long,
            help = "Checkpoint to load, defaults to the config artifact plus -best"
        )]
        checkpoint: Option<String>,
        #[arg(
            short,
            long,
            default_value = ARTIFACTS,
            help = "Directory of a trained model"
        )]
        artifacts: PathBuf,
        #[arg(short, long, default_value = OUTPUT, help = "Output directory for the safetensors file")]
        output: PathBuf,
    },
    #[command(about = "Show sizes of a config or a corpus")]
    Info {
        #[command(subcommand)]
        command: InfoCommand,
    },
}

#[derive(Subcommand)]
pub enum InfoCommand {
    Config {
        #[arg(index = 1, help = "Training config file")]
        path: PathBuf,
    },
    Corpus {
        #[arg(index = 1, help = "Corpus file")]
        path: PathBuf,
        #[arg(short, long, help = "Tokenizer file")]
        tokenizer: PathBuf,
    },
}
