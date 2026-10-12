use std::path::PathBuf;

use clap::{Parser, Subcommand};

use crate::{
    sampling::{
        MAX_CONTEXT_DEFAULT, MAX_TOKENS_DEFAULT, REPETITION_PENALTY_DEFAULT, SEED_DEFAULT,
        TEMPERATURE_DEFAULT, TOP_K_DEFAULT,
    },
    tokenizer::VOCAB_SIZE_DEFAULT,
};

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
        #[arg(short, long, default_value_t = VOCAB_SIZE_DEFAULT, help = "Size of the vocabulary")]
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
            default_value_t = MAX_TOKENS_DEFAULT,
            help = "Amount of tokens to generate"
        )]
        tokens: usize,
        #[arg(
            short = 'k',
            long,
            default_value_t = TOP_K_DEFAULT,
            help = "Amount of candidates to sample from"
        )]
        top_k: usize,
        #[arg(
            long,
            default_value_t = TEMPERATURE_DEFAULT,
            help = "Sampling temperature, 0 for greedy"
        )]
        temperature: f64,
        #[arg(long, default_value_t = REPETITION_PENALTY_DEFAULT, help = "Repetition penalty")]
        repetition_penalty: f64,
        #[arg(short, long, default_value_t = SEED_DEFAULT, help = "Seed of the sampler")]
        seed: u64,
        #[arg(
            long,
            default_value_t = MAX_CONTEXT_DEFAULT,
            help = "Most tokens a single generation can reach, prompt included"
        )]
        max_context: usize,
        #[arg(long, help = "Stream a response")]
        stream: bool,
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
    #[command(about = "Run a web server with the model in the specified directory")]
    Serve {
        #[arg(
            long,
            default_value = "127.0.0.1",
            help = "Host address used to serve the api"
        )]
        host: String,
        #[arg(
            short,
            long,
            default_value_t = 6248,
            help = "Port used to serve the api"
        )]
        port: u16,
        #[arg(
            short,
            long,
            default_value = ARTIFACTS,
            help = "Directory of a trained model"
        )]
        directory: PathBuf,
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
