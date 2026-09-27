use burn::config::Config;

#[derive(Config, Debug)]
pub struct ModelConfig {
    #[config(default = 1024)]
    pub d_model: usize,
    #[config(default = 2048)]
    pub d_ff: usize,
    #[config(default = 256)]
    pub seq_length: usize,
    #[config(default = 8)]
    pub heads: usize,
    #[config(default = 6)]
    pub layers: usize,
    #[config(default = 0.1)]
    pub dropout: f64,
}
