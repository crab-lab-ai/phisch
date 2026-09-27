use burn::nn::{Linear, LinearConfig, SwiGlu, SwiGluConfig};
use burn::prelude::*;

use crate::config::model::ModelConfig;

#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    pub up_proj: SwiGlu<B>,
    pub down_proj: Linear<B>,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            up_proj: SwiGluConfig::new(config.d_model, config.d_ff).init(device),
            down_proj: LinearConfig::new(config.d_ff, config.d_model)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.down_proj.forward(self.up_proj.forward(input))
    }
}
