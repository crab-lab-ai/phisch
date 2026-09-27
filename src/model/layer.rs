use burn::nn::{RmsNorm, RmsNormConfig};
use burn::prelude::*;

use crate::config::model::ModelConfig;
use crate::model::attention::Attention;
use crate::model::feed_forward::FeedForward;
use crate::model::kv_cache::KVCache;

#[derive(Module, Debug)]
pub struct DecoderLayer<B: Backend> {
    pub norm1: RmsNorm<B>,
    pub attention: Attention<B>,
    pub norm2: RmsNorm<B>,
    pub ff: FeedForward<B>,
}

impl<B: Backend> DecoderLayer<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        Self {
            norm1: RmsNormConfig::new(config.d_model).init(device),
            attention: Attention::new(config, device),
            norm2: RmsNormConfig::new(config.d_model).init(device),
            ff: FeedForward::new(config, device),
        }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: Option<&mut KVCache<B>>,
    ) -> Tensor<B, 3> {
        let x = self.norm1.forward(input.clone());

        let x = input + self.attention.forward(x, mask, cache);

        x.clone() + self.ff.forward(self.norm2.forward(x))
    }
}
