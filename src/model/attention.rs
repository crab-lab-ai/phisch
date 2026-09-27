use burn::nn::{Linear, LinearConfig, RotaryEncoding, RotaryEncodingConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::config::model::ModelConfig;
use crate::model::kv_cache::KVCache;

const MASK_VALUE: f32 = -1.0e4;

#[derive(Module, Debug)]
pub struct Attention<B: Backend> {
    pub d_model: usize,
    pub d_k: usize,
    pub q_proj: Linear<B>,
    pub k_proj: Linear<B>,
    pub v_proj: Linear<B>,
    pub rope: RotaryEncoding<B>,
    pub o_proj: Linear<B>,
}

impl<B: Backend> Attention<B> {
    pub fn new(config: &ModelConfig, device: &B::Device) -> Self {
        assert!(config.d_model.is_multiple_of(config.heads));

        let d_k = config.d_model / config.heads;

        Self {
            d_model: config.d_model,
            d_k,
            q_proj: LinearConfig::new(config.d_model, config.d_model)
                .with_bias(false)
                .init(device),
            k_proj: LinearConfig::new(config.d_model, config.d_model)
                .with_bias(false)
                .init(device),
            v_proj: LinearConfig::new(config.d_model, config.d_model)
                .with_bias(false)
                .init(device),
            rope: RotaryEncodingConfig::new(config.seq_length, d_k).init(device),
            o_proj: LinearConfig::new(config.d_model, config.d_model)
                .with_bias(false)
                .init(device),
        }
    }

    pub fn forward(
        &self,
        input: Tensor<B, 3>,
        mask: Option<Tensor<B, 4, Bool>>,
        cache: Option<&mut KVCache<B>>,
    ) -> Tensor<B, 3> {
        let mut q = self.split_heads(self.q_proj.forward(input.clone()));
        let mut k = self.split_heads(self.k_proj.forward(input.clone()));
        let mut v = self.split_heads(self.v_proj.forward(input));

        if let Some(cache) = cache {
            q = self.rope.apply(q, cache.seen);
            k = self.rope.apply(k, cache.seen);

            cache.append(k.clone(), v.clone());

            if let Some(k_cache) = cache.k_cache.clone() {
                k = k_cache;
            }

            if let Some(v_cache) = cache.v_cache.clone() {
                v = v_cache;
            }
        } else {
            q = self.rope.forward(q);
            k = self.rope.forward(k);
        }

        let context = self.attention(q, k, v, mask);

        self.o_proj.forward(self.merge_heads(context))
    }

    fn attention(
        &self,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        v: Tensor<B, 4>,
        mask: Option<Tensor<B, 4, Bool>>,
    ) -> Tensor<B, 4> {
        #[allow(clippy::cast_precision_loss)]
        let scores = q.matmul(k.transpose()).div_scalar((self.d_k as f64).sqrt());

        let scores = match mask {
            Some(mask) => scores.mask_fill(mask, MASK_VALUE),
            None => scores,
        };

        softmax(scores, 3).matmul(v)
    }

    fn split_heads(&self, input: Tensor<B, 3>) -> Tensor<B, 4> {
        let [batch_size, seq_len, d_model] = input.dims();
        input
            .reshape([batch_size, seq_len, d_model / self.d_k, self.d_k])
            .swap_dims(1, 2)
    }

    fn merge_heads(&self, input: Tensor<B, 4>) -> Tensor<B, 3> {
        let [batch_size, _, seq_len, _] = input.dims();

        input
            .swap_dims(1, 2)
            .reshape([batch_size, seq_len, self.d_model])
    }
}
