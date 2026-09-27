use burn::prelude::*;

#[derive(Debug, Clone)]
pub struct KVCache<B: Backend> {
    pub k_cache: Option<Tensor<B, 4>>,
    pub v_cache: Option<Tensor<B, 4>>,
    pub window: usize,
    pub seen: usize,
}

impl<B: Backend> KVCache<B> {
    pub fn empty(seq_length: usize) -> Self {
        Self {
            k_cache: None,
            v_cache: None,
            window: seq_length,
            seen: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.k_cache.as_ref().map_or(0, |k| k.dims()[2])
    }

    pub fn append(&mut self, keys_new: Tensor<B, 4>, values_new: Tensor<B, 4>) {
        let mut new_k;
        let mut new_v;

        self.seen += keys_new.dims()[2];

        if let Some(old_k) = self.k_cache.clone() {
            new_k = Tensor::cat(vec![old_k, keys_new], 2);
        } else {
            new_k = keys_new;
        }

        if let Some(old_v) = self.v_cache.clone() {
            new_v = Tensor::cat(vec![old_v, values_new], 2);
        } else {
            new_v = values_new;
        }

        let len = new_k.dims()[2];
        if len > self.window {
            new_k = new_k.narrow(2, len - self.window, self.window);
            new_v = new_v.narrow(2, len - self.window, self.window);
        }

        *self = Self {
            k_cache: Some(new_k),
            v_cache: Some(new_v),
            window: self.window,
            seen: self.seen,
        };
    }
}
