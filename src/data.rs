use burn::data::dataloader::batcher::Batcher;
use burn::data::dataset::Dataset;
use burn::prelude::*;

#[derive(Debug, Clone)]
pub struct Sample {
    pub input: Vec<u32>,
    pub target: Vec<u32>,
}

pub struct TokenDataset {
    tokens: Vec<u32>,
    context: usize,
}

impl TokenDataset {
    pub fn new(tokens: Vec<u32>, context: usize) -> Self {
        Self { tokens, context }
    }
}

impl Dataset<Sample> for TokenDataset {
    fn get(&self, index: usize) -> Option<Sample> {
        let start = index * self.context;
        let end = start + self.context;

        Some(Sample {
            input: self.tokens.get(start..end)?.to_vec(),
            target: self.tokens.get(start + 1..=end)?.to_vec(),
        })
    }

    fn len(&self) -> usize {
        self.tokens.len().saturating_sub(1) / self.context
    }
}

#[derive(Debug, Clone)]
pub struct Batch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

#[derive(Debug, Clone, Default)]
pub struct TokenBatcher;

impl<B: Backend> Batcher<B, Sample, Batch<B>> for TokenBatcher {
    fn batch(&self, items: Vec<Sample>, device: &B::Device) -> Batch<B> {
        let shape = [items.len(), items[0].input.len()];

        let flatten = |rows: Vec<Vec<u32>>| {
            let data = rows.concat();
            Tensor::from_data(TensorData::new(data, shape), device)
        };

        let (inputs, targets) = items.into_iter().map(|s| (s.input, s.target)).unzip();

        Batch {
            inputs: flatten(inputs),
            targets: flatten(targets),
        }
    }
}
