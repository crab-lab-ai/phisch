use burn::backend::{Autodiff, Wgpu};

pub type Inference = Wgpu;
pub type Training = Autodiff<Inference>;
pub type Device = burn::tensor::Device<Inference>;
