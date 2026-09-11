use burn::{
    nn::{BatchNorm, BatchNormConfig},
    prelude::*,
};
use nn::{
    Linear, LinearConfig, Relu,
    conv::{Conv2d, Conv2dConfig},
};

use crate::encode::UciMoveId;

#[derive(Config, Debug)]
pub struct ModelConfig {}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    batch1: BatchNorm<B>,
    residual: [ResBlock<B>; 10],
    conv2: Conv2d<B>,
    batch2: BatchNorm<B>,
    linear: Linear<B>,
    activation: Relu,
}

#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    batch1: BatchNorm<B>,
    conv2: Conv2d<B>,
    batch2: BatchNorm<B>,
    activation: Relu,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([13, 64], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            batch1: BatchNormConfig::new(64).init(device),
            residual: std::array::from_fn(|_| ResBlock {
                conv1: Conv2dConfig::new([64, 64], [3, 3])
                    .with_padding(nn::PaddingConfig2d::Same)
                    .init(device),
                batch1: BatchNormConfig::new(64).init(device),
                conv2: Conv2dConfig::new([64, 64], [3, 3])
                    .with_padding(nn::PaddingConfig2d::Same)
                    .init(device),
                batch2: BatchNormConfig::new(64).init(device),
                activation: Relu::new(),
            }),
            conv2: Conv2dConfig::new([64, 8], [1, 1])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            batch2: BatchNormConfig::new(8).init(device),
            linear: LinearConfig::new(8 * 64, UciMoveId::TOTAL as usize).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, ..] = input.dims();

        let x = self.conv1.forward(input);
        let x = self.batch1.forward(x);
        let mut x = self.activation.forward(x);
        for layer in &self.residual {
            x = layer.forward(x);
        }
        let x = self.conv2.forward(x);
        let x = self.batch2.forward(x);
        let x = self.activation.forward(x);
        let x = x.reshape([batch_size, 8 * 64]);

        self.linear.forward(x)
    }
}

impl<B: Backend> ResBlock<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 4> {
        let x = self.conv1.forward(input.clone());
        let x = self.batch1.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        let x = self.batch2.forward(x) + input;
        self.activation.forward(x)
    }
}
