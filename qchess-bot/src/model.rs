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
    batch1: BatchNorm<B, 2>,
    residual: [ResBlock<B>; 40],
    conv2: Conv2d<B>,
    batch2: BatchNorm<B, 2>,
    linear: Linear<B>,
    activation: Relu,
}

#[derive(Module, Debug)]
struct ResBlock<B: Backend> {
    conv1: Conv2d<B>,
    batch1: BatchNorm<B, 2>,
    conv2: Conv2d<B>,
    batch2: BatchNorm<B, 2>,
    activation: Relu,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([13, 256], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            batch1: BatchNormConfig::new(256).init(device),
            residual: std::array::from_fn(|_| ResBlock {
                conv1: Conv2dConfig::new([256, 256], [3, 3])
                    .with_padding(nn::PaddingConfig2d::Same)
                    .init(device),
                batch1: BatchNormConfig::new(256).init(device),
                conv2: Conv2dConfig::new([256, 256], [3, 3])
                    .with_padding(nn::PaddingConfig2d::Same)
                    .init(device),
                batch2: BatchNormConfig::new(256).init(device),
                activation: Relu::new(),
            }),
            conv2: Conv2dConfig::new([256, 2], [1, 1])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            batch2: BatchNormConfig::new(2).init(device),
            linear: LinearConfig::new(128, UciMoveId::TOTAL as usize).init(device),
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
        let x = x.reshape([batch_size, 2 * 64]);

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
