use burn::prelude::*;
use nn::{
    Linear, LinearConfig, Relu,
    conv::{Conv2d, Conv2dConfig},
};

#[derive(Config, Debug)]
pub struct ModelConfig {}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    conv1: Conv2d<B>,
    conv2: Conv2d<B>,
    conv3: Conv2d<B>,
    linear1: Linear<B>,
    linear2: Linear<B>,
    activation: Relu,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            conv1: Conv2dConfig::new([13, 16], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            conv2: Conv2dConfig::new([16, 16], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            conv3: Conv2dConfig::new([16, 8], [3, 3])
                .with_padding(nn::PaddingConfig2d::Same)
                .init(device),
            linear1: LinearConfig::new(8 * 64, 256).init(device),
            linear2: LinearConfig::new(256, 64 * 64).init(device),
            activation: Relu::new(),
        }
    }
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, input: Tensor<B, 4>) -> Tensor<B, 2> {
        let [batch_size, ..] = input.dims();

        let x = self.conv1.forward(input);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv3.forward(x);
        let x = self.activation.forward(x);
        let x = x.reshape([batch_size, 8 * 64]);

        let x = self.linear1.forward(x);
        let x = self.activation.forward(x);
        self.linear2.forward(x)
    }
}
