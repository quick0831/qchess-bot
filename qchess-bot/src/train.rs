use burn::{
    nn::loss::{HuberLossConfig, Reduction},
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use rand::{
    distr::{Bernoulli, Distribution},
    seq::IndexedRandom,
};
use shakmaty::{Chess, Move, Position};

use crate::{
    model::{Model, chess_to_tensor},
    replay::{GameFlag, ReplayBuffer, Transition},
};

#[derive(Debug, Clone)]
pub struct TrainerConfig {
    pub learning_startup: usize,
    pub batch_size: usize,
    pub train_frequency: u32,
    pub gamma: f64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            learning_startup: 64,
            batch_size: 16,
            train_frequency: 4,
            gamma: 0.5,
        }
    }
}

pub struct Trainer<'d, B: AutodiffBackend> {
    device: &'d B::Device,
    replay_buffer: ReplayBuffer,
    online_model: Model<B>,
    target_model: Model<B>,
    config: TrainerConfig,
    train_counter: u32,
}

impl<'d, B: AutodiffBackend> Trainer<'d, B> {
    pub fn new(model: Model<B>, device: &'d B::Device, config: TrainerConfig) -> Self {
        Self {
            device,
            replay_buffer: ReplayBuffer::new(10000),
            online_model: model.clone(),
            target_model: model,
            config,
            train_counter: 0,
        }
    }

    pub fn model(&self) -> &Model<B> {
        &self.online_model
    }

    /// Perform a chess move on input games, with probability on getting random moves
    ///
    /// `epsilon`: The probability of getting random moves
    /// - `epsilon = 1` means full random
    /// - `epsilon = 0.5` means half random, half inferenced
    /// - `epsilon = 0` means fully inferenced
    pub fn make_move(&self, games: &[Chess], epsilon: f64) -> Vec<Move> {
        let dist = Bernoulli::new(epsilon).expect("Failed to create distribution");
        let mut rng = rand::rng();
        let take_random: Vec<bool> = dist.sample_iter(&mut rng).take(games.len()).collect();

        let input_games: Vec<Chess> = games
            .iter()
            .zip(take_random.iter().copied())
            .filter_map(|(game, b)| if b { None } else { Some(game) })
            .cloned()
            .collect();
        let mut inferenced_moves = self
            .online_model
            .inference(&input_games, self.device)
            .into_iter();

        let mut final_moves = Vec::new();
        for (game, b) in games.iter().zip(take_random) {
            let m = if b {
                *game.legal_moves().sample(&mut rng, 1).next().unwrap()
            } else {
                inferenced_moves.next().unwrap()
            };
            final_moves.push(m);
        }
        final_moves
    }

    pub fn record(&mut self, transition: Transition) {
        self.replay_buffer.push(transition);
    }

    pub fn step_counter(&mut self) -> bool {
        if self.replay_buffer.len() < self.config.learning_startup {
            return false;
        }
        self.train_counter += 1;
        self.train_counter >= self.config.train_frequency
    }

    pub fn train(&mut self, optim: &mut impl Optimizer<Model<B>, B>, lr: f64) {
        self.train_counter = 0;
        let mut rng = rand::rng();
        let mini_batch = self
            .replay_buffer
            .sample(&mut rng, self.config.batch_size)
            .cloned();
        type Unzipped = (((Vec<Chess>, Vec<Chess>), Vec<u16>), (Vec<f32>, Vec<i32>));
        let (((states, next_states), action), (reward, terminated)): Unzipped = mini_batch
            .map(|t| {
                let terminated = if t.flag == GameFlag::Terminated { 1 } else { 0 };
                let m = t.action.to_move(&t.state).unwrap();
                let next_state = t.state.clone().play(m).unwrap();
                (
                    ((t.state, next_state), t.action.u16()),
                    (t.reward, terminated),
                )
            })
            .unzip();

        let action = Tensor::from_ints(action.as_slice(), self.device);
        let reward = Tensor::<B, 1>::from_floats(reward.as_slice(), self.device).reshape([-1, 1]);
        let terminated = Tensor::<B, 1, Int>::from_ints(terminated.as_slice(), self.device)
            .reshape([-1, 1])
            .float();
        let states = chess_to_tensor(&states, self.device);
        let next_states = chess_to_tensor(&next_states, self.device);

        let expected_value = self.target_model.forward(next_states).max_dim(1).detach();
        let q_score = self.online_model.forward(states).gather(1, action);
        let target = expected_value
            .mul(terminated)
            .mul_scalar(self.config.gamma)
            .add(reward);

        let delta = 0.5;
        let huber_loss = HuberLossConfig::new(delta).init();
        let loss = huber_loss.forward(q_score, target, Reduction::Auto);
        let grads = GradientsParams::from_grads(loss.backward(), &self.online_model);

        self.online_model = optim.step(lr, self.online_model.clone(), grads);
    }
}
