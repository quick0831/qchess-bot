use std::mem::replace;

use burn::{
    nn::loss::MseLoss,
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use rand::seq::IndexedRandom;
use shakmaty::{Chess, Move};

use crate::{
    encode::UciMoveId,
    model::{Model, chess_to_tensor},
    replay::{GameFlag, ReplayBuffer, Transition},
};

pub struct Agent<B: Backend> {
    memory: ReplayBuffer,
    model: Model<B>,
}

pub struct GameSession<'d, 'a, B: Backend> {
    trajectory: Vec<Transition>,
    state: Chess,
    last_move: Option<UciMoveId>,
    device: &'d B::Device,
    agent: &'a Agent<B>,
}

pub struct Trajectory(Vec<Transition>);

impl<B: Backend> GameSession<'_, '_, B> {
    pub fn make_action(&mut self) -> Move {
        let picked_move = self
            .agent
            .model
            .inference(std::slice::from_ref(&self.state), self.device)[0];
        self.last_move = Some(UciMoveId::from_move(&picked_move));
        picked_move
    }

    pub fn get_feedback(&mut self, next_state: Chess, reward: f32, flag: GameFlag) {
        let state = replace(&mut self.state, next_state);
        if let Some(last_move) = self.last_move.take() {
            self.trajectory.push(Transition {
                state,
                action: last_move,
                reward,
                flag,
            });
        }
    }

    pub fn game_end(self) -> Trajectory {
        Trajectory(self.trajectory)
    }
}

impl<B: Backend> Agent<B> {
    pub fn new(model: Model<B>) -> Self {
        let memory = ReplayBuffer::new(10000);
        Agent { memory, model }
    }

    pub fn start_new_game<'d>(
        &self,
        initial_state: Chess,
        device: &'d B::Device,
    ) -> GameSession<'d, '_, B> {
        GameSession {
            trajectory: Vec::new(),
            state: initial_state,
            last_move: None,
            device,
            agent: self,
        }
    }

    pub fn collect_trajectory(&mut self, trajectory: Trajectory, final_reward: f32) {
        let mut trajectory = trajectory.0;
        let decay = 0.9;
        let mut delayed_reward = final_reward;
        for record in trajectory.iter_mut().rev() {
            delayed_reward = record.reward + delayed_reward * decay;
            record.reward = delayed_reward;
        }
        for transition in trajectory {
            self.memory.push(transition);
        }
    }

    pub fn get_memory_len(&self) -> usize {
        self.memory.len()
    }

    pub fn model(&self) -> &Model<B> {
        &self.model
    }

    pub fn into_model(self) -> Model<B> {
        self.model
    }
}

impl<B: AutodiffBackend> Agent<B> {
    pub fn train_model(
        &mut self,
        device: &B::Device,
        optim: &mut impl Optimizer<Model<B>, B>,
        lr: f64,
    ) {
        let mut rng = rand::rng();
        let (states, (actions, rewards)): (Vec<_>, (Vec<_>, Vec<_>)) = self
            .memory
            .sample(&mut rng, 32)
            .cloned()
            .map(|record| (record.state, (record.action.u16(), record.reward)))
            .unzip();
        let model_input = chess_to_tensor(&states, device);
        let model_output = self.model.forward(model_input);
        let targets: Tensor<B, 1> = Tensor::from_floats(actions.as_slice(), device);
        let targets = targets.one_hot(UciMoveId::TOTAL as usize).detach();
        let rewards = Tensor::from_floats(rewards.as_slice(), device).detach();
        let grads = MseLoss::new()
            .forward(model_output, targets, nn::loss::Reduction::Mean)
            .mul(rewards)
            .backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = optim.step(lr, self.model.clone(), grads);
    }
}
