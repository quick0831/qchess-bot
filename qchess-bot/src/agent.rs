use std::mem::replace;

use burn::{
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::loss::MseLoss;
use rand::seq::IndexedRandom;
use shakmaty::{Bitboard, Chess, Color, Move, Position};

use crate::{
    encode::UciMoveId,
    model::Model,
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

fn chess_to_tensor<B: Backend>(chess: &[Chess], device: &B::Device) -> Tensor<B, 4> {
    let bitboard_to_arr = |mut b: Bitboard| -> [[f32; 8]; 8] {
        let mut arr = [[0.0; 8]; 8];
        while let Some(sq) = b.pop_back() {
            let (file, rank) = sq.coords();
            arr[file as usize][rank as usize] = 1.0;
        }
        arr
    };
    let data: Vec<f32> = chess
        .iter()
        .flat_map(|chess| {
            let mut board = chess.board().clone();
            if chess.turn() == Color::Black {
                board.swap_colors();
                board.flip_vertical();
            }
            [
                bitboard_to_arr(board.white().intersect(board.pawns())),
                bitboard_to_arr(board.white().intersect(board.knights())),
                bitboard_to_arr(board.white().intersect(board.bishops())),
                bitboard_to_arr(board.white().intersect(board.rooks())),
                bitboard_to_arr(board.white().intersect(board.queens())),
                bitboard_to_arr(board.white().intersect(board.kings())),
                bitboard_to_arr(board.black().intersect(board.pawns())),
                bitboard_to_arr(board.black().intersect(board.knights())),
                bitboard_to_arr(board.black().intersect(board.bishops())),
                bitboard_to_arr(board.black().intersect(board.rooks())),
                bitboard_to_arr(board.black().intersect(board.queens())),
                bitboard_to_arr(board.black().intersect(board.kings())),
                if let Some(ep_sq) = chess.maybe_ep_square() {
                    let mut arr = [[0.0; 8]; 8];
                    let (file, rank) = ep_sq.coords();
                    arr[file as usize][rank as usize] = 1.0;
                    arr
                } else {
                    [[0.0; 8]; 8]
                },
            ]
        })
        .flatten()
        .flatten()
        .collect();
    let data: &[f32] = &data;
    Tensor::<B, 1>::from_floats(data, device).reshape([-1, 13, 8, 8])
}

impl<B: Backend> GameSession<'_, '_, B> {
    pub fn make_action(&mut self) -> Move {
        let picked_move = self
            .agent
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

    pub fn inference(&self, games: &[Chess], device: &B::Device) -> Vec<Move> {
        // generate input for model
        let input_tensor: Tensor<B, 4> = chess_to_tensor(games, device);

        // run the model
        let output_tensor = self.model.forward(input_tensor);

        let mut picked_moves = Vec::new();
        for (idx, game) in games.iter().enumerate() {
            let data: Vec<f32> = output_tensor
                .clone()
                .slice(s![idx, ..])
                .into_data()
                .into_vec()
                .unwrap();
            let is_black = game.turn() == Color::Black;
            let legal_moves = game.legal_moves();
            let picked_move = legal_moves
                .into_iter()
                .map(|m| {
                    let flipped = if is_black { m.to_mirrored() } else { m };
                    let id = UciMoveId::from_move(&flipped);
                    let q_score = data[id.u16() as usize];
                    (m, q_score)
                })
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(m, _)| m)
                .expect("No valid move");

            // take the move
            picked_moves.push(picked_move);
        }

        picked_moves
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
