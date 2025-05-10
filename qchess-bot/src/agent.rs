use std::mem::{replace, take};

use burn::{
    optim::{GradientsParams, Optimizer},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::loss::CrossEntropyLossConfig;
use rand::distr::{Distribution, weighted::WeightedIndex};
use shakmaty::{Bitboard, Chess, Color, Move, Position};

use crate::model::Model;

pub struct Agent<B: Backend> {
    memory: Vec<Record>,
    model: Model<B>,
}

pub struct GameSession<'d, 'm, B: Backend> {
    trajectory: Vec<Record>,
    state: Chess,
    last_move: Option<u32>,
    device: &'d B::Device,
    model: &'m Model<B>,
}

pub struct Trajectory(Vec<Record>);

pub struct Record {
    state: Chess,
    action: u32,
    reward: f32,
}

fn chess_move_to_id(m: &Move) -> u32 {
    let from = m.from().unwrap();
    let to = m.to();
    from as u32 + to as u32 * 64
}

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
        // generate input for model
        let input_tensor: Tensor<B, 4> = chess_to_tensor(&[self.state.clone()], self.device);

        // run the model
        let output_tensor = self.model.forward(input_tensor);
        let data: Vec<f32> = output_tensor.to_data().into_vec().unwrap();

        // pick a random move base on weight
        let is_black = self.state.turn() == Color::Black;
        let legal_moves = self.state.legal_moves();
        let weights = legal_moves
            .iter()
            .map(|m| if is_black { m.to_mirrored() } else { m.clone() })
            .map(|m| chess_move_to_id(&m))
            .map(|id| data[id as usize])
            .map(f32::exp)
            // replace bad values with a small value
            .map(|w| if w > 0.01 { w } else { 0.01 })
            .collect::<Vec<_>>();
        let dist = WeightedIndex::new(weights).unwrap();
        let mut rng = rand::rng();
        let picked_move = legal_moves[dist.sample(&mut rng)].clone();

        // take the move
        self.last_move = Some(chess_move_to_id(&picked_move));
        picked_move
    }

    pub fn get_feedback(&mut self, next_state: Chess, reward: f32) {
        let state = replace(&mut self.state, next_state);
        if let Some(last_move) = self.last_move.take() {
            self.trajectory.push(Record {
                state,
                action: last_move,
                reward,
            });
        }
    }

    pub fn game_end(self) -> Trajectory {
        Trajectory(self.trajectory)
    }
}

impl<B: Backend> Agent<B> {
    pub fn new(model: Model<B>) -> Self {
        let memory = Vec::new();
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
            model: &self.model,
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
        self.memory.append(&mut trajectory);
    }

    pub fn get_memory_len(&self) -> usize {
        self.memory.len()
    }
}

impl<B: AutodiffBackend> Agent<B> {
    pub fn train_model(
        &mut self,
        device: &B::Device,
        optim: &mut impl Optimizer<Model<B>, B>,
        lr: f64,
    ) {
        let cross_entropy = CrossEntropyLossConfig::new().init(device);
        let (states, (actions, rewards)): (Vec<_>, (Vec<_>, Vec<_>)) = take(&mut self.memory)
            .into_iter()
            .map(|record| (record.state, (record.action, record.reward)))
            .unzip();
        let model_input = chess_to_tensor(&states, device);
        let model_output = self.model.forward(model_input);
        let targets = Tensor::from_ints(actions.as_slice(), device);
        let rewards = Tensor::from_floats(rewards.as_slice(), device);
        let grads = cross_entropy
            .forward(model_output, targets)
            .mul(rewards)
            .backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        self.model = optim.step(lr, self.model.clone(), grads);
    }
}
