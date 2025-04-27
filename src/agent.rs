use std::mem::take;

use burn::{
    optim::{GradientsParams, Optimizer, SgdConfig},
    prelude::*,
    tensor::backend::AutodiffBackend,
};
use nn::loss::CrossEntropyLossConfig;
use rand::distr::{Distribution, weighted::WeightedIndex};
use shakmaty::{Bitboard, Chess, Color, Move, Position};

use crate::model::Model;

pub struct Agent<B: Backend> {
    memory: Vec<Record<B>>,
    model: Model<B>,
}

pub struct GameSession<'d, 'm, B: Backend> {
    trajectory: Vec<Record<B>>,
    state: Chess,
    last_move: Option<Move>,
    last_output: Option<Tensor<B, 1>>,
    device: &'d B::Device,
    model: &'m Model<B>,
}

pub struct Trajectory<B: Backend>(Vec<Record<B>>);

pub struct Record<B: Backend> {
    action: Move,
    model_output: Tensor<B, 1>,
    reward: f32,
}

fn chess_move_to_id(m: &Move) -> usize {
    let from = m.from().unwrap();
    let to = m.to();
    from as usize + to as usize * 64
}

impl<B: Backend> GameSession<'_, '_, B> {
    pub fn make_action(&mut self) -> Move {
        // generate input for model
        let mut board = self.state.board().clone();
        if self.state.turn() == Color::Black {
            board.swap_colors();
            board.flip_vertical();
        }
        let ep_arr = if let Some(ep_sq) = self.state.maybe_ep_square() {
            let mut arr = [[0.0; 8]; 8];
            let (file, rank) = ep_sq.coords();
            arr[file as usize][rank as usize] = 1.0;
            arr
        } else {
            [[0.0; 8]; 8]
        };
        let bitboard_to_arr = |mut b: Bitboard| -> [[f32; 8]; 8] {
            let mut arr = [[0.0; 8]; 8];
            while let Some(sq) = b.pop_back() {
                let (file, rank) = sq.coords();
                arr[file as usize][rank as usize] = 1.0;
            }
            arr
        };
        let input_tensor: Tensor<B, 4> = Tensor::from_data(
            [[
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
                ep_arr,
            ]],
            self.device,
        );

        // run the model
        let output_tensor = self.model.forward(input_tensor);
        let data: Vec<f32> = output_tensor.to_data().into_vec().unwrap();

        // pick a random move base on weight
        let legal_moves = self.state.legal_moves();
        let weights = legal_moves
            .iter()
            .map(chess_move_to_id)
            .map(|id| data[id])
            .map(f32::exp)
            .collect::<Vec<_>>();
        let dist = WeightedIndex::new(weights).unwrap();
        let mut rng = rand::rng();
        let picked_move = legal_moves[dist.sample(&mut rng)].clone();

        // take the move
        self.last_move = Some(picked_move.clone());
        self.last_output = Some(output_tensor.reshape([-1]));
        picked_move
    }

    pub fn get_feedback(&mut self, next_state: Chess, reward: f32) {
        self.state = next_state;
        if let Some(last_move) = self.last_move.take() {
            self.trajectory.push(Record {
                action: last_move,
                model_output: self.last_output.clone().unwrap(),
                reward,
            });
        }
    }

    pub fn game_end(self) -> Trajectory<B> {
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
            last_output: None,
            device,
            model: &self.model,
        }
    }

    pub fn collect_trajectory(&mut self, mut trajectory: Trajectory<B>) {
        self.memory.append(&mut trajectory.0);
    }

    pub fn get_memory_len(&self) -> usize {
        self.memory.len()
    }
}

impl<B: AutodiffBackend> Agent<B> {
    pub fn train_model(&mut self, device: &B::Device) {
        let cross_entropy = CrossEntropyLossConfig::new().init(device);
        let loss = take(&mut self.memory)
            .into_iter()
            .map(|record| {
                let target = chess_move_to_id(&record.action) as i32;
                cross_entropy
                    .forward(
                        record.model_output.reshape([1, -1]),
                        Tensor::from_ints([target], device),
                    )
                    .mul_scalar(record.reward)
            })
            .fold(Tensor::zeros([1], device), |acc, x| acc + x);
        let grads = loss.backward();
        let grads = GradientsParams::from_grads(grads, &self.model);
        let mut optim = SgdConfig::new().init();
        let lr = 0.00001;
        self.model = optim.step(lr, self.model.clone(), grads);
        self.memory.clear();
    }
}
