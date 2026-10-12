use burn::{
    nn::{
        BatchNorm, BatchNormConfig, Linear, LinearConfig, Relu,
        conv::{Conv2d, Conv2dConfig},
    },
    prelude::*,
};
use shakmaty::{Bitboard, Chess, Color, Move, Position as _};

use crate::encode::UciMoveId;

#[derive(Config, Debug)]
pub struct ModelConfig {}

#[derive(Module, Debug)]
pub struct Model {
    conv1: Conv2d,
    batch1: BatchNorm,
    residual: [ResBlock; 10],
    conv2: Conv2d,
    batch2: BatchNorm,
    linear: Linear,
    activation: Relu,
}

#[derive(Module, Debug)]
struct ResBlock {
    conv1: Conv2d,
    batch1: BatchNorm,
    conv2: Conv2d,
    batch2: BatchNorm,
    activation: Relu,
}

impl ModelConfig {
    pub fn init(&self, device: &Device) -> Model {
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

impl Model {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<2> {
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

impl ResBlock {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let x = self.conv1.forward(input.clone());
        let x = self.batch1.forward(x);
        let x = self.activation.forward(x);
        let x = self.conv2.forward(x);
        let x = self.batch2.forward(x) + input;
        self.activation.forward(x)
    }
}

impl Model {
    pub fn inference(&self, games: &[Chess], device: &Device) -> Vec<Move> {
        // empty input tensor crashes the model
        if games.is_empty() {
            return Vec::new();
        }

        // generate input for model
        let input_tensor: Tensor<4> = chess_to_tensor(games, device);

        // run the model
        let output_tensor = self.forward(input_tensor);

        let mut picked_moves = Vec::new();
        for (idx, game) in games.iter().enumerate() {
            let data: Vec<f32> = output_tensor
                .clone()
                .slice(s![idx, ..])
                .into_data()
                .try_into_vec()
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

pub fn chess_to_tensor(games: &[Chess], device: &Device) -> Tensor<4> {
    let bitboard_to_arr = |mut b: Bitboard| -> [[f32; 8]; 8] {
        let mut arr = [[0.0; 8]; 8];
        while let Some(sq) = b.pop_back() {
            let (file, rank) = sq.coords();
            arr[file as usize][rank as usize] = 1.0;
        }
        arr
    };
    let data: Vec<f32> = games
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
    Tensor::<1>::from_floats(data, device).reshape([-1, 13, 8, 8])
}
