#![recursion_limit = "256"]

use std::mem::take;

use burn::{optim::AdamConfig, prelude::*, tensor::Device};
use qchess_bot::{
    encode::UciMoveId,
    model::{Model, ModelConfig},
    replay::{GameFlag, Transition},
    train::{Trainer, TrainerConfig},
};
use rand::seq::IndexedRandom;
use shakmaty::{Chess, Color, KnownOutcome, Outcome, Position};

fn main() {
    let device = cfg_select! {
        feature = "wgpu" => Device::wgpu(DeviceKind::DefaultDevice),
        feature = "cuda" => Device::cuda(),
        feature = "cpu" => Device::flex(),
    };

    let device = device.autodiff();

    let model_config = ModelConfig {};

    let model: Model = model_config.init(&device);

    println!("{model}");

    let trainer_config = TrainerConfig::default();
    let mut trainer = Trainer::new(model, &device, trainer_config);

    let mut optim = AdamConfig::new().init();
    let lr = 0.0001;
    let epsilon = 0.5;
    for epoch in 0..2000 {
        println!("epoch: {:5}", epoch + 1);
        let mut train_counter = 0;
        while train_counter < 10 {
            let mut games = vec![Chess::new(); 10];
            let mut finished_games = Vec::new();
            while !games.is_empty() {
                let actions = trainer.make_move(&games, epsilon);
                for (mut game, action) in take(&mut games).into_iter().zip(actions) {
                    let mut reward = 0.0 + -0.002 * game.fullmoves().get() as f32; // punish for making the game long
                    if action.is_capture() {
                        // reward for capturing
                        reward += 0.1;
                    }
                    game.play_unchecked(action);
                    if game.is_check() {
                        // reward for checking the opposing king
                        reward += 0.1;
                    }
                    // Stop the session if the game is too long
                    let flag = match game.outcome() {
                        Outcome::Known(outcome) => {
                            reward += match outcome {
                                KnownOutcome::Decisive { winner: _ } => 1.,
                                KnownOutcome::Draw => -0.3,
                            };
                            GameFlag::Terminated
                        }
                        Outcome::Unknown => match game.fullmoves().get() {
                            250.. => GameFlag::Truncated,
                            _ => GameFlag::Nothing,
                        },
                    };
                    trainer.record(Transition {
                        state: game.clone(),
                        action: UciMoveId::from_move(&action),
                        reward,
                        flag,
                    });
                    if flag == GameFlag::Nothing {
                        games.push(game);
                    } else {
                        finished_games.push(game);
                    }
                }
            }
            for (idx, game) in games.iter().enumerate() {
                println!(
                    "[{:2}] Fullmoves: {:3}, Result: {}",
                    idx,
                    game.fullmoves(),
                    game.outcome()
                );
            }
            if trainer.step_counter() {
                trainer.train(&mut optim, lr);
                train_counter += 1;
            }

            device.memory_cleanup();
        }

        // run evaluation against random
        if (epoch + 1) % 10 == 0 {
            let mut wins = 0;
            let mut draws = 0;
            let mut games = vec![Chess::new(); 10];
            let mut rng = rand::rng();
            while !games.is_empty() {
                let output = trainer.make_move(&games, 0.);
                for (idx, game) in games.iter_mut().enumerate() {
                    game.play_unchecked(output[idx]);
                    if let Outcome::Known(_) = game.outcome() {
                        continue;
                    }
                    let black_action = *game.legal_moves().choose(&mut rng).unwrap();
                    game.play_unchecked(black_action);
                    if let Outcome::Known(_) = game.outcome() {
                        continue;
                    }
                }
                games.retain(|game| {
                    if let Outcome::Known(outcome) = game.outcome() {
                        match outcome {
                            KnownOutcome::Decisive {
                                winner: Color::White,
                            } => {
                                wins += 1;
                            }
                            KnownOutcome::Decisive {
                                winner: Color::Black,
                            } => {}
                            KnownOutcome::Draw => {
                                draws += 1;
                            }
                        }
                        false
                    } else {
                        true
                    }
                });
            }
            let loses = 10 - wins - draws;
            println!(
                "Against random play: {:2} wins, {:2} draws, {:2} loses",
                wins, draws, loses,
            );
        }
    }

    let model_path = "model";
    trainer
        .model()
        .clone()
        .save_file(model_path)
        .expect("Should be able to save the model");
}
