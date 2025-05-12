use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use qchess_bot::{
    agent::{Agent, GameSession},
    model::{Model, ModelConfig},
};
use rand::seq::IndexedRandom;
use shakmaty::{Chess, Color, Outcome, Position};

fn main() {
    type MyBackend = Autodiff<Wgpu<f32, i32>>;

    let model_config = ModelConfig {};

    let device = Default::default();
    let model: Model<MyBackend> = model_config.init(&device);

    println!("{model}");

    let mut agent = Agent::new(model);

    let mut optim = AdamConfig::new().init();
    let lr = 0.0001;
    for epoch in 0..2000 {
        println!("epoch: {:5}", epoch + 1);
        while agent.get_memory_len() < 200 {
            let mut chess = Chess::new();
            let mut game_white = agent.start_new_game(chess.clone(), &device);
            let mut game_black: Option<GameSession<_>> = None;
            let mut white_reward;
            let mut black_reward = 0.0;
            let (white_trajectory, black_trajectory, outcome) = loop {
                white_reward = 0.1 + -0.002 * chess.fullmoves().get() as f32; // punish for making the game long
                let white_action = game_white.make_action();
                if white_action.is_capture() {
                    // reward for capturing
                    white_reward += 0.05;
                    // punish for being captured
                    black_reward -= 0.05;
                }
                chess.play_unchecked(&white_action);
                if chess.is_check() {
                    // reward for checking the opposing king
                    white_reward += 0.2;
                    // punish for being checked
                    black_reward -= 0.2;
                }
                if let Some(ref mut game_black) = game_black {
                    game_black.get_feedback(chess.clone(), black_reward);
                }
                if let Some(outcome) = chess.outcome() {
                    break (
                        game_white.game_end(),
                        game_black.map(|g| g.game_end()),
                        outcome,
                    );
                }

                black_reward = 0.1 + -0.002 * chess.fullmoves().get() as f32; // punish for making the game long
                if game_black.is_none() {
                    game_black = Some(agent.start_new_game(chess.clone(), &device));
                }
                let black_action = game_black.as_mut().unwrap().make_action();
                if black_action.is_capture() {
                    // reward for capturing
                    black_reward += 0.05;
                    // punish for being captured
                    white_reward -= 0.05;
                }
                chess.play_unchecked(&black_action);
                // reward for capturing
                if black_action.is_capture() {
                    black_reward += 0.05;
                }
                if chess.is_check() {
                    // punish for being checked
                    white_reward -= 0.2;
                    // reward for checking the opposing king
                    black_reward += 0.2;
                }
                game_white.get_feedback(chess.clone(), white_reward);
                if let Some(outcome) = chess.outcome() {
                    break (
                        game_white.game_end(),
                        Some(game_black.unwrap().game_end()),
                        outcome,
                    );
                }
                // Stop the session if the game is too long
                if chess.fullmoves().get() > 250 {
                    break (
                        game_white.game_end(),
                        Some(game_black.unwrap().game_end()),
                        Outcome::Draw,
                    );
                }
            };
            println!("Fullmoves: {:3}, Result: {}", chess.fullmoves(), outcome);
            let (white_final_reward, black_final_reward) = match outcome {
                Outcome::Decisive {
                    winner: Color::White,
                } => (1.0, -1.0),
                Outcome::Decisive {
                    winner: Color::Black,
                } => (-1.0, 1.0),
                Outcome::Draw => (-0.1, -0.1),
            };
            agent.collect_trajectory(white_trajectory, white_final_reward);
            if let Some(black_trajectory) = black_trajectory {
                agent.collect_trajectory(black_trajectory, black_final_reward);
            }
        }
        agent.train_model(&device, &mut optim, lr);

        // run evaluation against random
        if (epoch + 1) % 10 == 0 {
            let mut wins = 0;
            let mut draws = 0;
            let mut games = vec![Chess::new(); 10];
            let mut rng = rand::rng();
            while !games.is_empty() {
                let output = agent.inference(&games, &device);
                for (idx, game) in games.iter_mut().enumerate() {
                    game.play_unchecked(&output[idx]);
                    if game.outcome().is_some() {
                        continue;
                    }
                    let black_action = game.legal_moves().choose(&mut rng).unwrap().clone();
                    game.play_unchecked(&black_action);
                }
                games.retain(|game| {
                    if let Some(outcome) = game.outcome() {
                        match outcome {
                            Outcome::Decisive {
                                winner: Color::White,
                            } => {
                                wins += 1;
                            }
                            Outcome::Decisive {
                                winner: Color::Black,
                            } => {}
                            Outcome::Draw => {
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
}
