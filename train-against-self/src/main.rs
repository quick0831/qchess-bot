use burn::{
    backend::{Autodiff, Wgpu},
    optim::AdamConfig,
};
use qchess_bot::{
    agent::{Agent, GameSession},
    model::{Model, ModelConfig},
};
use shakmaty::{Chess, Color, Outcome, Position, san::SanPlus};

fn main() {
    type MyBackend = Autodiff<Wgpu<f32, i32>>;

    let model_config = ModelConfig {};

    let device = Default::default();
    let model: Model<MyBackend> = model_config.init(&device);

    println!("{model}");

    let mut agent = Agent::new(model);

    let mut optim = AdamConfig::new().init();
    let lr = 0.0001;
    for _epoch in 0..200 {
        while agent.get_memory_len() < 200 {
            let mut chess = Chess::new();
            let mut game_white = agent.start_new_game(chess.clone(), &device);
            let mut game_black: Option<GameSession<_>> = None;
            let mut white_reward;
            let mut black_reward = 0.0;
            let mut black_move = None;
            let (white_trajectory, black_trajectory, outcome) = loop {
                white_reward = 0.1 + -0.002 * chess.fullmoves().get() as f32; // punish for making the game long
                let white_action = game_white.make_action();
                let white_move = SanPlus::from_move(chess.clone(), &white_action);
                if white_action.is_capture() {
                    // reward for capturing
                    white_reward += 0.05;
                    // punish for being captured
                    black_reward -= 0.05;
                }
                chess.play_unchecked(&white_action);
                if chess.is_check() {
                    // reward for checking the opposing king
                    white_reward += 0.05;
                    // punish for being checked
                    black_reward -= 0.05;
                }
                if let Some(black_move) = black_move {
                    println!(
                        "{:4}. b\t{}\treward: {:+.4}",
                        chess.fullmoves().get() - 1,
                        black_move,
                        black_reward,
                    );
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
                black_move = Some(SanPlus::from_move(chess.clone(), &black_action));
                // reward for capturing
                if black_action.is_capture() {
                    black_reward += 0.05;
                }
                if chess.is_check() {
                    // punish for being checked
                    white_reward -= 0.05;
                    // reward for checking the opposing king
                    black_reward += 0.05;
                }
                println!(
                    "{:4}. w\t{}\treward: {:+.4}",
                    chess.fullmoves().get() - 1,
                    white_move,
                    white_reward,
                );
                game_white.get_feedback(chess.clone(), white_reward);
                if let Some(outcome) = chess.outcome() {
                    break (
                        game_white.game_end(),
                        Some(game_black.unwrap().game_end()),
                        outcome,
                    );
                }
            };
            println!("{:?}", outcome);
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
        println!("{}", agent.get_memory_len());
        agent.train_model(&device, &mut optim, lr);
    }
}
