use burn::{
    backend::{Autodiff, Wgpu},
    optim::SgdConfig,
};
use qchess_bot::{
    agent::Agent,
    model::{Model, ModelConfig},
};
use shakmaty::{Chess, Color, Outcome, Position, fen::Fen, san::SanPlus, uci::UciMove};
use uciengine::uciengine::*;

#[tokio::main(flavor = "current_thread")]
async fn main() {
    type MyBackend = Autodiff<Wgpu<f32, i32>>;

    let model_config = ModelConfig {};

    let device = Default::default();
    let model: Model<MyBackend> = model_config.init(&device);

    println!("{model}");

    let engine = UciEngine::new("/home/quick/stockfish/stockfish-ubuntu-x86-64-avx2");

    let mut agent = Agent::new(model);

    let mut optim = SgdConfig::new().init();
    let lr = 0.00001;
    for _epoch in 0..10 {
        while agent.get_memory_len() < 50 {
            let mut chess = Chess::new();
            let mut game = agent.start_new_game(chess.clone(), &device);
            let (trajectory, outcome) = loop {
                let mut reward = -0.002; // punish for making the game long
                let action = game.make_action();
                // reward for capturing
                if action.is_capture() {
                    reward += 0.05;
                }
                let white_move = SanPlus::from_move(chess.clone(), &action);
                chess.play_unchecked(&action);
                // reward for checking the opposing king
                if let Some(outcome) = chess.outcome() {
                    break (game.game_end(), outcome);
                }
                if chess.is_check() {
                    reward += 0.05;
                }
                let fen =
                    Fen::from_setup(chess.clone().into_setup(shakmaty::EnPassantMode::Always))
                        .to_string();
                let go_job = GoJob::new()
                    .uci_opt("UCI_LimitStrength", "true")
                    .pos_fen(fen)
                    .go_opt("movetime", "500");
                let engine_move = engine.go(go_job).await.unwrap();
                let engine_move: UciMove = engine_move.bestmove.unwrap().parse().unwrap();
                let engine_move = engine_move.to_move(&chess).unwrap();
                let black_move = SanPlus::from_move(chess.clone(), &engine_move);
                chess.play_unchecked(&engine_move);
                // punish for being checked
                if chess.is_check() {
                    reward -= 0.05;
                }
                println!(
                    "{:5}.\t{}\t{}\treward: {:+.4}",
                    chess.fullmoves().get() - 1,
                    white_move,
                    black_move,
                    reward,
                );
                game.get_feedback(chess.clone(), reward);
                if let Some(outcome) = chess.outcome() {
                    break (game.game_end(), outcome);
                }
            };
            println!("{:?}", chess.outcome());
            let final_reward = match outcome {
                Outcome::Decisive {
                    winner: Color::White,
                } => 1.0,
                Outcome::Decisive {
                    winner: Color::Black,
                } => -1.0,
                Outcome::Draw => -0.1,
            };
            agent.collect_trajectory(trajectory, final_reward);
        }
        println!("{}", agent.get_memory_len());
        agent.train_model(&device, &mut optim, lr);
    }
}
