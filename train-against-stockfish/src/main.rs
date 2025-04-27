use burn::backend::{Autodiff, Wgpu};
use qchess_bot::{
    agent::Agent,
    model::{Model, ModelConfig},
};
use shakmaty::{Chess, Position, fen::Fen, uci::UciMove};
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

    for _epoch in 0..10 {
        while agent.get_memory_len() < 50 {
            let mut chess = Chess::new();
            let mut game = agent.start_new_game(chess.clone(), &device);
            let trajectory = loop {
                let action = game.make_action();
                chess.play_unchecked(&action);
                if chess.is_game_over() {
                    break game.game_end();
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
                chess.play_unchecked(&engine_move);
                let reward = 0.0;
                game.get_feedback(chess.clone(), reward);
                if chess.is_game_over() {
                    break game.game_end();
                }
            };
            println!("{:?}", chess.outcome());
            agent.collect_trajectory(trajectory);
        }
        println!("{}", agent.get_memory_len());
        agent.train_model(&device);
    }
}
