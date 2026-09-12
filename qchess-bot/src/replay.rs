use std::{collections::VecDeque, ops::Index};

use rand::seq::IndexedRandom;
use shakmaty::Chess;

use crate::encode::UciMoveId;

#[derive(Debug, Clone)]
pub enum GameFlag {
    // The game will continue
    Nothing,
    // The game has reached the end
    Terminated,
    // The game is stopped by external factor
    Truncated,
}

#[derive(Debug, Clone)]
pub struct Transition {
    pub state: Chess,
    pub action: UciMoveId,
    pub reward: f32,
    pub flag: GameFlag,
}

#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    buffer: VecDeque<Transition>,
    buffer_size: usize,
}

impl ReplayBuffer {
    pub fn new(buffer_size: usize) -> Self {
        let mut buffer = VecDeque::new();
        buffer.reserve_exact(buffer_size);
        Self {
            buffer,
            buffer_size,
        }
    }

    pub fn push(&mut self, transition: Transition) {
        if self.buffer.len() >= self.buffer_size {
            self.buffer.pop_front();
        }
        self.buffer.push_back(transition);
    }
}

impl IndexedRandom for ReplayBuffer {
    fn len(&self) -> usize {
        self.buffer.len()
    }
}

impl Index<usize> for ReplayBuffer {
    type Output = Transition;

    fn index(&self, index: usize) -> &Self::Output {
        &self.buffer[index]
    }
}
