use shakmaty::{
    CastlingMode, File, Move, Position, Rank, Role, Square,
    uci::{IllegalUciMoveError, UciMove},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UciMoveId(u16);

impl UciMoveId {
    /// The total amount of possible UciMoveId
    pub const TOTAL: u16 = 1880;

    pub const fn u16(&self) -> u16 {
        self.0
    }

    pub const fn new(id: u16) -> Option<UciMoveId> {
        if id < Self::TOTAL {
            Some(Self(id))
        } else {
            None
        }
    }

    pub fn from_move(m: &Move) -> Self {
        UciMoveId::from_uci(m.to_uci(CastlingMode::Standard)).expect("Invalid move")
    }

    pub fn to_move(&self, pos: &impl Position) -> Result<Move, IllegalUciMoveError> {
        self.to_uci().to_move(pos)
    }

    pub const fn from_uci(uci: UciMove) -> Option<Self> {
        let UciMove::Normal {
            from,
            to,
            promotion,
        } = uci
        else {
            return None;
        };

        let id = if let Some(promotion) = promotion {
            let promotion_id = match promotion {
                Role::Knight => 0,
                Role::Bishop => 1,
                Role::Rook => 2,
                Role::Queen => 3,
                _ => unreachable!(),
            };
            let from_file = from.file();
            let file_id = match (from_file as i8) - (to.file() as i8) {
                // promotion
                0 => 0,
                // take promote towards king-side
                ..0 => 8,
                // take promote towards queen-side
                1.. => 14,
            } + from_file as u16;
            1792 + 4 * file_id + promotion_id
        } else {
            let ff = from.file() as u16;
            let fr = from.rank() as u16;
            let tf = to.file() as u16;
            let tr = to.rank() as u16;
            let fd = tf as i8 - ff as i8;
            let rd = tr as i8 - fr as i8;
            match (fd, rd) {
                (0, 0) => return None,
                (_, 0) => ff * 56 + fr * 7 + tf - if fd >= 0 { 1 } else { 0 },
                (0, _) => ff * 56 + fr * 7 + tr - if rd >= 0 { 1 } else { 0 } + 448,
                (_, _) if fd == rd => {
                    let fv = (ff + fr) / 2;
                    let tv = (tf + tr) / 2 - if rd >= 0 { 1 } else { 0 };
                    let (off, l) = match ff as i8 - fr as i8 {
                        -6 => (890, 1),
                        -5 => (892, 2),
                        -4 => (896, 3),
                        -3 => (911, 4),
                        -2 => (930, 5),
                        -1 => (966, 6),
                        0 => (1008, 7),
                        1 => (1064, 6),
                        2 => (1100, 5),
                        3 => (1131, 4),
                        4 => (1148, 3),
                        5 => (1162, 2),
                        6 => (1168, 1),
                        _ => unreachable!(),
                    };
                    off + fv * l + tv
                }
                (_, _) if fd == -rd => {
                    let fv = (fr + 7 - ff) / 2;
                    let tv = (tr + 7 - tf) / 2 - if rd >= 0 { 1 } else { 0 };
                    let (off, l) = match ff + fr {
                        13 => (1170, 1),
                        12 => (1172, 2),
                        11 => (1176, 3),
                        10 => (1191, 4),
                        9 => (1210, 5),
                        8 => (1246, 6),
                        7 => (1288, 7),
                        6 => (1344, 6),
                        5 => (1380, 5),
                        4 => (1411, 4),
                        3 => (1428, 3),
                        2 => (1442, 2),
                        1 => (1448, 1),
                        _ => unreachable!(),
                    };
                    off + fv * l + tv
                }
                (1, 2) => 1456 + ff * 6 + fr,
                (2, 1) => 1498 + ff * 7 + fr,
                (2, -1) => 1539 + ff * 7 + fr,
                (1, -2) => 1580 + ff * 6 + fr,
                (-1, -2) => 1616 + ff * 6 + fr,
                (-2, -1) => 1651 + ff * 7 + fr,
                (-2, 1) => 1694 + ff * 7 + fr,
                (-1, 2) => 1744 + ff * 6 + fr,
                _ => return None,
            }
        };

        Some(Self(id))
    }

    pub const fn to_uci(&self) -> UciMove {
        let id = self.0;

        const fn diag(id: u16, len: u16, f: u32, r: u32) -> UciMove {
            let x = id / len;
            let y = id % len;
            let y = y + if y >= x { 1 } else { 0 };
            let file = File::new(x as u32 + f);
            let rank = Rank::new(x as u32 + r);
            let from = Square::from_coords(file, rank);
            let to_file = File::new(y as u32 + f);
            let to_rank = Rank::new(y as u32 + r);
            let to = Square::from_coords(to_file, to_rank);
            UciMove::Normal {
                from,
                to,
                promotion: None,
            }
        }

        const fn diag2(id: u16, len: u16, f: u32, r: u32) -> UciMove {
            let x = id / len;
            let y = id % len;
            let y = y + if y >= x { 1 } else { 0 };
            let file = File::new(f - x as u32);
            let rank = Rank::new(x as u32 + r);
            let from = Square::from_coords(file, rank);
            let to_file = File::new(f - y as u32);
            let to_rank = Rank::new(y as u32 + r);
            let to = Square::from_coords(to_file, to_rank);
            UciMove::Normal {
                from,
                to,
                promotion: None,
            }
        }

        const fn knight(id: u16, l: i16, f: i16, r: i16) -> UciMove {
            let id = id as i16;
            let x = id / l - if f < 0 { f } else { 0 };
            let y = id % l - if r < 0 { r } else { 0 };
            let file = File::new(x as u32);
            let rank = Rank::new(y as u32);
            let from = Square::from_coords(file, rank);
            let to_file = File::new((x + f) as u32);
            let to_rank = Rank::new((y + r) as u32);
            let to = Square::from_coords(to_file, to_rank);
            UciMove::Normal {
                from,
                to,
                promotion: None,
            }
        }

        match id {
            0..448 => {
                let sq = (id / 7) as u32;
                let sub = id % 7;
                let rank = Rank::new(sq % 8);
                let file = File::new(sq / 8);
                let from = Square::from_coords(file, rank);
                let to_file = (sub + if sub >= file as u16 { 1 } else { 0 }) as u32;
                let to = Square::from_coords(File::new(to_file), rank);
                UciMove::Normal {
                    from,
                    to,
                    promotion: None,
                }
            }
            448..896 => {
                let id = id - 448;
                let sq = (id / 7) as u32;
                let sub = id % 7;
                let rank = Rank::new(sq % 8);
                let file = File::new(sq / 8);
                let from = Square::from_coords(file, rank);
                let to_rank = (sub + if sub >= rank as u16 { 1 } else { 0 }) as u32;
                let to = Square::from_coords(file, Rank::new(to_rank));
                UciMove::Normal {
                    from,
                    to,
                    promotion: None,
                }
            }
            896..898 => diag(id - 896, 1, 0, 6),
            898..904 => diag(id - 898, 2, 0, 5),
            904..916 => diag(id - 904, 3, 0, 4),
            916..936 => diag(id - 916, 4, 0, 3),
            936..966 => diag(id - 936, 5, 0, 2),
            966..1008 => diag(id - 966, 6, 0, 1),
            1008..1064 => diag(id - 1008, 7, 0, 0),
            1064..1106 => diag(id - 1064, 6, 1, 0),
            1106..1136 => diag(id - 1106, 5, 2, 0),
            1136..1156 => diag(id - 1136, 4, 3, 0),
            1156..1168 => diag(id - 1156, 3, 4, 0),
            1168..1174 => diag(id - 1168, 2, 5, 0),
            1174..1176 => diag(id - 1174, 1, 6, 0),
            1176..1178 => diag2(id - 1176, 1, 7, 6),
            1178..1184 => diag2(id - 1178, 2, 7, 5),
            1184..1196 => diag2(id - 1184, 3, 7, 4),
            1196..1216 => diag2(id - 1196, 4, 7, 3),
            1216..1246 => diag2(id - 1216, 5, 7, 2),
            1246..1288 => diag2(id - 1246, 6, 7, 1),
            1288..1344 => diag2(id - 1288, 7, 7, 0),
            1344..1386 => diag2(id - 1344, 6, 6, 0),
            1386..1416 => diag2(id - 1386, 5, 5, 0),
            1416..1436 => diag2(id - 1416, 4, 4, 0),
            1436..1448 => diag2(id - 1436, 3, 3, 0),
            1448..1454 => diag2(id - 1448, 2, 2, 0),
            1454..1456 => diag2(id - 1454, 1, 1, 0),
            1456..1498 => knight(id - 1456, 6, 1, 2),
            1498..1540 => knight(id - 1498, 7, 2, 1),
            1540..1582 => knight(id - 1540, 7, 2, -1),
            1582..1624 => knight(id - 1582, 6, 1, -2),
            1624..1666 => knight(id - 1624, 6, -1, -2),
            1666..1708 => knight(id - 1666, 7, -2, -1),
            1708..1750 => knight(id - 1708, 7, -2, 1),
            1750..1792 => knight(id - 1750, 6, -1, 2),
            1792..1880 => {
                let id = id - 1792;
                let promotion_id = id % 4;
                let promotion = match promotion_id {
                    0 => Role::Knight,
                    1 => Role::Bishop,
                    2 => Role::Rook,
                    3 => Role::Queen,
                    _ => unreachable!(),
                };
                let file_id = id / 4;
                let (from_file, to_file) = match file_id {
                    0..8 => (file_id, file_id),
                    8..15 => (file_id - 8, file_id - 7),
                    15..22 => (file_id - 14, file_id - 15),
                    22.. => unreachable!(),
                };
                let from_file = File::new(from_file as u32);
                let to_file = File::new(to_file as u32);
                UciMove::Normal {
                    from: Square::from_coords(from_file, Rank::Seventh),
                    to: Square::from_coords(to_file, Rank::Eighth),
                    promotion: Some(promotion),
                }
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uci_match() {
        for n in 0..UciMoveId::TOTAL {
            let id = UciMoveId::new(n).unwrap();
            let id2 = UciMoveId::from_uci(id.to_uci()).unwrap();
            assert_eq!(id, id2);
        }
    }
}
