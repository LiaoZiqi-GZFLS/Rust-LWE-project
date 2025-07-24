use crate::client::EncryptedGrid;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::sync::Arc;
use tfhe::prelude::*;
use tfhe::{FheUint8, ServerKey, set_server_key};

pub(crate) struct Server {
    server_key: Arc<ServerKey>,
    grid: EncryptedGrid,
}

impl Server {
    pub(crate) fn new(server_key: ServerKey, grid: EncryptedGrid) -> Self {
        Server {
            server_key: Arc::new(server_key),
            grid,
        }
    }

    pub(crate) fn run(&self, steps: u32) -> EncryptedGrid {
        let mut current = self.grid.clone();
        for _ in 0..steps {
            current = self.step(&current);
        }
        current
    }

    fn step(&self, grid: &EncryptedGrid) -> EncryptedGrid {
        let sk = Arc::clone(&self.server_key);

        ThreadPoolBuilder::new()
            .start_handler(move |_| {
                // ✅ 每个线程启动时设置一次 ServerKey
                set_server_key((*sk).clone());
            })
            .build()
            .unwrap()
            .install(|| {
                (0..grid.len())
                    .into_par_iter()
                    .map(|i| {
                        (0..grid[i].len())
                            .map(|j| self.update_cell(i, j, grid))
                            .collect::<Vec<FheUint8>>()
                    })
                    .collect()
            })
    }

    fn update_cell(&self, x: usize, y: usize, grid: &EncryptedGrid) -> FheUint8 {
        let mut count = FheUint8::try_encrypt_trivial(0u8).unwrap();
        for dx in [-1isize, 0, 1].iter() {
            for dy in [-1isize, 0, 1].iter() {
                if *dx == 0 && *dy == 0 {
                    continue;
                }
                let nx = x.wrapping_add(*dx as usize);
                let ny = y.wrapping_add(*dy as usize);
                if nx < grid.len() && ny < grid[nx].len() {
                    count += grid[nx][ny].clone();
                }
            }
        }

        let cell = &grid[x][y];
        let zero = FheUint8::try_encrypt_trivial(0u8).unwrap();
        let one = FheUint8::try_encrypt_trivial(1u8).unwrap();
        let two = FheUint8::try_encrypt_trivial(2u8).unwrap();
        let three = FheUint8::try_encrypt_trivial(3u8).unwrap();

        let alive = cell.eq(&one);
        let eq_three = count.eq(&three).select(&one, &zero);
        let eq_two_or_three = (count.eq(&two) | count.eq(&three)).select(&one, &zero);

        alive.if_then_else(&eq_two_or_three, &eq_three)
    }
}
