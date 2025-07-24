use crate::client::EncryptedGrid;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::sync::Arc;
use tfhe::prelude::*;
use tfhe::{FheUint8, ServerKey, set_server_key};

pub(crate) struct Server {
    // server_key: Arc<ServerKey>,
    grid: EncryptedGrid,
    pool: Arc<ThreadPool>,
}

impl Server {
    pub(crate) fn new(server_key: ServerKey, grid: EncryptedGrid) -> Self {
        let arc_key = Arc::new(server_key);

        // ✅ 构建线程池并设置每个线程都 set_server_key
        let key_clone = arc_key.clone();
        let pool = ThreadPoolBuilder::new()
            .start_handler(move |_| {
                set_server_key((*key_clone).clone());
            })
            .build()
            .unwrap();

        Server {
            // server_key: arc_key,
            grid,
            pool: Arc::new(pool),
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
        // ✅ 重用线程池（无需每步重新构建）
        self.pool.install(|| {
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
