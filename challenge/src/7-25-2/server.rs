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
    zero: FheUint8,
    one: FheUint8, 
    two: FheUint8,
    three: FheUint8,
}

impl Server {
    pub(crate) fn new(server_key: ServerKey, grid: EncryptedGrid) -> Self {
        let arc_key = Arc::new(server_key);
        set_server_key((*arc_key).clone());
        let zero = FheUint8::try_encrypt_trivial(0u8).unwrap();
        let one = FheUint8::try_encrypt_trivial(1u8).unwrap();
        let two = FheUint8::try_encrypt_trivial(2u8).unwrap();
        let three = FheUint8::try_encrypt_trivial(3u8).unwrap();

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
            zero,
            one,
            two,
            three,
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

    // 用加法树减少加密电路深度
    fn add_tree(&self, items: &[&FheUint8]) -> FheUint8 {
        if items.len() == 1 {
            items[0].clone()
        } else {
            let mid = items.len() / 2;
            &self.add_tree(&items[..mid]) + &self.add_tree(&items[mid..])
        }
    }

    fn update_cell(&self, x: usize, y: usize, grid: &EncryptedGrid) -> FheUint8 {
        let zero = &self.zero;
        let one = &self.one;
        let two = &self.two;
        let three = &self.three;
        let cell = &grid[x][y];

        //let mut count = FheUint8::try_encrypt_trivial(0u8).unwrap();
        //let mut count = self.zero.clone();

        let min_x = x.saturating_sub(1);
        let max_x = (x + 2).min(grid.len());
        let min_y = y.saturating_sub(1);
        let max_y = (y + 2).min(grid[x].len());

        let mut neighbors = vec![];
        for nx in min_x..max_x {
            for ny in min_y..max_y {
                if nx == x && ny == y {
                    continue;
                }
                neighbors.push(&grid[nx][ny]);
            }
        }

        let count = self.add_tree(&neighbors);


        //let zero = FheUint8::try_encrypt_trivial(0u8).unwrap();
        //let one = FheUint8::try_encrypt_trivial(1u8).unwrap();
        //let two = FheUint8::try_encrypt_trivial(2u8).unwrap();
        //let three = FheUint8::try_encrypt_trivial(3u8).unwrap();

        let alive = cell.eq(one);
        let is_three = count.eq(three);
        let is_two = count.eq(two);
        let eq_three = is_three.select(one, zero);
        let eq_two_or_three = (is_two | is_three).select(one, zero);


        alive.if_then_else(&eq_two_or_three, &eq_three)
    }
}
