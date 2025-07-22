use crate::client::EncryptedGrid;
use rayon::ThreadPoolBuilder;
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use tfhe::prelude::*;
use tfhe::{FheUint8, ServerKey, set_server_key};
use std::sync::Arc;

pub(crate) struct Server {
    server_key: ServerKey,
    grid: EncryptedGrid,
    thread_pool: Arc<rayon::ThreadPool>,
}

impl Server {
    pub(crate) fn new(server_key: ServerKey, grid: EncryptedGrid) -> Self {
        let thread_pool = Arc::new(
            ThreadPoolBuilder::new()
                .num_threads(num_cpus::get())
                .build()
                .unwrap(),
        );

        Server {
            server_key,
            grid,
            thread_pool,
        }
    }

    pub(crate) fn run(&self, steps: u32) -> EncryptedGrid {
        let mut current_grid = self.grid.clone();
        for _ in 0..steps {
            current_grid = self.step(&current_grid);
        }
        current_grid
    }

    fn step(&self, grid: &EncryptedGrid) -> EncryptedGrid {
        let grid = Arc::new(grid.clone());
        let server_key = self.server_key.clone();
        let thread_pool = Arc::clone(&self.thread_pool); // 获取线程池引用
        
        // 使用线程池执行并行任务
        let result: Vec<_> = thread_pool.install(|| {
            (0..grid.len())
                .into_par_iter()
                .map(|i| {
                    let mut row = vec![];
                    for j in 0..grid[i].len() {
                        let local_server_key = server_key.clone();
                        set_server_key(local_server_key);
                        row.push(update_cell(i, j, &grid));
                    }
                    row
                })
                .collect()
        });
        
        result
    }
}

fn update_cell(x: usize, y: usize, grid: &EncryptedGrid) -> FheUint8 {
    // 保持原有逻辑不变
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