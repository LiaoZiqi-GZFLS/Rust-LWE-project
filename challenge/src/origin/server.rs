use crate::client::EncryptedGrid;
use tfhe::prelude::*;
use tfhe::{FheUint8, ServerKey, set_server_key};

pub(crate) struct Server {
    server_key: ServerKey,
    grid: EncryptedGrid,
}

impl Server {
    pub(crate) fn new(server_key: ServerKey, grid: EncryptedGrid) -> Self {
        Server { server_key, grid }
    }

    pub(crate) fn run(&self, steps: u32) -> EncryptedGrid {
        let mut current_grid = self.grid.clone();
        for _ in 0..steps {
            current_grid = self.step(&current_grid);
        }
        current_grid
    }

    fn step(&self, grid: &EncryptedGrid) -> EncryptedGrid {
        let mut new_grid = vec![];
        for i in 0..self.grid.len() {
            let mut row = vec![];
            for j in 0..self.grid[i].len() {
                row.push(self.update_cell(i, j, grid));
            }
            new_grid.push(row);
        }
        new_grid
    }

    fn update_cell(&self, x: usize, y: usize, grid: &EncryptedGrid) -> FheUint8 {
        set_server_key(self.server_key.clone());

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
