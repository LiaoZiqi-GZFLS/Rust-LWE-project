use crate::client::Grid;

pub(crate) struct Server {
    grid: Grid,
}

impl Server {
    pub(crate) fn new(grid: Grid) -> Self {
        Server { grid }
    }

    pub(crate) fn run(&self, steps: u32) -> Grid {
        let mut current_grid = self.grid.clone();
        for _ in 0..steps {
            current_grid = self.step(&current_grid);
        }
        current_grid
    }

    fn step(&self, grid: &Grid) -> Grid {
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

    fn update_cell(&self, x: usize, y: usize, grid: &Grid) -> u8 {
        let mut count = 0;
        for dx in [-1isize, 0, 1].iter() {
            for dy in [-1isize, 0, 1].iter() {
                if *dx == 0 && *dy == 0 {
                    continue;
                }

                let nx = x.wrapping_add(*dx as usize);
                let ny = y.wrapping_add(*dy as usize);

                if nx < grid.len() && ny < grid[nx].len() {
                    count += grid[nx][ny];
                }
            }
        }

        let cell = grid[x][y];
        if cell == 1 {
            if count < 2 || count > 3 {
                0
            } else {
                1
            }
        } else {
            if count == 3 {
                1
            } else {
                0
            }
        }
    }
}