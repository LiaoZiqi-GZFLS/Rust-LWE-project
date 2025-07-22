use rand::Rng;

pub(crate) type Grid = Vec<Vec<u8>>;

pub(crate) struct Client {
    grid: Grid,
}

impl Client {
    // Create a new client with a grid of size m * n
    pub(crate) fn new(m: u32, n: u32) -> Self {
        let mut rng = rand::rng();
        let grid = (0..m)
           .map(|_| (0..n).map(|_| rng.random_range(0..=1)).collect::<Vec<u8>>())
           .collect::<Vec<Vec<u8>>>();

        Client {
            grid,
        }
    }

    // 添加一个公共方法来获取 grid
    pub(crate) fn get_grid(&self) -> Grid {
        self.grid.clone()
    }

    /// Verify the grid against the expected state after a number of steps
    /// # Arguments
    /// * `grid` - The grid to verify.
    /// * `steps` - The number of steps to simulate.
    /// # Returns
    /// A boolean indicating whether the verification was successful.
    pub(crate) fn verify(&self, grid: Grid, steps: u32) -> bool {
        let expected_grid = self.grid_after_steps(steps);

        for i in 0..self.grid.len() {
            for j in 0..self.grid[i].len() {
                if expected_grid[i][j] != grid[i][j] {
                    return false;
                }
            }
        }

        true
    }

    fn grid_after_steps(&self, steps: u32) -> Grid {
        let mut current_grid = self.grid.clone();
        for _ in 0..steps {
            current_grid = self.next_generation(&current_grid);
        }
        current_grid
    }

    fn next_generation(&self, grid: &Grid) -> Grid {
        let mut new_grid = grid.clone();
        let directions = [
            (-1, -1),
            (-1, 0),
            (-1, 1),
            (0, -1),
            (0, 1),
            (1, -1),
            (1, 0),
            (1, 1),
        ];

        for i in 0..grid.len() {
            for j in 0..grid[i].len() {
                let mut live_neighbors = 0;
                for &(dx, dy) in &directions {
                    let ni = i as isize + dx;
                    let nj = j as isize + dy;
                    if ni >= 0 && ni < grid.len() as isize && nj >= 0 && nj < grid[i].len() as isize
                    {
                        live_neighbors += grid[ni as usize][nj as usize];
                    }
                }

                if grid[i][j] == 1 {
                    new_grid[i][j] = if live_neighbors < 2 || live_neighbors > 3 {
                        0
                    } else {
                        1
                    };
                } else {
                    new_grid[i][j] = if live_neighbors == 3 { 1 } else { 0 };
                }
            }
        }

        new_grid
    }
}