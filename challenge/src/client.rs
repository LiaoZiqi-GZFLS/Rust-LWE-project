use rand::Rng;
use tfhe::prelude::*;
use tfhe::{ClientKey, ConfigBuilder, FheUint8, ServerKey};
use tfhe::shortint::parameters::current_params::V1_3_PARAM_MULTI_BIT_GROUP_3_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64;

pub(crate) type EncryptedGrid = Vec<Vec<FheUint8>>;

pub(crate) struct Client {
    client_key: ClientKey,
    server_key: ServerKey,
    grid: Vec<Vec<u8>>,
}

impl Client {
    // Create a new client with a grid of size m * n
    pub(crate) fn new(m: u32, n: u32) -> Self {
        let config = if m < 3 && n < 3 {
            ConfigBuilder::default().use_custom_parameters(V1_3_PARAM_MULTI_BIT_GROUP_3_MESSAGE_2_CARRY_2_KS_PBS_GAUSSIAN_2M64).build()
        } else {
            ConfigBuilder::default().build()
        };
        //let config = ConfigBuilder::default().build();
        let client_key = ClientKey::generate(config);
        let server_key = ServerKey::new(&client_key);

        // Initial state
        let mut rng = rand::rng();
        let grid = (0..m)
            .map(|_| (0..n).map(|_| rng.random_range(0..=1)).collect::<Vec<u8>>())
            .collect::<Vec<Vec<u8>>>();

        Client {
            client_key,
            server_key,
            grid,
        }
    }

    /// Encrypt every instance
    ///
    /// # Returns
    /// A tuple containing the server key and the encrypted grid.
    pub(crate) fn encrypt(&self) -> (ServerKey, EncryptedGrid) {
        let encrypted_grid = self
            .grid
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&cell| FheUint8::encrypt(cell, &self.client_key))
                    .collect::<Vec<FheUint8>>()
            })
            .collect::<Vec<Vec<FheUint8>>>();
        let server_key = ServerKey::new(&self.client_key);

        (server_key, encrypted_grid)
    }

    /// Verify the encrypted grid against the expected state after a number of steps
    /// # Arguments
    /// * `encrypted_grid` - The encrypted grid to verify.
    /// * `steps` - The number of steps to simulate.
    /// # Returns
    /// A boolean indicating whether the verification was successful.
    pub(crate) fn verify(&self, encrypted_grid: EncryptedGrid, steps: u32) -> bool {
        let decrypted_grid = self.decrypt(encrypted_grid);
        let expected_grid = self.grid_after_steps(steps);

        for i in 0..self.grid.len() {
            for j in 0..self.grid[i].len() {
                if expected_grid[i][j] != decrypted_grid[i][j] {
                    return false;
                }
            }
        }

        true
    }

    fn decrypt(&self, encrypted_grid: EncryptedGrid) -> Vec<Vec<u8>> {
        encrypted_grid
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| FheUint8::decrypt(cell, &self.client_key))
                    .collect::<Vec<u8>>()
            })
            .collect::<Vec<Vec<u8>>>()
    }

    fn grid_after_steps(&self, steps: u32) -> Vec<Vec<u8>> {
        let mut current_grid = self.grid.clone();
        for _ in 0..steps {
            current_grid = self.next_generation(&current_grid);
        }
        current_grid
    }

    fn next_generation(&self, grid: &Vec<Vec<u8>>) -> Vec<Vec<u8>> {
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
