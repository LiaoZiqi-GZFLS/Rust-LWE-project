#[allow(dead_code, unused_variables)]
mod client;
mod server;

// m, n, steps, threshold (seconds) and score
const TEST_CASES: [(u32, u32, u32, f64, u32); 9] = [
    (3, 3, 1, 7.0, 2),
    (5, 5, 1, 11.0, 3),
    (5, 5, 2, 20.0, 5),
    (7, 7, 2, 29.0, 7),
    (7, 7, 4, 55.0, 9),
    (8, 10, 4, 79.0, 11),
    (10, 12, 4, 100.0, 13),
    (15, 15, 4, 128.0, 17),
    (17, 17, 5, 256.0, 19),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut score = 0;
    println!("    #      m      n   steps   time (s)   res");
    println!("----- ------ ------ ------- ---------- -----");

    for (i, (m, n, steps, threshold, sco)) in TEST_CASES.iter().enumerate() {
        let start = std::time::Instant::now();
        let client = client::Client::new(*m, *n);
        let (server_key, encrypted_grid) = client.encrypt();
        let server = server::Server::new(server_key, encrypted_grid);

        // Run the server simulation
        let result_grid = server.run(*steps);

        // Verify the result
        let mut pass = client.verify(result_grid, *steps);
        let duration = start.elapsed().as_secs_f64();

        pass &= duration <= *threshold;
        score += if pass { *sco } else { 0u32 };

        println!(
            "{:5} {:6} {:6} {:7} {:10.4}  {}",
            i,
            m,
            n,
            steps,
            duration,
            if pass { "PASS" } else { "FAIL" }
        );
    }

    println!("Score: {}/86", score);

    Ok(())
}
