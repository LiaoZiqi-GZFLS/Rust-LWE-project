#[allow(dead_code, unused_variables)]
mod client;
mod server;

// m, n, steps, threshold (seconds) and score
const TEST_CASES: [(u32, u32, u32, f64, u32); 9] = [
    (3, 3, 1, 7.0, 2),
    (5, 5, 1, 10.0, 3),
    (5, 5, 2, 15.0, 5),
    (7, 7, 2, 21.0, 7),
    (7, 7, 4, 39.0, 9),
    (8, 10, 4, 70.0, 11),
    (10, 12, 4, 96.0, 13),
    (15, 15, 4, 128.0, 17),
    (17, 17, 5, 196.0, 19),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut score = 0;
    // 打印表头，增加结果正确性和时间是否超时的列
    println!("    #      m      n   steps   time (s)   res   correct   time_limit");
    println!("----- ------ ------ ------- ---------- ----- ---------  -----------");

    for (i, (m, n, steps, threshold, sco)) in TEST_CASES.iter().enumerate() {
        let start = std::time::Instant::now();
        let client = client::Client::new(*m, *n);
        let (server_key, encrypted_grid) = client.encrypt();
        let server = server::Server::new(server_key, encrypted_grid);

        // 运行服务器模拟
        let result_grid = server.run(*steps);

        // 验证结果
        let correct = client.verify(result_grid, *steps);
        let duration = start.elapsed().as_secs_f64();
        let time_limit_passed = duration <= *threshold;

        let pass = correct && time_limit_passed;
        score += if pass { *sco } else { 0u32 };

        // 打印每个测试用例的结果，包括结果正确性和时间是否超时
        println!(
            "{:5} {:6} {:6} {:7} {:10.4}  {}       {}         {}",
            i,
            m,
            n,
            steps,
            duration,
            if pass { "PASS" } else { "FAIL" },
            if correct { "P" } else { "F" },
            if time_limit_passed { "P" } else { "F" }
        );
    }

    println!("Score: {}/86", score);

    Ok(())
}