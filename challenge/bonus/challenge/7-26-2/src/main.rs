extern "C" {
    fn run_bkz_on_lattice(matrix: *mut i32, dim: i32, block_size: i32);
}

fn main() {
    let dim = 3;
    let mut lattice = vec![
        105, 821, 114,
        -46,  75, -17,
         41,  67,  92,
    ];

    println!("[Rust] Calling fplll BKZ on {}×{} lattice...", dim, dim);
    unsafe {
        run_bkz_on_lattice(lattice.as_mut_ptr(), dim, 20);
    }
}
