// 引入自动生成的 FFI
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

fn main() {
    unsafe {
        let a = add(3, 4);
        let b = mul(5, 6);
        println!("add = {a}, mul = {b}");
    }
}