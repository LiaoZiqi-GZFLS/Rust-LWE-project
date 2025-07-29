use std::path::PathBuf;

fn main() {
    // 1️⃣ 收集 C 源文件
    let mut build = cc::Build::new();
    build.include("c_src/include").flag_if_supported("-Wall");
    for entry in glob::glob("c_src/**/*.c").expect("glob c") {
        build.file(entry.unwrap());
    }
    build.compile("clib");

    // 2️⃣ 收集 C++ 源文件（如果没有 .cpp 可跳过）
    let mut cpp = cc::Build::new();
    cpp.cpp(true).include("c_src/include");
    for entry in glob::glob("c_src/**/*.cpp").expect("glob cpp") {
        cpp.file(entry.unwrap());
    }
    cpp.compile("cpplib");

    // 3️⃣ 生成绑定
    let bindings = bindgen::Builder::default()
        .header("c_src/include/mathlib.h")
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new())) // ✅ 修复弃用
        .generate()
        .expect("bindgen failed");

    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    bindings.write_to_file(out.join("bindings.rs"))
            .expect("write failed");

    // 4️⃣ 监控目录变化
    println!("cargo:rerun-if-changed=c_src");
}