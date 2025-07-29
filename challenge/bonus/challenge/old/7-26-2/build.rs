fn main() {
    // 明确指定 fplll 的头文件路径
    let fplll_include_path = "/usr/local/include";

    cc::Build::new()
        .cpp(true)
        .include("wrapper")                    // 你自己的头文件
        .include(fplll_include_path)           // fplll 的 include 路径
        .file("wrapper/fplll_wrapper.cpp")
        .flag_if_supported("-std=c++17")
        .compile("fplllwrapper");

    // 链接 fplll 静态库（系统安装）
    println!("cargo:rustc-link-search=native=/usr/local/lib");
    println!("cargo:rustc-link-lib=static=fplllwrapper"); // 你构建的 .a 库
    println!("cargo:rustc-link-lib=fplll");               // fplll 本体
    println!("cargo:rustc-link-lib=gmp");
    println!("cargo:rustc-link-lib=mpfr");

    println!("cargo:rerun-if-changed=wrapper/fplll_wrapper.cpp");
}
