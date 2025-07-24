# 关于夜神骇客队RUST代码运行的说明

我们的代码由docker构建，由stable模式运行。

平时测试指令：`docker compose up stable`

完整运行指令：`bash -c "cargo build --release && cargo run --release"`

不过我们没有改什么cargo的版本与依赖，正常来说在linux中也可以直接运行。
