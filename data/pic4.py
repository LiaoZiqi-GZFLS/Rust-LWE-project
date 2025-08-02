import matplotlib.pyplot as plt
import matplotlib

# 模拟不同编译参数组合下的运行时间（单位：秒）
params = [
    "-O2", 
    "-O3", 
    "-O3 + native", 
    "-O3 + native + lto", 
    "-O3 + native + lto + codegen-units=1"
]
times = [120.4, 90.2, 65.1, 58.9, 55.2]  # 模拟运行时间

matplotlib.rcParams['font.family'] = 'SimHei'

plt.figure(figsize=(10, 5))
plt.barh(params[::-1], times[::-1], color='skyblue')
plt.xlabel("运行时间（秒）", fontsize=12)
plt.title("FHE 编译器参数优化效果", fontsize=14)
plt.tight_layout()
plt.savefig("fhe_compiler_flags_vs_time.png", dpi=300)
plt.show()
