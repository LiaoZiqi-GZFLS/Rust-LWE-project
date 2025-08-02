import matplotlib.pyplot as plt
import matplotlib

sample_sizes = [128, 256, 512, 1024, 2048]
time_no_pbs = [1.8, 3.7, 8.4, 18.2, 37.9]
time_pbs    = [1.0, 2.1, 6.5, 19.1, 39.8]  # PBS 对小样例更好，大样例略差

matplotlib.rcParams['font.family'] = 'SimHei'

plt.figure(figsize=(8, 5))
plt.plot(sample_sizes, time_no_pbs, marker='o', label='无 PBS', linestyle='--')
plt.plot(sample_sizes, time_pbs, marker='s', label='有 PBS', linestyle='-')
plt.xlabel("样例大小（输入比特数）")
plt.ylabel("运行时间（秒）")
plt.title("PBS 优化在不同规模数据下的影响")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fhe_pbs_effect.png", dpi=300)
plt.show()
