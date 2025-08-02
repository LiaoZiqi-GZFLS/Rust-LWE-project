import matplotlib.pyplot as plt
import matplotlib

# 假设我们对样例编号 0~6 的维度与恢复误差进行统计
dimensions = [10, 20, 30, 40, 45, 50, 55]
errors = [0, 0, 3.1, 6.4, 12.0, 18.5, 25.3]  # L2 范数误差（模拟数据）

matplotlib.rcParams['font.family'] = 'SimHei'

plt.figure(figsize=(7, 4))
plt.scatter(dimensions, errors, color='crimson')
plt.title("维度 vs LWE 恢复误差范数", fontsize=14)
plt.xlabel("LWE 维度 (n)", fontsize=12)
plt.ylabel("恢复误差范数 (L2)", fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig("lwe_dimension_vs_error.png", dpi=300)
plt.show()
