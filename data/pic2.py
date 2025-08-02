import matplotlib.pyplot as plt
import matplotlib

# 模拟枚举半径与成功恢复次数关系
radius = list(range(1, 11))  # 半径 1 到 10
success_counts = [0, 1, 2, 2, 3, 2, 1, 0, 0, 0]  # 假设的恢复成功次数

matplotlib.rcParams['font.family'] = 'SimHei'

plt.figure(figsize=(7, 4))
plt.plot(radius, success_counts, marker='o', color='teal', linewidth=2)
plt.title("LWE 枚举半径与恢复成功次数", fontsize=14)
plt.xlabel("枚举半径", fontsize=12)
plt.ylabel("成功恢复次数", fontsize=12)
plt.grid(True)
plt.xticks(radius)
plt.tight_layout()
plt.savefig("lwe_radius_vs_success.png", dpi=300)
plt.show()
