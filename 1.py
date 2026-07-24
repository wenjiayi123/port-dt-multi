import numpy as np
import matplotlib.pyplot as plt

# ========= 1) 你的数据（把下面示例值改成你的真实结果）=========
factors = ["燃料价格", "碳价", "CAPEX", "OPEX", "残值"]

# 低情景/高情景相对基准的 NPV 变化（单位：%）
# 约定：低情景一般在左侧（负值），高情景在右侧（正值）
low  = np.array([-30, -18, -10,  -8, -3], dtype=float)
high = np.array([ 32,  20,  12,   9,  2], dtype=float)

# ========= 2) 排序：按最大绝对影响从大到小（龙卷风图关键）=========
impact = np.maximum(np.abs(low), np.abs(high))
order = np.argsort(impact)[::-1]
factors = [factors[i] for i in order]
low, high = low[order], high[order]

# ========= 3) 画图（更精致的排版/样式）=========
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)

y = np.arange(len(factors))

# 颜色：给左右两侧不同色（你也可以换成论文主色系）
c_low  = "#4C78A8"   # 蓝
c_high = "#F58518"   # 橙

# 条形（左右对称）
ax.barh(y, low,  height=0.62, color=c_low,  alpha=0.90, edgecolor="white", linewidth=1.0, label="低情景")
ax.barh(y, high, height=0.62, color=c_high, alpha=0.90, edgecolor="white", linewidth=1.0, label="高情景")

# 0 参考线
ax.axvline(0, color="black", linewidth=1.2)

# 网格（更像论文图）
ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.35)
ax.set_axisbelow(True)

# 坐标/标题
ax.set_yticks(y)
ax.set_yticklabels(factors, fontsize=11)
ax.set_xlabel("NPV 变化（%）", fontsize=12)
ax.set_title("关键因素对 NPV 的敏感性分析（龙卷风图）", fontsize=13, pad=10)

# x 轴范围：自动对称留白
xmin = min(low.min(), 0)
xmax = max(high.max(), 0)
lim = max(abs(xmin), abs(xmax)) * 1.15
ax.set_xlim(-lim, lim)

# 数值标注：靠近条形末端
def add_labels(values, ys, side):
    for v, yy in zip(values, ys):
        if v == 0:
            continue
        offset = lim * 0.015
        if side == "left":
            ax.text(v - offset, yy, f"{v:.0f}%", va="center", ha="right", fontsize=9)
        else:
            ax.text(v + offset, yy, f"{v:.0f}%", va="center", ha="left", fontsize=9)

add_labels(low, y, "left")
add_labels(high, y, "right")

# 图例
ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=10)

# 去掉多余边框，让图更干净
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
plt.show()
