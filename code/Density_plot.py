import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde


def plot_density_with_histogram(data, column):
    plt.figure(figsize=(12, 8))

    # 创建两个子图
    gs = plt.GridSpec(2, 1, height_ratios=[3, 1])
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])

    # 绘制密度图
    for is_dead, color, label in [(0, '#1f77b4', 'Non-death'), (1, '#ff7f0e', 'Death')]:
        subset = data[data['is_dead'] == is_dead][column].dropna()
        kde = gaussian_kde(subset)
        x_range = np.linspace(subset.min(), subset.max(), 1000)
        y = kde(x_range)

        # 密度线
        ax1.plot(x_range, y, color=color, label=label, linewidth=3)
        ax1.fill_between(x_range, y, alpha=0.3, color=color)

        # 找到并标记峰值
        peak_x = x_range[np.argmax(y)]
        peak_y = np.max(y)
        ax1.axvline(peak_x, color=color, linestyle='--', alpha=0.7, linewidth=2)
        ax1.text(peak_x, peak_y, f'{peak_x:.2f}', color=color, ha='right', va='bottom', fontsize=12)

    ax1.set_ylabel('Density', fontsize=18)
    ax1.legend(fontsize=16)

    # 绘制直方图
    sns.histplot(data=data, x=column, hue='is_dead', multiple="stack",
                 palette=['#1f77b4', '#ff7f0e'], ax=ax2, alpha=0.7)

    ax2.set_xlabel('Hour of day', fontsize=18)
    ax2.set_ylabel('Count', fontsize=18)

    # 移除坐标轴边框和刻度线
    for ax in [ax1, ax2]:
        # 移除坐标轴的所有边框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        # 隐藏刻度线
        ax.tick_params(left=False, bottom=False)

        # 隐藏网格线
        ax.grid(False)

        # 隐藏 x 和 y 轴的刻度标签
        ax.tick_params(axis='x', which='both', labelbottom=False)  # 隐藏 x 轴刻度标签
        ax.tick_params(axis='y', which='both', labelleft=False)  # 隐藏 y 轴刻度标签

    # 只在下面的子图中显示 x 轴的刻度标签
    ax2.tick_params(axis='x', which='both', labelbottom=True)

    plt.tight_layout()

    # 保存图像，背景透明
    plt.savefig(f'density_histogram_plot_{column}.png', dpi=300, bbox_inches='tight', transparent=True)

    plt.close()


# 读取数据
data = pd.read_csv('nGP_combines.csv')

# 处理无穷大的值
data = data.replace([np.inf, -np.inf], np.nan)

# 只绘制指定的三列
columns_to_plot = ['Light_Total_Min', 'MVPA_Total_Min', 'Sedentary_Total_Min']

for column in columns_to_plot:
    plot_density_with_histogram(data, column)

print("所有密度图和直方图已生成完毕。")