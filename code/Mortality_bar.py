import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_connected_bar_chart(data, column):
    # 计算三分位数
    tertiles = data[column].quantile([1 / 3, 2 / 3])

    # 创建三分位组
    data['tertile'] = pd.cut(data[column],
                             bins=[-np.inf, tertiles.iloc[0], tertiles.iloc[1], np.inf],
                             labels=[1, 2, 3])

    # 计算每个三分位组的死亡率
    mortality_rates = data.groupby('tertile')['is_dead'].mean()

    # 获取每个三分位组的范围
    tertile_ranges = [
        f'≤{tertiles.iloc[0]:.0f}',
        f'{tertiles.iloc[0]:.0f}-{tertiles.iloc[1]:.0f}',
        f'{tertiles.iloc[1]:.0f}-{data[column].max():.0f}'
    ]

    # 创建图表
    plt.figure(figsize=(20, 15))

    # 绘制柱状图
    bars = plt.bar(range(1, 4), mortality_rates, width=0.6, color='skyblue', edgecolor='navy')

    # 添加连接线
    plt.plot(range(1, 4), mortality_rates, color='navy', linewidth=3, marker='o', markersize=15)

    # 添加数据标签
    for i, rate in enumerate(mortality_rates):
        plt.text(i + 1, rate, f'{rate:.2%}',
                 ha='center', va='bottom', fontsize=36, fontweight='bold')

   # plt.title(f'Mortality Rate by {column} Tertiles', fontsize=44, fontweight='bold')
    plt.xlabel('Tertiles', fontsize=36)
    plt.ylabel('Mortality Rate', fontsize=36)

    plt.xticks(range(1, 4), tertile_ranges, fontsize=28)
    plt.yticks(fontsize=28)

    # 设置y轴为百分比格式
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # 添加网格线
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(f'E:/EP_ukb/Results/Figures/Figure3/new_mortality_rate_{column}_connected_bar.png', dpi=300, bbox_inches='tight')
    plt.close()


# 读取数据
data = pd.read_csv('nGP_combines.csv')

# 处理无穷大的值
data = data.replace([np.inf, -np.inf], np.nan)

# 绘制指定列的图表
columns_to_plot = ['Light_Total_Min', 'MVPA_Total_Min', 'Sedentary_Total_Min']

for column in columns_to_plot:
    plot_connected_bar_chart(data, column)

print("所有相连柱状图已生成完毕。")