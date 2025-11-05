import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# 设置全局字体为Arial
plt.rcParams['font.family'] = 'Arial'


# 格式化p值的辅助函数
def format_p_value(p_value):
    """格式化p值，<0.001的显示为<0.001，其他保留3位有效数字"""
    if p_value < 0.001:
        return "<0.001"
    elif p_value < 0.01:
        return f"{p_value:.3f}"
    else:
        return f"{p_value:.2f}"


# 三分位阈值映射
tertile_thresholds = {
    'Light_Tertile': ['≤4.20 h', '>4.20-5.60 h', '>5.60 h'],
    'MVPA_Tertile': ['≤0.40 h', '>0.40-0.80 h', '>0.80 h'],
    'Sedentary_Tertile': ['≤8.60 h', '>8.60-10.20 h', '>10.20 h']
}


# 修改ISMR柱状图 - 组内比较，同组柱子相邻排列
def plot_ismr_bar_chart(ismr_data, save_dir):
    """创建柱状图展示ISMR数据，同组柱子（T1、T2、T3）连续放置"""

    # 获取唯一的变量名
    variables = ismr_data['Variable'].unique()

    # 设置图表样式
    sns.set_style("whitegrid")

    # 为每个变量创建一个图表
    for variable in variables:
        # 筛选当前变量的数据
        var_data = ismr_data[ismr_data['Variable'] == variable]

        # 创建图表
        fig, ax = plt.subplots(figsize=(18, 12))

        # 设置分组和颜色 - 更新图例标签
        groups = ['Epilepsy', 'Non-Epilepsy']
        legend_labels = ['People with Epilepsy', 'People without Epilepsy']  # 更新的图例标签
        colors = ['orange', 'steelblue']
        categories = [1, 2, 3]

        # 存储p值信息，用于稍后集中显示
        p_value_info = []

        # 设置柱子位置 - 每组三个柱子连续放置
        bar_positions = []
        group_positions = []  # 用于xticks位置
        group_labels = []  # 用于xticks标签

        # 定义柱宽和组间间隔
        width = 0.7
        group_gap = 1.5

        # 计算柱子位置
        for i, group in enumerate(groups):
            # 计算当前组的起始位置
            start_pos = i * (3 + group_gap) + 1
            # 该组的三个柱子位置
            positions = [start_pos, start_pos + 1, start_pos + 2]
            bar_positions.append(positions)

            # 计算组标签位置（中心位置）
            group_positions.append(start_pos + 1)
            group_labels.append(group)

            # 处理各组数据
            group_data = var_data[var_data['Group'] == group].sort_values('Category')

            # 确保有所有三个分类
            group_ismr = []
            group_lower = []
            group_upper = []
            group_n = []
            group_p = []
            group_sig = []

            for cat in categories:
                cat_data = group_data[group_data['Category'] == cat]
                if len(cat_data) > 0:
                    group_ismr.append(cat_data['ISMR'].values[0])
                    group_lower.append(cat_data['Lower_CI'].values[0])
                    group_upper.append(cat_data['Upper_CI'].values[0])
                    group_n.append(cat_data['N'].values[0])
                    if cat > 1:  # 参考组没有p值
                        group_p.append(cat_data['P_value'].values[0])
                        group_sig.append(cat_data['Significance'].values[0])
                else:
                    group_ismr.append(0)
                    group_lower.append(0)
                    group_upper.append(0)
                    group_n.append(0)
                    if cat > 1:
                        group_p.append(1.0)
                        group_sig.append('ns')

            # 计算误差线高度
            yerr = [np.array(group_ismr) - np.array(group_lower),
                    np.array(group_upper) - np.array(group_ismr)]

            # 绘制柱状图 - 移除label
            bars = ax.bar(positions, group_ismr, width, color=colors[i],
                          alpha=0.8, edgecolor='black', linewidth=1.5)

            # 添加误差线
            ax.errorbar(positions, group_ismr, yerr=yerr, fmt='none',
                        color='black', capsize=8, capthick=2, elinewidth=2)

            # 在柱子顶部添加点
            ax.scatter(positions, group_ismr, color='black', s=100, zorder=3)

            # 添加ISMR值标签
            for j, (pos, ismr, n) in enumerate(zip(positions, group_ismr, group_n)):
                if ismr > 0:  # 只为有值的柱子添加标签
                    ax.text(pos, ismr + yerr[1][j] + 0.1, f'{ismr:.2f}',
                            ha='center', va='bottom', fontsize=32, fontweight='bold')
                    ax.text(pos, ismr / 2, f'n={n}', ha='center', va='center',
                            fontsize=28, color='white', fontweight='bold')

            # 收集有效的p值信息 - 使用阈值标签
            thresholds = tertile_thresholds.get(variable, [f'T{i}' for i in range(1, 4)])
            for j in range(len(group_p)):
                p_value_info.append(
                    f"{legend_labels[i]} {thresholds[j + 1]} vs {thresholds[0]}: p={format_p_value(group_p[j])} {group_sig[j]}")

        # 添加参考线（ISMR=1，表示无差异）
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2.5)

        # 设置图表标题和标签 - 调整标题大小
        var_display = {
            'Light_Tertile': 'Light Physical Activity',
            'MVPA_Tertile': 'Moderate-to-Vigorous Physical Activity',
            'Sedentary_Tertile': 'Sedentary Behavior'
        }.get(variable, variable)

        ax.set_title(f'Internal SMR by {var_display} Tertiles', fontsize=36, fontweight='bold')
        ax.set_xlabel('Groups and Tertile Categories', fontsize=40)
        ax.set_ylabel('Internal Standardized Mortality Ratio (ISMR)', fontsize=40)

        # 获取当前变量的阈值标签
        tertile_labels = tertile_thresholds.get(variable, [f'T{i}' for i in range(1, 4)])

        # 设置x轴刻度位置 - 所有柱子的位置
        all_positions = [pos for group_pos in bar_positions for pos in group_pos]

        # 设置x轴标签 - 只使用时间范围标签，不添加组标签
        x_labels = []
        for i in range(len(groups)):
            x_labels.extend(tertile_labels)

        ax.set_xticks(all_positions)
        ax.set_xticklabels(x_labels, fontsize=28)

        # 设置刻度线颜色
        ax.tick_params(axis='y', labelsize=36)

        # 设置y轴范围
        max_upper = 0
        for group in groups:
            group_data = var_data[var_data['Group'] == group]
            if not group_data.empty:
                max_upper = max(max_upper, group_data['Upper_CI'].max())

        max_upper = max(max_upper + 0.5, 2.5)
        ax.set_ylim(0, max_upper)

        # 根据变量类型决定p值标签位置
        if variable == 'MVPA_Tertile':
            # MVPA图的p值放在右上角
            p_x, p_ha = 0.97, 'right'
        else:
            # Light和Sedentary图的p值放在左上角
            p_x, p_ha = 0.03, 'left'

        # 集中显示p值信息
        if p_value_info:
            p_info_text = '\n'.join(p_value_info)
            text_box = ax.text(p_x, 0.97, p_info_text, transform=ax.transAxes, fontsize=28,
                               verticalalignment='top', horizontalalignment=p_ha,
                               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

        # 不添加图例（根据要求）

        # 添加网格线
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # 保存图表
        plt.tight_layout()
        var_file_name = variable.split('_')[0].lower()
        plt.savefig(f'{save_dir}ismr_bar_{var_file_name}.png', dpi=300, bbox_inches='tight')
        plt.close()


# 修改组间SMR柱状图绘制函数
def plot_between_group_smr(smr_data, save_dir):
    """创建柱状图展示癫痫vs非癫痫的SMR数据，以两组柱状图方式呈现"""

    # 获取唯一的变量名
    variables = smr_data['Variable'].unique()

    # 设置图表样式
    sns.set_style("whitegrid")

    # 为每个变量创建一个图表
    for variable in variables:
        # 筛选当前变量的数据
        var_data = smr_data[smr_data['Variable'] == variable].sort_values('Category')

        # 创建图表
        fig, ax = plt.subplots(figsize=(16, 12))

        # 提取数据
        categories = var_data['Category'].values
        smr_values = var_data['SMR'].values
        lower_ci = var_data['Lower_CI'].values
        upper_ci = var_data['Upper_CI'].values
        epi_n = var_data['Epilepsy_N'].values
        non_epi_n = var_data['Non_Epilepsy_N'].values
        p_values = var_data['P_value'].values
        significance = var_data['Significance'].values

        # 设置柱状图位置参数
        width = 0.35  # 柱宽

        # 计算误差线高度
        yerr = [smr_values - lower_ci, upper_ci - smr_values]

        # 非癫痫组的位置（SMR固定为1，无误差）
        non_epi_pos = np.array(categories) - width / 2
        epi_pos = np.array(categories) + width / 2

        # 绘制非癫痫组柱状图（参考组，SMR=1）
        ax.bar(non_epi_pos, np.ones_like(categories), width, color='steelblue',
               alpha=0.8, edgecolor='black', linewidth=1.5)  # 移除label

        # 绘制癫痫组柱状图
        epi_bars = ax.bar(epi_pos, smr_values, width, color='orange',
                          alpha=0.8, edgecolor='black', linewidth=1.5)  # 移除label

        # 添加误差线
        ax.errorbar(epi_pos, smr_values, yerr=yerr, fmt='none',
                    color='black', capsize=8, capthick=2, elinewidth=2)

        # 在癫痫组柱子顶部添加点
        ax.scatter(epi_pos, smr_values, color='black', s=100, zorder=3)

        # 添加SMR值和样本量标签
        for i, (pos, smr, epi, non_epi) in enumerate(zip(epi_pos, smr_values, epi_n, non_epi_n)):
            # 癫痫组显示SMR值
            ax.text(pos, smr + yerr[1][i] + 0.1, f'{smr:.2f}',
                    ha='center', va='bottom', fontsize=32, fontweight='bold')
            ax.text(pos, smr / 2, f'n={epi}', ha='center', va='center',
                    fontsize=28, color='white', fontweight='bold')

            # 非癫痫组显示样本量
            ax.text(non_epi_pos[i], 0.5, f'n={non_epi}', ha='center', va='center',
                    fontsize=28, color='white', fontweight='bold')

        # 添加参考线（SMR=1，表示无差异）
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.7, linewidth=2.5)

        # 设置图表标题和标签 - 调整标题大小
        var_display = {
            'Light_Tertile': 'Light Physical Activity',
            'MVPA_Tertile': 'Moderate-to-Vigorous Physical Activity',
            'Sedentary_Tertile': 'Sedentary Behavior'
        }.get(variable, variable)

        ax.set_title(f'SMR by {var_display} Tertiles', fontsize=36, fontweight='bold')  # 标题简化并调整大小
        ax.set_xlabel('Tertile Category', fontsize=40)
        ax.set_ylabel('Standardized Mortality Ratio (SMR)', fontsize=40)

        # 获取当前变量的阈值标签
        tertile_labels = tertile_thresholds.get(variable, [f'T{i}' for i in range(1, 4)])

        # 设置x轴刻度 - 使用阈值标签
        ax.set_xticks(categories)
        ax.set_xticklabels([tertile_labels[int(c) - 1] for c in categories], fontsize=36)
        ax.tick_params(axis='y', labelsize=36)

        # 设置y轴范围
        max_val = max(np.max(upper_ci) + 0.5, 2.5)
        ax.set_ylim(0, max_val)

        # 收集p值信息
        p_value_info = []
        for i, (cat, p, sig) in enumerate(zip(categories, p_values, significance)):
            cat_idx = int(cat) - 1
            p_value_info.append(f"{tertile_labels[cat_idx]}: p={format_p_value(p)} {sig}")

        # 根据变量类型决定p值标签位置
        if variable == 'Sedentary_Tertile':
            # Sedentary图的p值放在右上角
            p_x, p_ha = 0.97, 'right'
        else:
            # Light和MVPA图的p值放在左上角
            p_x, p_ha = 0.03, 'left'

        # 集中显示p值信息
        if p_value_info:
            p_info_text = '\n'.join(p_value_info)
            ax.text(p_x, 0.97, p_info_text, transform=ax.transAxes, fontsize=28,
                    verticalalignment='top', horizontalalignment=p_ha,
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='gray'))

        # 不添加图例（根据要求）

        # 添加网格线
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)

        # 保存图表
        plt.tight_layout()
        var_file_name = variable.split('_')[0].lower()
        plt.savefig(f'{save_dir}between_group_smr_{var_file_name}.png', dpi=300, bbox_inches='tight')
        plt.close()


# 创建单独的图例图片
def create_legend_chart(save_dir):
    """创建单独的图例图片供PPT使用"""

    fig, ax = plt.subplots(figsize=(12, 3))

    # 隐藏坐标轴
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')

    # 创建图例元素
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, color='steelblue', alpha=0.8,
                      edgecolor='black', linewidth=1.5, label='People without Epilepsy (Reference)'),
        plt.Rectangle((0, 0), 1, 1, color='orange', alpha=0.8,
                      edgecolor='black', linewidth=1.5, label='People with Epilepsy')
    ]

    # 添加图例
    legend = ax.legend(handles=legend_elements, fontsize=36, loc='center',
                       frameon=True, fancybox=True, shadow=True, ncol=2)

    # 设置图例框样式
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_alpha(0.9)

    # 保存图例
    plt.tight_layout()
    plt.savefig(f'{save_dir}legend_chart.png', dpi=300, bbox_inches='tight')
    plt.close()


# 主函数
def main():
    # 设置文件目录
    save_dir = 'F:/Research_Project/EP_ukb/Revision_Tertile/tertile_ismr/revision/'

    # 读取ISMR结果和组间SMR结果
    print("读取SMR和ISMR结果...")
    ismr_data = pd.read_csv(f'{save_dir}all_tertile_ismr_results.csv')
    between_group_smr = pd.read_csv(f'{save_dir}between_group_smr_results.csv')

    # 确保输出目录存在
    os.makedirs(save_dir, exist_ok=True)

    # 生成组内ISMR柱状图
    print("生成组内ISMR柱状图...")
    plot_ismr_bar_chart(ismr_data, save_dir)

    # 生成组间SMR柱状图
    print("生成组间SMR柱状图...")
    plot_between_group_smr(between_group_smr, save_dir)

    # 生成单独的图例图片
    print("生成图例图片...")
    create_legend_chart(save_dir)

    print("所有图表已生成，保存到:", save_dir)


if __name__ == "__main__":
    main()