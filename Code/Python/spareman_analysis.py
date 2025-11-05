import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, ttest_ind
import os
from matplotlib import rcParams

# 设置全局字体
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['text.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.edgecolor'] = 'black'

# 彩色调色板
box_palette = ['#4393c3', '#d6604d', '#2ca02c', '#ffbf00', '#7b3294', '#b2182b']

# 1. 读取数据
data = pd.read_csv('F:/Research_Project/EP_ukb/Revision_correlation/nGP_combines_G40_hour_imputed.csv')

core_vars = ['Sedentary_Total_Hour', 'Light_Total_Hour', 'MVPA_Total_Hour']
# 移除睡眠相关变量
corr_vars = [
    'charlson_score', 'BMI', 'Townsend_index', 'enrollment_age', 'gender',
    'Disable_long', 'education_level', 'smoking', 'alcohol_use', 'ethic', 'overall_health'
]

# 简化的变量名（主图用）- 核心变量使用简写
nice_varname_simple = {
    'Sedentary_Total_Hour': 'Sedentary',
    'Light_Total_Hour': 'LPA',
    'MVPA_Total_Hour': 'MVPA',
    'charlson_score': 'Charlson Score',
    'BMI': 'BMI',
    'Townsend_index': 'Townsend Index',
    'enrollment_age': 'Age',
    'gender': 'Gender',
    'Disable_long': 'Long-term Disability',
    'education_level': 'Education Level',
    'smoking': 'Smoking',
    'alcohol_use': 'Alcohol Use',
    'ethic': 'Ethnicity',
    'overall_health': 'Overall Health'
}

# 详细的变量名（附图用）- 核心变量使用简写加单位
nice_varname_detailed = {
    'Sedentary_Total_Hour': 'Sedentary (hours/day)',
    'Light_Total_Hour': 'LPA (hours/day)',
    'MVPA_Total_Hour': 'MVPA (hours/day)',
    'charlson_score': 'Charlson Comorbidity Score',
    'BMI': 'Body Mass Index (kg/m²)',
    'Townsend_index': 'Townsend Deprivation Index',
    'enrollment_age': 'Age at Enrollment (years)',
    'gender': 'Gender',
    'Disable_long': 'Long-term Disability',
    'education_level': 'Education Level',
    'smoking': 'Smoking Status',
    'alcohol_use': 'Alcohol Use',
    'ethic': 'Ethnicity',
    'overall_health': 'Self-rated Overall Health'
}

category_map = {
    'gender': {1: 'Male', 2: 'Female'},
    'Disable_long': {1: 'Yes', 2: 'No'},
    'education_level': {1: 'Below college degree', 2: 'Above college degree'},
    'smoking': {1: 'Never', 2: 'Previous', 3: 'Current'},
    'alcohol_use': {1: 'Never', 2: 'Previous', 3: 'Current'},
    'ethic': {1: 'White', 2: 'Others'},
    'overall_health': {1: 'Excellent', 2: 'Good', 3: 'Fair', 4: 'Poor'}
}
category_vars = list(category_map.keys())
continuous_vars = [v for v in corr_vars if v not in category_vars]

# 2. Spearman相关性和P值
corr_coef_matrix = pd.DataFrame(index=core_vars, columns=corr_vars)
corr_p_matrix = pd.DataFrame(index=core_vars, columns=corr_vars)

for pa in core_vars:
    for covar in corr_vars:
        pa_data = data[pa]
        covar_data = data[covar]
        mask = ~pa_data.isna() & ~covar_data.isna()
        if mask.sum() > 2:
            coef, p = spearmanr(pa_data[mask], covar_data[mask])
        else:
            coef, p = np.nan, np.nan
        corr_coef_matrix.loc[pa, covar] = coef
        corr_p_matrix.loc[pa, covar] = p


# P值与相关系数格式化
def p_str(p):
    if pd.isna(p):
        return ''
    if p < 0.001:
        return '<0.001'
    return f"{p:.3f}"


def r_str(r):
    if pd.isna(r):
        return ''
    return f"{r:.3f}"


# 保存带格式的相关性表格
out_df = []
for pa in core_vars:
    for covar in corr_vars:
        pval = corr_p_matrix.loc[pa, covar]
        rval = corr_coef_matrix.loc[pa, covar]
        out_df.append({
            'Physical Activity Metric': nice_varname_simple[pa],
            'Covariate': nice_varname_simple[covar],
            'Spearman r': r_str(rval),
            'P value': p_str(pval)
        })
out_df = pd.DataFrame(out_df)
out_df.to_excel('F:/Research_Project/EP_ukb/Revision_correlation/Results1/PA_spearman_correlation_with_p.xlsx',
                index=False)

# 3. 主图热图（简化，放大字体）
plt.figure(figsize=(16, 6))
sns.set(font_scale=1.3)  # 适中的字体缩放
ax = sns.heatmap(
    corr_coef_matrix.astype(float),
    annot=False,  # 不显示数值
    cmap='RdBu_r',
    cbar_kws={'label': 'Spearman Correlation Coefficient'},
    linewidths=0.5, center=0, vmin=-0.5, vmax=0.5,
    xticklabels=[nice_varname_simple[v] for v in corr_vars],
    yticklabels=[nice_varname_simple[v] for v in core_vars]
)

# 设置轴标签
ax.set_xlabel('Covariates', fontsize=22, color='black', labelpad=15)
ax.set_ylabel('Physical Activity Variables', fontsize=22, color='black', labelpad=15)

# 设置刻度标签 - 横轴放在下方，适中字体
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=18)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=20, rotation=0)

for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontname('Arial')

# 设置颜色条
cbar = ax.collections[0].colorbar
cbar.ax.tick_params(labelsize=16, colors='black')
cbar.set_label('Spearman Correlation Coefficient', size=18, color='black')

plt.tight_layout()
plt.savefig('F:/Research_Project/EP_ukb/Revision_correlation/Results1/PA_correlation_heatmap_main.png', dpi=300,
            bbox_inches='tight')
plt.close()

# 分类变量箱线图（放大字体）
parent_dir = 'F:/Research_Project/EP_ukb/Revision_correlation/Results1/Boxplots'
os.makedirs(parent_dir, exist_ok=True)

for cat in category_vars:
    sub_dir = os.path.join(parent_dir, cat)
    os.makedirs(sub_dir, exist_ok=True)
    for pa in core_vars:
        cat_codes = sorted([c for c in data[cat].unique() if pd.notna(c)])
        group_names = [category_map[cat].get(code, str(code)) for code in cat_codes]
        values = [data[data[cat] == code][pa] for code in cat_codes]
        means = [v.mean() for v in values]
        stds = [v.std() for v in values]
        # T检验只做二分类
        ttest_p = None
        ttest_str = ""
        if len(cat_codes) == 2:
            t, t_p = ttest_ind(values[0], values[1], nan_policy='omit')
            ttest_p = t_p
            ttest_str = f"T-test p={'<0.001' if t_p < 0.001 else f'{t_p:.3f}'}"

        plt.figure(figsize=(10, 8))
        sns.boxplot(x=data[cat].map(category_map[cat]), y=data[pa], palette=box_palette[:len(cat_codes)])
        plt.xlabel(nice_varname_detailed[cat], fontsize=28, color='black')
        plt.ylabel(nice_varname_detailed[pa], fontsize=28, color='black')

        stat_text = '\n'.join([f"{group_names[i]}: {means[i]:.2f}±{stds[i]:.2f}" for i in range(len(group_names))])
        if ttest_str:
            stat_text += f"\n{ttest_str}"

        plt.xticks(fontsize=24, fontname='Arial', color='black')
        plt.yticks(fontsize=22, fontname='Arial', color='black')
        plt.gca().text(0.01, 0.99, stat_text, fontsize=20, color='black', fontname='Arial',
                       transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='left',
                       bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.8))
        plt.tight_layout()
        plt.savefig(os.path.join(sub_dir, f'Boxplot_{pa}_by_{cat}.png'), dpi=300, bbox_inches='tight')
        plt.close()

# 连续变量散点图（放大字体）
scatter_dir = 'F:/Research_Project/EP_ukb/Revision_correlation/Results1/Scatterplots'
os.makedirs(scatter_dir, exist_ok=True)

for cont in continuous_vars:
    for pa in core_vars:
        mask = ~data[cont].isna() & ~data[pa].isna()
        if mask.sum() > 2:
            x = data[cont][mask]
            y = data[pa][mask]
            coef, pval = spearmanr(x, y)
            coef_str = r_str(coef)
            pval_str = p_str(pval)

            plt.figure(figsize=(10, 8))
            sns.regplot(x=x, y=y,
                        scatter_kws={'s': 20, 'color': '#ff7f0e', 'alpha': 0.7},
                        line_kws={'color': '#4393c3', 'lw': 3})
            plt.xlabel(nice_varname_detailed[cont], fontsize=28, color='black')
            plt.ylabel(nice_varname_detailed[pa], fontsize=28, color='black')

            plt.gca().text(0.99, 0.99, f"Spearman r={coef_str}\np={pval_str}",
                           fontsize=20, color='black', fontname='Arial',
                           transform=plt.gca().transAxes, verticalalignment='top', horizontalalignment='right',
                           bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3', alpha=0.8))
            plt.xticks(fontsize=24, fontname='Arial', color='black')
            plt.yticks(fontsize=22, fontname='Arial', color='black')
            plt.tight_layout()
            plt.savefig(os.path.join(scatter_dir, f'Scatter_{pa}_vs_{cont}.png'), dpi=300, bbox_inches='tight')
            plt.close()

print("Updated plots with simplified PA variable names (Sedentary, LPA, MVPA).")