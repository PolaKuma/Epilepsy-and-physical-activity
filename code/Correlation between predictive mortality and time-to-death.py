import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import spearmanr, kruskal
import scikit_posthocs as sp
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from lifelines.utils import concordance_index
from scipy.stats import chi2_contingency, fisher_exact
import itertools

# 1. Read Data
csv_path = 'F:/Research_Project/EP_ukb/Revision_machine_learning/Results_medicine/time_to_death/TOD_predict_combined.csv'  # <-- Replace with your CSV file path
df = pd.read_csv(csv_path)
df['risk_group'] = pd.qcut(df['pred_prob'], 4, labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'])
df['risk_group'] = pd.Categorical(df['risk_group'],
                                  categories=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'],
                                  ordered=True)

# 2. Analysis
rho, pval = spearmanr(df['pred_prob'], df['followup_time'])
cidx = concordance_index(df['followup_time'], -df['pred_prob'], df['true_label'])
median_times = df.groupby('risk_group')['followup_time'].median()
mortality_rates = df.groupby('risk_group')['true_label'].mean()
group_stats = df.groupby('risk_group')['followup_time'].agg(['mean', 'std']).applymap(lambda x: np.round(x, 2))

# Kruskal-Wallis & Dunn’s test
grouped = [df.loc[df['risk_group'] == q, 'followup_time'] for q in df['risk_group'].cat.categories]
kw_stat, kw_p = kruskal(*grouped)
dunn = sp.posthoc_dunn(df, val_col='followup_time', group_col='risk_group', p_adjust='bonferroni')

# Chi-square for mortality
contingency = pd.crosstab(df['risk_group'], df['true_label'])
chi2, p_chi2, _, _ = chi2_contingency(contingency)

# 3. Visualization settings
plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 16
})

# --------- KM curve with filled area and P-value brackets ---------
plt.figure(figsize=(10,7))
colors = sns.color_palette("Set2", 4)
kmf = KaplanMeierFitter()
km_curves = {}

for idx, (name, grouped) in enumerate(df.groupby('risk_group')):
    ax = plt.gca()
    kmf.fit(durations=grouped['followup_time'], event_observed=grouped['true_label'], label=name)
    kmf.plot(ci_show=False, ax=ax, color=colors[idx], linewidth=2)
    # Fill area under curve
    plt.fill_between(kmf.survival_function_.index,
                     0,
                     kmf.survival_function_['KM_estimate'],
                     color=colors[idx], alpha=0.15, step='post')
    # Save for later logrank test
    km_curves[name] = grouped

plt.xlabel("Time (years)")
plt.ylabel("Survival probability")
plt.title("Kaplan-Meier survival by predicted risk quartile")
plt.legend(title='Risk group')

# Log-rank tests Q1 vs Q2, Q1 vs Q3, Q1 vs Q4, and annotate with brackets
groups_to_compare = [('Q1 (Lowest)', 'Q2'), ('Q1 (Lowest)', 'Q3'), ('Q1 (Lowest)', 'Q4 (Highest)')]
y_bracket = 1.05
step_bracket = 0.07
for i, (g1, g2) in enumerate(groups_to_compare):
    qA = km_curves[g1]
    qB = km_curves[g2]
    results = logrank_test(qA['followup_time'], qB['followup_time'],
                           event_observed_A=qA['true_label'], event_observed_B=qB['true_label'])
    p_lrt = results.p_value
    pstr = "<0.001" if p_lrt < 0.001 else f"p={p_lrt:.3f}"

    # Bracket coordinates
    x1, x2 = i, i+1
    y = y_bracket + i * step_bracket
    plt.plot([x1, x1, x2, x2], [y-0.01, y, y, y-0.01], color='k', lw=1.5, clip_on=False)
    plt.text((x1 + x2)/2, y + 0.01, pstr, ha='center', va='bottom', fontsize=17)

plt.ylim(-0.05, 1.18)
plt.tight_layout()
plt.savefig('F:/Research_Project/EP_ukb/Revision_machine_learning/Results_medicine/time_to_death/TOD_predict_combined.csvKMcurve_quartile_filled.png', dpi=200)
plt.show()

# --------- Boxplot: follow-up time by risk group, show mean±std ---------
plt.figure(figsize=(8,6))
sns.boxplot(x='risk_group', y='followup_time', data=df, palette='Set2', showmeans=True,
            meanprops={"marker":"o",
                       "markerfacecolor":"red",
                       "markeredgecolor":"black",
                       "markersize":"8"})
plt.xlabel('Predicted risk quartile')
plt.ylabel('Time to death or last follow-up (years)')
plt.title('Follow-up time by predicted risk quartile')
plt.gca().invert_yaxis()

# Annotate mean ± std above each box
for i, cat in enumerate(df['risk_group'].cat.categories):
    mean = group_stats.loc[cat, 'mean']
    std = group_stats.loc[cat, 'std']
    plt.text(i, df['followup_time'].min()-0.5, f"{mean:.2f}±{std:.2f}",
             ha='center', va='bottom', color='red', fontsize=15)

plt.tight_layout()
plt.savefig('boxplot_quartile_followup_meanstd.png', dpi=200)
plt.show()