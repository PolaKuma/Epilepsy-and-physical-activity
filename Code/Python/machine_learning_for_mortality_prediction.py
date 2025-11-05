import numpy as np
import pandas as pd
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import RFE
from sklearn.metrics import (accuracy_score, roc_auc_score, f1_score, recall_score,
                             precision_score, confusion_matrix, average_precision_score)
from sklearn.preprocessing import StandardScaler
import shap

# 路径设置
output_dir = '/rfe_final'
os.makedirs(output_dir, exist_ok=True)

# 1. 数据加载
data = pd.read_csv(
    '/PSM_covariates_PA_medicine_tsfresh.csv')

# 确定标签和特征
label_col = 'label'

demographic_cols = [
    'age',  # 年龄
    'gender',  # 性别
    'ethnicity',  # 种族
    'education_level',  # 教育水平
    'Townsend_deprivation_index',  # Townsend剥夺指数
    'smoking_status',  # 吸烟状态
    'alcohol_consumption',  # 饮酒
    'disability_status',  # 残疾状态
    'overall_health_rating',  # 整体健康评分
    'comorbidities_status',  # 共病状态
    'age_at_onset',  # 癫痫发病年龄（原文关键变量）
    'charlson_index',  # Charlson共病指数（原文关键变量）
    'ASM_number',  # ASM数量（抗癫痫药物数量）
    'sodium_channel_blocking_ASM'  # 钠通道阻滞剂ASM（是否使用）
]

# PA特征（777个时频域特征）
PA_feature_start = data.columns.get_loc('PA_feature_1')
PA_feature_end = PA_feature_start + 777
PA_feature_cols = data.columns[PA_feature_start:PA_feature_end]

# 拆分特征
X_demo = data[demographic_cols]
X_PA = data[PA_feature_cols]
y = data[label_col].values

print(f"数据集大小: {len(data)} 样本")
print(f"基础特征数: {len(demographic_cols)}")
print(f"PA特征数: {len(PA_feature_cols)}")
print(f"标签分布: {np.bincount(y)}")

# 2. 划分训练集和测试集（80%训练，20%测试）
X_demo_train, X_demo_test, X_PA_train, X_PA_test, y_train, y_test = train_test_split(
    X_demo, X_PA, y, test_size=0.2, random_state=0, stratify=y
)

print(f"\n训练集大小: {len(y_train)} (正类: {sum(y_train)}, 负类: {len(y_train) - sum(y_train)})")
print(f"测试集大小: {len(y_test)} (正类: {sum(y_test)}, 负类: {len(y_test) - sum(y_test)})")

# 3. 标准化PA特征（只在训练集上fit，避免数据泄露）
scaler = StandardScaler()
X_PA_train_scaled = scaler.fit_transform(X_PA_train)
X_PA_test_scaled = scaler.transform(X_PA_test)

# 4. 使用RFE在训练集上选择最重要的8个PA特征
# 修正：按原文描述，使用基础RF模型进行特征选择
print("\n开始RFE特征选择...")
rf_for_rfe = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rfe = RFE(estimator=rf_for_rfe, n_features_to_select=8, step=1)
rfe.fit(X_PA_train_scaled, y_train)

# 获取选中的特征
PA_selected_idx = np.where(rfe.support_)[0]
PA_selected_cols = PA_feature_cols[PA_selected_idx]
print(f"RFE选择的8个PA特征: {list(PA_selected_cols)}")

# 保存选择的特征名称
pd.Series(PA_selected_cols).to_csv(os.path.join(output_dir, 'selected_PA_features.csv'), index=False)

# 5. 构建最终特征集
# 基线模型：仅使用基础特征
X_train_base = X_demo_train.values
X_test_base = X_demo_test.values

# 联合模型：基础特征 + 8个选定的PA特征
X_train_PA8 = X_PA_train_scaled[:, PA_selected_idx]
X_test_PA8 = X_PA_test_scaled[:, PA_selected_idx]
X_train_full = np.hstack([X_train_base, X_train_PA8])
X_test_full = np.hstack([X_test_base, X_test_PA8])

full_feature_names = list(X_demo.columns) + list(PA_selected_cols)

# 6. 使用GridSearchCV在训练集上优化超参数
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 8, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# 基线模型（仅基础特征）
print("\n训练基线模型（仅基础特征）...")
grid_base = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)
grid_base.fit(X_train_base, y_train)
best_base_rf = grid_base.best_estimator_
print(f"基线模型最优参数: {grid_base.best_params_}")

# 联合模型（基础特征 + PA特征）
print("\n训练联合模型（基础特征 + PA特征）...")
grid_full = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)
grid_full.fit(X_train_full, y_train)
best_full_rf = grid_full.best_estimator_
print(f"联合模型最优参数: {grid_full.best_params_}")

# 7. 在测试集上进行预测
y_pred_base = best_base_rf.predict(X_test_base)
y_prob_base = best_base_rf.predict_proba(X_test_base)[:, 1]

y_pred_full = best_full_rf.predict(X_test_full)
y_prob_full = best_full_rf.predict_proba(X_test_full)[:, 1]


# 8. 性能评估函数
def evaluate(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        'Accuracy': acc,
        'AUROC': auroc,
        'AUPRC': auprc,
        'F1': f1,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision
    }


# 9. 计算并输出性能指标
result_base = evaluate(y_test, y_pred_base, y_prob_base)
result_full = evaluate(y_test, y_pred_full, y_prob_full)

print("\n" + "=" * 60)
print("模型性能评估结果（测试集）")
print("=" * 60)

print("\n【基线模型】仅基础特征:")
for metric, value in result_base.items():
    print(f"  {metric:15s}: {value:.4f} ({value * 100:.2f}%)")

print("\n【联合模型】基础特征 + PA特征:")
for metric, value in result_full.items():
    print(f"  {metric:15s}: {value:.4f} ({value * 100:.2f}%)")

print("\n性能提升:")
print(
    f"  AUROC提升: {(result_full['AUROC'] - result_base['AUROC']) * 100:.1f}% (从 {result_base['AUROC'] * 100:.1f}% 到 {result_full['AUROC'] * 100:.1f}%)")
print(
    f"  AUPRC提升: {(result_full['AUPRC'] - result_base['AUPRC']) * 100:.1f}% (从 {result_base['AUPRC'] * 100:.1f}% 到 {result_full['AUPRC'] * 100:.1f}%)")

# 保存性能指标到CSV
perf_df = pd.DataFrame([result_base, result_full], index=['Baseline', 'Comprehensive'])
perf_df.to_csv(os.path.join(output_dir, 'model_performance.csv'))
print(f"\n性能指标已保存至: {os.path.join(output_dir, 'model_performance.csv')}")

# 10. SHAP特征重要性分析（仅保存数值，不生成图）
print("\n计算SHAP值...")
explainer = shap.TreeExplainer(best_full_rf)
shap_values = explainer.shap_values(X_test_full)

# 获取正类的SHAP值
if isinstance(shap_values, list):
    shap_values_pos = shap_values[1]
else:
    shap_values_pos = shap_values

# 计算特征重要性（平均绝对SHAP值）
mean_abs_shap = np.abs(shap_values_pos).mean(axis=0)
feature_importance = pd.DataFrame({
    'Feature': full_feature_names,
    'Mean_Abs_SHAP': mean_abs_shap
}).sort_values('Mean_Abs_SHAP', ascending=False)

print("\nTop 10 最重要特征（按SHAP值）:")
print(feature_importance.head(10).to_string(index=False))

# 保存完整的SHAP重要性
feature_importance.to_csv(os.path.join(output_dir, 'shap_feature_importance.csv'), index=False)

# 保存SHAP值矩阵
shap_df = pd.DataFrame(shap_values_pos, columns=full_feature_names)
shap_df.to_csv(os.path.join(output_dir, 'shap_values.csv'), index=False)

# 11. 保存训练好的模型
with open(os.path.join(output_dir, 'best_base_rf.pkl'), 'wb') as f:
    pickle.dump(best_base_rf, f)
with open(os.path.join(output_dir, 'best_full_rf.pkl'), 'wb') as f:
    pickle.dump(best_full_rf, f)
with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)
with open(os.path.join(output_dir, 'selected_PA_idx.pkl'), 'wb') as f:
    pickle.dump(PA_selected_idx, f)

print("\n" + "=" * 60)
print("所有结果已保存至:", output_dir)
print("=" * 60)
print("\n保存的文件:")
print("  - selected_PA_features.csv          (选择的8个PA特征)")
print("  - model_performance.csv             (模型性能指标)")
print("  - shap_feature_importance.csv       (SHAP特征重要性)")
print("  - shap_values.csv                   (SHAP值矩阵)")
print("  - best_base_rf.pkl                  (基线模型)")
print("  - best_full_rf.pkl                  (联合模型)")
print("  - scaler.pkl                        (标准化器)")
print("  - selected_PA_idx.pkl               (选择的PA特征索引)")