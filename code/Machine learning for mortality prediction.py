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
import matplotlib.pyplot as plt

# 路径设置
output_dir = 'F:/Research_Project/EP_ukb/Revision_machine_learning/Results_medicine/rfe_final'
os.makedirs(output_dir, exist_ok=True)

# 1. 数据加载
data = pd.read_csv('F:/Research_Project/EP_ukb/Revision_machine_learning/Results_medicine/PSM_covariates_PA_medicine_tsfresh.csv')

# 确定标签和特征
label_col = 'label'  # 标签列名
demographic_cols = [
    'age', 'gender', 'ethnicity', 'education_level', 'Townsend_deprivation_index',
    'smoking_status', 'alcohol_consumption', 'disability_status', 'overall_health_rating',
    'comorbidities_status', 'ASM_1' ,'ASM_2'
]
# PA特征起始列
PA_feature_start = data.columns.get_loc('PA_feature_1')
PA_feature_end = PA_feature_start + 777
PA_feature_cols = data.columns[PA_feature_start:PA_feature_end]

# 拆分特征
X_demo = data[demographic_cols]
X_PA = data[PA_feature_cols]
y = data[label_col].values

# 标准化PA特征（仅PA部分）
scaler = StandardScaler()
X_PA_scaled = scaler.fit_transform(X_PA)

# 2. 划分训练/测试集（确保PSM后为平衡数据）
X_demo_train, X_demo_test, X_PA_train, X_PA_test, y_train, y_test = train_test_split(
    X_demo, X_PA_scaled, y, test_size=0.2, random_state=0, stratify=y
)

# 3. RFE+GridSearchCV在训练集做特征选择与调参
# 超参数网格
param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [4, 8, 12, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid = GridSearchCV(rf, param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid.fit(X_PA_train, y_train)
best_rf_for_rfe = grid.best_estimator_


rfe = RFE(best_rf_for_rfe, n_features_to_select=8)
rfe.fit(X_PA_train, y_train)
PA_selected_idx = np.where(rfe.support_)[0]
PA_selected_cols = PA_feature_cols[PA_selected_idx]
print("RFE选择的8个PA特征：", list(PA_selected_cols))

# 保存特征名
pd.Series(PA_selected_cols).to_csv(os.path.join(output_dir, 'selected_PA_features.csv'), index=False)

# 4. 再次联合全部特征（基础+8个PA），用于后续建模
X_train_base = X_demo_train.values
X_test_base = X_demo_test.values
X_train_PA8 = X_PA_train[:, PA_selected_idx]
X_test_PA8 = X_PA_test[:, PA_selected_idx]
X_train_full = np.hstack([X_train_base, X_train_PA8])
X_test_full = np.hstack([X_test_base, X_test_PA8])

full_feature_names = list(X_demo.columns) + list(PA_selected_cols)

# 5. 在训练集上用GridSearchCV调最优随机森林参数
grid_base = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_base.fit(X_train_base, y_train)
best_base_rf = grid_base.best_estimator_

grid_full = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
grid_full.fit(X_train_full, y_train)
best_full_rf = grid_full.best_estimator_

# 6. 训练最终模型并在测试集评估
# 基线模型
best_base_rf.fit(X_train_base, y_train)
y_pred_base = best_base_rf.predict(X_test_base)
y_prob_base = best_base_rf.predict_proba(X_test_base)[:, 1]

# 联合模型
best_full_rf.fit(X_train_full, y_train)
y_pred_full = best_full_rf.predict(X_test_full)
y_prob_full = best_full_rf.predict_proba(X_test_full)[:, 1]

# 7. 性能评估函数
def evaluate(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    auc_ = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    specificity = tn / (tn + fp)
    return {
        'Accuracy': acc,
        'AUROC': auc_,
        'AUPRC': auprc,
        'F1': f1,
        'Sensitivity': recall,
        'Specificity': specificity,
        'Precision': precision
    }

result_base = evaluate(y_test, y_pred_base, y_prob_base)
result_full = evaluate(y_test, y_pred_full, y_prob_full)
print("基础模型（仅基础特征）性能：", result_base)
print("联合模型（基础+PA特征）性能：", result_full)
pd.DataFrame([result_base, result_full], index=['Base', 'Full']).to_csv(os.path.join(output_dir, 'model_performance.csv'))

# 8. SHAP分析（只对联合模型）
explainer = shap.TreeExplainer(best_full_rf)
shap_values = explainer.shap_values(X_test_full)
shap.summary_plot(shap_values[1], X_test_full, feature_names=full_feature_names,
                  show=False, plot_type='bar')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary_bar.png'), dpi=300)
plt.close()

shap.summary_plot(shap_values[1], X_test_full, feature_names=full_feature_names,
                  show=False, plot_type='dot')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'shap_summary_dot.png'), dpi=300)
plt.close()

# 保存模型
with open(os.path.join(output_dir, 'best_base_rf.pkl'), 'wb') as f:
    pickle.dump(best_base_rf, f)
with open(os.path.join(output_dir, 'best_full_rf.pkl'), 'wb') as f:
    pickle.dump(best_full_rf, f)
with open(os.path.join(output_dir, 'scaler.pkl'), 'wb') as f:
    pickle.dump(scaler, f)

print("所有结果已保存到：", output_dir)