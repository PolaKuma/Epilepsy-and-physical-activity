import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, precision_recall_curve, auc
import sklearn.neighbors._base
from sklearn.impute import SimpleImputer
import shap

# 设置全局字体大小
plt.rcParams.update({'font.size': 30})

# 读取数据
data = pd.read_csv('sensor_features.csv')
n = 110  # 选择的特征数量

# 分离特征和标签
X = data.iloc[:, 2:]  # 从第3列到最后一列作为特征
y = data.iloc[:, 1]   # 第2列作为标签

# 特征归一化
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 特征选择
svc = SVC(kernel="linear")
rfe = RFE(estimator=svc, n_features_to_select=n, step=1)
rfe.fit(X_scaled, y)

# 获取选定的特征
selected_features = X.columns[rfe.support_]

# 创建特征重要性DataFrame
feature_importance = pd.DataFrame({
    'Feature': selected_features,
    'Importance': abs(rfe.estimator_.coef_[0])
})

# 按重要性排序
feature_importance = feature_importance.sort_values('Importance', ascending=False)

# 绘制特征重要性条形图
plt.figure(figsize=(24, 24))
plt.bar(feature_importance['Feature'], feature_importance['Importance'])
plt.xticks(rotation=90)
plt.xlabel('Selected Features')
plt.ylabel('Importance')
plt.title('Feature Importance of Selected Features')
plt.tight_layout()
#plt.savefig('E:/EP_ukb/machinelearning/Minutes_epoch/results/selected_features_importance.png', dpi=300)
plt.show()
plt.close()

# 打印选定的特征及其重要性
print("Selected Features and Their Importance:")
print(feature_importance.to_string(index=False))

# 选取选定特征
X_selected = X_scaled[:, rfe.support_]

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

# 参数选择
parameters = {'kernel': ['linear', 'rbf'], 'C': [0.1, 1, 10], 'gamma': ['scale', 'auto']}
clf = GridSearchCV(SVC(probability=True), parameters, scoring='accuracy', cv=5, n_jobs=-1)
clf.fit(X_train, y_train)

print("Best parameters:", clf.best_params_)

# 在训练集和测试集上进行预测
y_train_pred = clf.predict(X_train)
y_test_pred = clf.predict(X_test)
y_train_proba = clf.predict_proba(X_train)[:, 1]
y_test_proba = clf.predict_proba(X_test)[:, 1]

# 计算并打印评价指标
def print_metrics(y_true, y_pred, y_proba, set_name):
    print(f"\n{set_name} Metrics:")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"AUC: {roc_auc_score(y_true, y_proba):.4f}")
    print(f"Sensitivity (Recall): {recall_score(y_true, y_pred):.4f}")
    print(f"Specificity: {recall_score(y_true, y_pred, pos_label=0):.4f}")
    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")

print_metrics(y_train, y_train_pred, y_train_proba, "Train")
print_metrics(y_test, y_test_pred, y_test_proba, "Test")

# 绘制并保存混淆矩阵
def plot_confusion_matrix(y_true, y_pred, set_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 16))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    classes = np.unique(y)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix - {set_name} Set')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{set_name.lower()}.png', dpi=300)
    plt.close()

plot_confusion_matrix(y_train, y_train_pred, "Train")
plot_confusion_matrix(y_test, y_test_pred, "Test")

# 打印分类报告
print("\nTrain Set Classification Report:")
print(classification_report(y_train, y_train_pred))
print("\nTest Set Classification Report:")
print(classification_report(y_test, y_test_pred))

# # 绘制ROC曲线
# plt.figure(figsize=(20, 16))
# fpr_train, tpr_train, _ = roc_curve(y_train, y_train_proba)
# fpr_test, tpr_test, _ = roc_curve(y_test, y_test_proba)
# plt.plot(fpr_train, tpr_train, color='#8A4210', lw=4, label=f'Train ROC (AUC = {auc(fpr_train, tpr_train):.2f})')
# plt.plot(fpr_test, tpr_test, color='#629FCA', lw=4, label=f'Test ROC (AUC = {auc(fpr_test, tpr_test):.2f})')
# plt.fill_between(fpr_test, tpr_test, alpha=0.3, color='#629FCA')
# plt.plot([0, 1], [0, 1], color='navy', lw=4, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic (ROC) Curve')
# plt.legend(loc="lower right")
# plt.savefig('roc_curve.png', dpi=300)
# plt.close()
#
# # 绘制PR曲线
# plt.figure(figsize=(20, 16))
# precision_train, recall_train, _ = precision_recall_curve(y_train, y_train_proba)
# precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_proba)
# plt.plot(recall_train, precision_train, color='#8A4210', lw=4, label=f'Train PR (AP = {auc(recall_train, precision_train):.2f})')
# plt.plot(recall_test, precision_test, color='#629FCA', lw=4, label=f'Test PR (AP = {auc(recall_test, precision_test):.2f})')
# plt.fill_between(recall_test, precision_test, alpha=0.3, color='#629FCA')
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.title('Precision-Recall Curve')
# plt.legend(loc="lower left")
# plt.savefig('pr_curve.png', dpi=300)
# plt.close()
#
# # SHAP值计算和可视化
# background = shap.sample(X_train, 100)  # 使用100个样本作为背景
#
# def f(X):
#     return clf.predict_proba(X)[:, 1]
#
# explainer = shap.KernelExplainer(f, background)
#
# n_explain = min(100, X_test.shape[0])  # 解释最多100个样本
# shap_values = explainer.shap_values(X_test[:n_explain], nsamples=100)
#
# plt.figure(figsize=(34, 18))
# shap.summary_plot(shap_values, X_test[:n_explain], plot_type="bar", feature_names=list(selected_features), show=False)
# plt.title('SHAP Feature Importance (Bar Plot)')
# plt.tight_layout()
# plt.savefig('shap_summary_bar.png', dpi=300)
# plt.close()
#
# plt.figure(figsize=(34, 18))
# shap.summary_plot(shap_values, X_test[:n_explain], feature_names=list(selected_features), show=False)
# plt.title('SHAP Feature Importance (Dot Plot)')
# plt.tight_layout()
# plt.savefig('shap_summary_dot.png', dpi=300)
# plt.close()
#
# # SHAP 瀑布图 (第一个样本)
# plt.figure(figsize=(34, 18))
# shap.plots.waterfall(shap.Explanation(values=shap_values[0], base_values=explainer.expected_value, data=X_test[0], feature_names=list(selected_features)), show=False)
# plt.title('SHAP Waterfall Plot (First Sample)')
# plt.tight_layout()
# plt.savefig('shap_waterfall.png', dpi=300)
# plt.close()
#
# # SHAP 力图 (前几个样本)
# plt.figure(figsize=(60, 20))
# shap.plots.force(shap.Explanation(values=shap_values[:10], base_values=explainer.expected_value, data=X_test[:10], feature_names=list(selected_features)), show=False)
# plt.title('SHAP Force Plot (First 10 Samples)')
# plt.tight_layout()
# plt.savefig('shap_force.png', dpi=300)
# plt.close()
#
# print("All analyses completed and results saved.")