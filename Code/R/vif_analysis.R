# 加载所需的包
install.packages("car")
install.packages("ggplot2")
installed.packages("corrplot")
installed.packages("broom")
installed.packages("openxlsx")
library(car)      # 用于VIF计算
library(ggplot2)  # 用于可视化
library(corrplot) # 用于相关矩阵可视化
library(dplyr)    # 用于数据处理
library(broom)    # 用于整理模型结果
library(openxlsx) # 用于Excel输出

# 读取数据
# 请替换为您的文件路径
file_path <- "F:/Research_Project/EP_ukb/Revision_VIF/nGP_combines_G40_hour_imputed.csv"
data <- read.csv(file_path)

# 检查数据
cat("数据集前5行：\n")
print(head(data))

# 提取我们关注的变量
pa_variables <- c("Sedentary_Total_Hour", "MVPA_Total_Hour", "Light_Total_Hour")

# 1. 基本统计描述
cat("\n基本统计描述：\n")
desc_stats <- summary(data[pa_variables])
print(desc_stats)

# 更详细的描述性统计
detailed_stats <- data.frame(
  变量 = pa_variables,
  平均值 = sapply(data[pa_variables], mean, na.rm = TRUE),
  标准差 = sapply(data[pa_variables], sd, na.rm = TRUE),
  中位数 = sapply(data[pa_variables], median, na.rm = TRUE),
  最小值 = sapply(data[pa_variables], min, na.rm = TRUE),
  最大值 = sapply(data[pa_variables], max, na.rm = TRUE)
)
print(detailed_stats)

# 2. 相关性分析
cat("\n相关性分析：\n")
correlation_matrix <- cor(data[pa_variables], use = "complete.obs")
print(correlation_matrix)

# 相关性矩阵p值计算
cor_test_matrix <- function(data) {
  n <- ncol(data)
  p_matrix <- matrix(NA, n, n)
  colnames(p_matrix) <- colnames(data)
  rownames(p_matrix) <- colnames(data)
  
  for(i in 1:n) {
    for(j in 1:n) {
      if(i != j) {
        test <- cor.test(data[,i], data[,j])
        p_matrix[i,j] <- test$p.value
      } else {
        p_matrix[i,j] <- 1
      }
    }
  }
  return(p_matrix)
}

correlation_p_values <- cor_test_matrix(data[pa_variables])
print("相关系数的p值矩阵：")
print(correlation_p_values)

# 绘制相关性热图
png("correlation_heatmap.png", width = 800, height = 600)
corrplot(correlation_matrix, method = "color", type = "upper", 
         addCoef.col = "black", tl.col = "black", tl.srt = 45,
         title = "PA指标之间的相关性矩阵", mar = c(0, 0, 1, 0))
dev.off()

# 3. 计算VIF值
# 首先确保没有缺失值
data_clean <- na.omit(data[pa_variables])

# 使用线性回归模型计算VIF
cat("\n方差膨胀因子(VIF)分析：\n")

# 创建一个数据框来存储VIF结果
vif_results <- data.frame(变量 = character(), VIF = numeric(), stringsAsFactors = FALSE)

for(var in pa_variables) {
  # 构建公式：当前变量 ~ 其他所有变量
  predictors <- pa_variables[pa_variables != var]
  formula_str <- paste(var, "~", paste(predictors, collapse = " + "))
  formula_obj <- as.formula(formula_str)
  
  # 拟合模型
  model <- lm(formula_obj, data = data_clean)
  
  # 计算VIF (VIF = 1/(1-R²))
  r_squared <- summary(model)$r.squared
  vif_value <- 1 / (1 - r_squared)
  
  # 添加到结果
  vif_results <- rbind(vif_results, data.frame(变量 = var, VIF = vif_value))
}

# 显示VIF结果
print(vif_results)

# 4. 线性回归分析
cat("\n线性回归分析以评估多重共线性：\n")

# 创建一个列表来存储回归结果
regression_results <- list()
ci_results <- list()
p_values <- list()
regression_summary <- data.frame()

for(var in pa_variables) {
  # 构建模型
  predictors <- pa_variables[pa_variables != var]
  formula_str <- paste(var, "~", paste(predictors, collapse = " + "))
  formula_obj <- as.formula(formula_str)
  
  model <- lm(formula_obj, data = data_clean)
  
  # 存储结果
  regression_results[[var]] <- summary(model)
  ci_results[[var]] <- confint(model, level = 0.95)
  p_values[[var]] <- summary(model)$coefficients[, 4]
  
  # 打印摘要
  cat(paste("\n当", var, "作为因变量时：\n"))
  print(summary(model))
  
  # 收集回归结果
  temp_result <- tidy(model) %>%
    mutate(dependent_var = var) %>%
    left_join(tidy(confint(model)) %>% rename(term = .rownames), by = "term")
  
  regression_summary <- rbind(regression_summary, temp_result)
}

# 5. 保存综合结果到CSV
results_df <- data.frame(
  变量 = pa_variables,
  平均值 = sapply(data_clean[pa_variables], mean),
  标准差 = sapply(data_clean[pa_variables], sd)
)

# 添加R²值和调整R²
for(var in pa_variables) {
  predictors <- pa_variables[pa_variables != var]
  formula_str <- paste(var, "~", paste(predictors, collapse = " + "))
  formula_obj <- as.formula(formula_str)
  model <- lm(formula_obj, data = data_clean)
  
  results_df$R2[results_df$变量 == var] <- summary(model)$r.squared
  results_df$调整R2[results_df$变量 == var] <- summary(model)$adj.r.squared
}

# 添加VIF值
results_df <- merge(results_df, vif_results, by = "变量")

# 判断多重共线性
results_df$多重共线性判断 <- ifelse(results_df$VIF > 10, "严重", 
                             ifelse(results_df$VIF > 5, "中等", "较低"))

# 保存结果
write.csv(results_df, "F:/Research_Project/EP_ukb/Revision_VIF/VIF_EP/PA_multicollinearity_results.csv", row.names = FALSE)
cat("\n综合结果已保存至 'PA_multicollinearity_results.csv'\n")

# 6. 图形化展示
# VIF条形图
ggplot(vif_results, aes(x = 变量, y = VIF)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  geom_hline(yintercept = 5, color = "red", alpha = 0.3) +
  geom_hline(yintercept = 10, color = "red", alpha = 0.7) +
  annotate("text", x = length(pa_variables), y = 5.5, label = "VIF=5 (中等多重共线性阈值)", color = "red", hjust = 1) +
  annotate("text", x = length(pa_variables), y = 10.5, label = "VIF=10 (严重多重共线性阈值)", color = "red", hjust = 1) +
  labs(title = "PA指标的方差膨胀因子", x = "变量", y = "VIF值") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("vif_values.png", width = 10, height = 6)

# 7. 创建一个详细的Excel文件
wb <- createWorkbook()

# 基本统计量
addWorksheet(wb, "基本统计量")
writeData(wb, "基本统计量", detailed_stats)

# 相关性矩阵
addWorksheet(wb, "相关性矩阵")
writeData(wb, "相关性矩阵", as.data.frame(correlation_matrix))

# 相关性p值
addWorksheet(wb, "相关性P值")
writeData(wb, "相关性P值", as.data.frame(correlation_p_values))

# VIF分析
addWorksheet(wb, "VIF分析")
writeData(wb, "VIF分析", vif_results)

# 综合结果
addWorksheet(wb, "综合结果")
writeData(wb, "综合结果", results_df)

# 线性回归结果
addWorksheet(wb, "线性回归结果")
writeData(wb, "线性回归结果", regression_summary)

# 为每个变量的回归分析创建单独的工作表
for(var in pa_variables) {
  sheet_name <- paste0(substr(var, 1, 10), "回归分析")
  addWorksheet(wb, sheet_name)
  
  # 获取模型系数和统计量
  model_summary <- regression_results[[var]]$coefficients
  model_summary_df <- as.data.frame(model_summary)
  
  # 获取置信区间
  model_ci <- ci_results[[var]]
  model_ci_df <- as.data.frame(model_ci)
  colnames(model_ci_df) <- c("2.5%", "97.5%")
  
  # 合并结果
  combined_results <- cbind(model_summary_df, model_ci_df)
  
  # 写入工作表
  writeData(wb, sheet_name, combined_results)
}

# 保存Excel文件
saveWorkbook(wb, "F:/Research_Project/EP_ukb/Revision_VIF/VIF_EP/PA_multicollinearity_detailed_results.xlsx", overwrite = TRUE)
cat("\n详细结果已保存至 'PA_multicollinearity_detailed_results.xlsx'\n")