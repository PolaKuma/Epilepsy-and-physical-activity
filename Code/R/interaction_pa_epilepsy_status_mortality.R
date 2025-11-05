# 加载必要的包
{library(survival)
  library(dplyr)
  library(broom)
  library(knitr)
  library(tidyr)
  library(stringr)
  library(ggplot2)
  library(gridExtra)
  library(forestplot)
  # 安装并加载qvalue包
  if (!requireNamespace("qvalue", quietly = TRUE)) {
    install.packages("qvalue")
  }
  library(qvalue)}


# 读取数据
data <- read.csv("F:/Research_Project/EP_ukb/Revision_Interaction_Effect_between_Epilepsy_and_PA_on_Mortality_Risk/combined_epilepsy_data.csv")


# 定义协变量组
covariates_demo <- c("gender", "education_level", "smoking", "alcohol_use", "ethic", "BMI", "Townsend_index", "enrollment_age")
covariates_health <- c("Disable_long", "charlson_score", "overall_health")
covariates_pa <- c("Light_Total_Hour", "MVPA_Total_Hour", "Sedentary_Total_Hour")
covariates_ep <- c("epilepsy_status")
covariates_lightPA <- c("Light_Total_Hour")
covariates_sedentary <- c("Sedentary_Total_Hour")
covariates_MVPA <- c("MVPA_Total_Hour")

####合并所有协变量
all_vars <- c(covariates_ep, covariates_pa)

# 函数：执行Cox模型并提取结果
run_cox_model <- function(data, var_name, adjust_vars = NULL, time_var = "Followingyears_birthday", status_var = "is_dead") {
  # 构建公式
  if (is.null(adjust_vars)) {
    # 单变量模型
    formula_str <- paste0("Surv(", time_var, ", ", status_var, ") ~ ", var_name)
  } else {
    # 多变量模型
    # 排除当前分析的变量（如果它在adjust_vars中）
    adjust_vars <- setdiff(adjust_vars, var_name)
    formula_str <- paste0("Surv(", time_var, ", ", status_var, ") ~ ", 
                          paste(c(var_name, adjust_vars), collapse = " + "))
  }
  
  # 运行Cox模型
  model <- coxph(as.formula(formula_str), data = data)
  
  # 提取结果
  result <- tidy(model, exponentiate = TRUE, conf.int = TRUE)
  
  # 提取当前变量的结果
  var_result <- result %>% dplyr::filter(term == var_name)
  
  if (nrow(var_result) == 0) {
    # 如果变量是分类变量，可能有多行结果
    var_result <- result %>% dplyr::filter(grepl(paste0("^", var_name), term))
  }
  
  return(var_result)
}

# 初始化结果数据框
results <- data.frame(Variable = character(),
                      Model = character(),
                      HR = numeric(),
                      CI_lower = numeric(),
                      CI_upper = numeric(),
                      P_value = numeric(),
                      stringsAsFactors = FALSE)

# 执行单变量分析
cat("执行单变量Cox分析...\n")
for (var in all_vars) {
  # 跳过有太多缺失值的变量
  if (sum(is.na(data[[var]])) / nrow(data) > 0.5) {
    cat(paste0("跳过变量 ", var, " (>50% 缺失值)\n"))
    next
  }
  
  var_result <- run_cox_model(data, var)
  
  # 添加到结果数据框
  for (i in 1:nrow(var_result)) {
    results <- rbind(results, data.frame(
      Variable = ifelse(nrow(var_result) > 1, var_result$term[i], var),
      Model = "单变量",
      HR = round(var_result$estimate[i], 2),
      CI_lower = round(var_result$conf.low[i], 2),
      CI_upper = round(var_result$conf.high[i], 2),
      P_value = var_result$p.value[i],
      stringsAsFactors = FALSE
    ))
  }
}

# 执行多变量分析
cat("执行多变量Cox分析...\n")
for (var in all_vars) {
  # 跳过有太多缺失值的变量
  if (sum(is.na(data[[var]])) / nrow(data) > 0.5) next
  
  # 对于每个变量，调整所有其他变量
  var_result <- run_cox_model(data, var, all_vars)
  
  # 添加到结果数据框
  for (i in 1:nrow(var_result)) {
    results <- rbind(results, data.frame(
      Variable = ifelse(nrow(var_result) > 1, var_result$term[i], var),
      Model = "多变量",
      HR = round(var_result$estimate[i], 2),
      CI_lower = round(var_result$conf.low[i], 2),
      CI_upper = round(var_result$conf.high[i], 2),
      P_value = var_result$p.value[i],
      stringsAsFactors = FALSE
    ))
  }
}

# =============================================
# 仅对PA变量进行多重校正（包含Q值）
# =============================================
cat("仅对身体活动变量执行多重比较校正...\n")

# 将结果分为PA变量和非PA变量
pa_results <- results %>%
  dplyr::filter(Variable %in% covariates_pa)

non_pa_results <- results %>%
  dplyr::filter(!(Variable %in% covariates_pa))

# 只对PA变量进行多重校正
pa_results_corrected <- pa_results %>%
  dplyr::group_by(Model) %>%
  dplyr::mutate(
    # Bonferroni校正
    P_bonferroni = pmin(P_value * n(), 1),
    
    # Benjamini-Hochberg校正（FDR）
    P_BH = p.adjust(P_value, method = "BH"),
    
    # Holm校正
    P_holm = p.adjust(P_value, method = "holm"),
    
    # Storey的q值校正（通常比BH更宽松）
    q_value = tryCatch({
      qobj <- qvalue(P_value)
      qobj$qvalues
    }, error = function(e) {
      # 如果q值计算失败（通常因为样本太小），退回到BH方法
      warning("Q值计算失败，使用BH方法代替")
      p.adjust(P_value, method = "BH")
    }),
    
    # 标记显著性
    Significance = dplyr::case_when(
      P_bonferroni < 0.05 ~ "显著 (Bonferroni校正后)",
      P_BH < 0.05 ~ "显著 (BH校正后)",
      q_value < 0.05 ~ "显著 (q值校正后)",
      P_value < 0.05 ~ "原始P值显著",
      TRUE ~ "不显著"
    )
  ) %>%
  dplyr::ungroup()

# 为非PA变量保留原始P值，不进行校正
non_pa_results <- non_pa_results %>%
  dplyr::mutate(
    P_bonferroni = NA,
    P_BH = NA,
    P_holm = NA,
    q_value = NA,
    Significance = dplyr::case_when(
      P_value < 0.05 ~ "原始P值显著",
      TRUE ~ "不显著"
    )
  )

# 合并结果
results_corrected <- bind_rows(pa_results_corrected, non_pa_results) %>%
  dplyr::mutate(
    CI = paste0(HR, " (", CI_lower, "-", CI_upper, ")"),
    P_value_formatted = dplyr::case_when(
      P_value < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", P_value)
    ),
    P_bonferroni_formatted = dplyr::case_when(
      is.na(P_bonferroni) ~ "NA",
      P_bonferroni < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", P_bonferroni)
    ),
    P_BH_formatted = dplyr::case_when(
      is.na(P_BH) ~ "NA",
      P_BH < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", P_BH)
    ),
    q_value_formatted = dplyr::case_when(
      is.na(q_value) ~ "NA",
      q_value < 0.001 ~ "<0.001",
      TRUE ~ sprintf("%.3f", q_value)
    )
  )

# 创建分类变量和连续变量的标签
results_corrected <- results_corrected %>%
  dplyr::mutate(
    Variable_Type = dplyr::case_when(
      Variable %in% continuous_vars ~ "连续变量",
      TRUE ~ "分类变量"
    ),
    Category = dplyr::case_when(
      Variable %in% covariates_demo ~ "人口学变量",
      Variable %in% covariates_health ~ "健康状况变量",
      Variable %in% covariates_pa ~ "身体活动变量",
      TRUE ~ "其他变量"
    )
  )

# 创建输出目录
dir.create("./", showWarnings = FALSE)

# 按不同方式整理结果
# 1. 按变量分组
cat("\n按变量分组的Cox模型结果（PA变量已校正）：\n")
results_by_var <- results_corrected %>%
  dplyr::select(Category, Variable, Model, CI, P_value_formatted, P_bonferroni_formatted, P_BH_formatted, q_value_formatted, Significance) %>%
  tidyr::pivot_wider(names_from = Model, values_from = c(CI, P_value_formatted, P_bonferroni_formatted, P_BH_formatted, q_value_formatted, Significance))

print(head(results_by_var))

# 2. 按单变量和多变量分组
cat("\n按分析模型分组的Cox模型结果（PA变量已校正）：\n")
results_by_model <- results_corrected %>%
  dplyr::arrange(Model, Category, Variable) %>%
  dplyr::select(Model, Category, Variable, HR, CI_lower, CI_upper, P_value_formatted, P_bonferroni_formatted, P_BH_formatted, q_value_formatted, Significance)

print(head(results_by_model))

# 保存结果
write.csv(results_corrected, "./cox_results_all_corrected.csv", row.names = FALSE)
write.csv(results_by_var, "./cox_results_by_variable_corrected.csv", row.names = FALSE)
write.csv(results_by_model, "./cox_results_by_model_corrected.csv", row.names = FALSE)

# 为身体活动变量创建详细的结果表（包含所有校正方法）
pa_table_detailed <- pa_results_corrected %>%
  dplyr::arrange(Model, Variable) %>%
  dplyr::mutate(
    CI = paste0(HR, " [", CI_lower, ", ", CI_upper, "]"),
    Significant_Original = ifelse(P_value < 0.05, "是", "否"),
    Significant_Bonferroni = ifelse(P_bonferroni < 0.05, "是", "否"),
    Significant_BH = ifelse(P_BH < 0.05, "是", "否"),
    Significant_q = ifelse(q_value < 0.05, "是", "否"),
    # 增加边缘显著性标记
    Marginal_Significant = dplyr::case_when(
      P_value < 0.05 ~ "显著 (p<0.05)",
      P_value < 0.10 ~ "边缘显著 (p<0.10)",
      TRUE ~ "不显著"
    )
  ) %>%
  dplyr::select(
    Variable, Model, HR, CI, 
    P_value, P_bonferroni, P_BH, q_value,
    Significant_Original, Significant_Bonferroni, Significant_BH, Significant_q,
    Marginal_Significant
  )

# 输出PA变量表格
cat("\n身体活动变量的Cox回归结果（含多种校正方法）：\n")
print(knitr::kable(pa_table_detailed %>% 
                     dplyr::select(Variable, Model, HR, CI, P_value, P_bonferroni, P_BH, q_value),
                   caption = "身体活动变量的Cox回归结果（含多种校正方法）"))

# 保存PA变量详细结果
write.csv(pa_table_detailed, "./physical_activity_cox_results_ep_corrections.csv", row.names = FALSE)

# 创建一个简洁版表格用于发表
pa_table_publication <- pa_table_detailed %>%
  dplyr::select(Variable, Model, HR, CI, P_value, P_BH, q_value) %>%
  dplyr::mutate(
    P_value = ifelse(P_value < 0.001, "<0.001", sprintf("%.3f", P_value)),
    P_BH = ifelse(P_BH < 0.001, "<0.001", sprintf("%.3f", P_BH)),
    q_value = ifelse(q_value < 0.001, "<0.001", sprintf("%.3f", q_value))
  )


# 输出边缘显著性分析结果
cat("\n考虑边缘显著性(p<0.10)的结果：\n")
marginal_table <- pa_table_detailed %>%
  dplyr::select(Variable, Model, HR, CI, P_value, q_value, Marginal_Significant) %>%
  dplyr::mutate(
    P_value = ifelse(P_value < 0.001, "<0.001", sprintf("%.3f", P_value)),
    q_value = ifelse(q_value < 0.001, "<0.001", sprintf("%.3f", q_value))
  )

print(knitr::kable(marginal_table,
                   caption = "考虑边缘显著性的身体活动变量结果"))

cat("\n分析完成！结果已保存到 E:/phd_project/EP_ukb/Revision_cox/ 目录\n")
cat("\n注意：请检查q值校正结果。q值方法通常比BH方法更宽松，有助于发现更多潜在关联。\n")

