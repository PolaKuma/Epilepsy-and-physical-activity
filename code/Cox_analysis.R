# 加载必要的包
library(survival)
library(dplyr)
library(broom)
library(knitr)
library(tidyr)
library(stringr)
library(tools) # file_path_sans_ext

# =======================
# 变量类型明确定义
# =======================
# 连续变量
continuous_vars <- c(
  "Light_Total_Hour", "MVPA_Total_Hour", "Sedentary_Total_Hour",
  "BMI", "Townsend_index", "enrollment_age", "Sleep_Total_Hour"
)
# 分类变量
categorical_vars <- c(
  "gender", "education_level", "smoking", "alcohol_use",
  "ethic", "Disable_long", "charlson_score", "overall_health"
)

# =======================
# 主函数：批量处理CSV文件
# =======================
process_csv_files <- function(input_dir, output_dir) {
  csv_files <- list.files(path = input_dir, pattern = "\\.csv$", full.names = TRUE)
  if(length(csv_files) == 0) {
    cat("错误：指定目录中没有找到CSV文件\n")
    return()
  }
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    cat(paste0("创建输出目录: ", output_dir, "\n"))
  }
  cat(paste0("找到 ", length(csv_files), " 个CSV文件需要处理\n"))
  for(i in seq_along(csv_files)) {
    csv_path <- csv_files[i]
    csv_name <- file_path_sans_ext(basename(csv_path))
    file_output_dir <- file.path(output_dir, csv_name)
    if(!dir.exists(file_output_dir)) dir.create(file_output_dir, recursive = TRUE)
    cat(paste0("\n[", i, "/", length(csv_files), "] 处理文件: ", csv_name, "\n"))
    process_single_csv(csv_path, file_output_dir)
  }
  cat("\n全部分析完成！每个文件的结果已保存到相应的子目录中\n")
}

# =======================
# 处理单个CSV文件的函数
# =======================
process_single_csv <- function(file_path, output_dir) {
  cat("读取数据...\n")
  data <- read.csv(file_path)
  
  # 定义协变量组
  covariates_demo   <- c("gender", "education_level", "smoking", "alcohol_use", "ethic", "BMI", "Townsend_index", "enrollment_age")
  covariates_health <- c("Disable_long", "charlson_score", "overall_health")
  covariates_pa     <- c("Light_Total_Hour", "MVPA_Total_Hour", "Sedentary_Total_Hour")
  
  all_vars <- c(covariates_demo, covariates_health, covariates_pa)
  # 检查实际存在的变量
  existing_vars   <- all_vars[all_vars %in% colnames(data)]
  existing_demo   <- covariates_demo[covariates_demo %in% colnames(data)]
  existing_health <- covariates_health[covariates_health %in% colnames(data)]
  existing_pa     <- covariates_pa[covariates_pa %in% colnames(data)]
  
  if(length(existing_vars) < length(all_vars)) {
    missing_vars <- all_vars[!all_vars %in% colnames(data)]
    cat(paste0("警告: 以下变量在数据中不存在: ", paste(missing_vars, collapse=", "), "\n"))
    all_vars <- existing_vars
  }
  
  # =======================
  # Cox模型函数
  # =======================
  run_cox_model <- function(data, var_name, adjust_vars = NULL, time_var = "Followingyears_birthday", status_var = "is_dead") {
    if(!all(c(time_var, status_var) %in% colnames(data))) {
      cat(paste0("错误: 生存时间变量('", time_var, "')或事件状态变量('", status_var, "')不存在\n"))
      return(NULL)
    }
    if (is.null(adjust_vars)) {
      formula_str <- paste0("Surv(", time_var, ", ", status_var, ") ~ ", var_name)
    } else {
      adjust_vars <- setdiff(adjust_vars, var_name)
      formula_str <- paste0("Surv(", time_var, ", ", status_var, ") ~ ", 
                            paste(c(var_name, adjust_vars), collapse = " + "))
    }
    tryCatch({
      model <- coxph(as.formula(formula_str), data = data)
      result <- tidy(model, exponentiate = TRUE, conf.int = TRUE)
      var_result <- result %>% dplyr::filter(term == var_name)
      if (nrow(var_result) == 0) {
        var_result <- result %>% dplyr::filter(grepl(paste0("^", var_name), term))
      }
      return(var_result)
    }, error = function(e) {
      cat(paste0("错误: 变量 '", var_name, "' 的Cox模型计算失败: ", e$message, "\n"))
      return(NULL)
    })
  }
  
  # =======================
  # 结果数据框(含类型)
  # =======================
  get_var_type <- function(var) {
    if(var %in% continuous_vars) return("连续变量")
    if(var %in% categorical_vars) return("分类变量")
    return("未知")
  }
  get_var_category <- function(var) {
    if(var %in% covariates_demo) return("人口学变量")
    if(var %in% covariates_health) return("健康状况变量")
    if(var %in% covariates_pa) return("身体活动变量")
    return("其他变量")
  }
  results <- data.frame(
    Variable = character(),
    Model = character(),
    HR = numeric(),
    CI_lower = numeric(),
    CI_upper = numeric(),
    P_value = numeric(),
    Variable_Type = character(),
    Category = character(),
    stringsAsFactors = FALSE
  )
  
  # 单变量Cox
  cat("执行单变量Cox分析（模型1）...\n")
  for (var in all_vars) {
    var_result <- run_cox_model(data, var)
    if(is.null(var_result)) next
    for (i in 1:nrow(var_result)) {
      results <- rbind(results, data.frame(
        Variable = ifelse(nrow(var_result) > 1, var_result$term[i], var),
        Model = "模型1: 单变量",
        HR = round(var_result$estimate[i], 2),
        CI_lower = round(var_result$conf.low[i], 2),
        CI_upper = round(var_result$conf.high[i], 2),
        P_value = var_result$p.value[i],
        Variable_Type = get_var_type(var),
        Category = get_var_category(var),
        stringsAsFactors = FALSE
      ))
    }
  }
  # 人口学调整
  cat("执行人口学调整Cox分析（模型2）...\n")
  for (var in all_vars) {
    var_result <- run_cox_model(data, var, existing_demo)
    if(is.null(var_result)) next
    for (i in 1:nrow(var_result)) {
      results <- rbind(results, data.frame(
        Variable = ifelse(nrow(var_result) > 1, var_result$term[i], var),
        Model = "模型2: 人口学调整",
        HR = round(var_result$estimate[i], 2),
        CI_lower = round(var_result$conf.low[i], 2),
        CI_upper = round(var_result$conf.high[i], 2),
        P_value = var_result$p.value[i],
        Variable_Type = get_var_type(var),
        Category = get_var_category(var),
        stringsAsFactors = FALSE
      ))
    }
  }
  # 全变量调整
  cat("执行全变量调整Cox分析（模型3）...\n")
  for (var in all_vars) {
    var_result <- run_cox_model(data, var, all_vars)
    if(is.null(var_result)) next
    for (i in 1:nrow(var_result)) {
      results <- rbind(results, data.frame(
        Variable = ifelse(nrow(var_result) > 1, var_result$term[i], var),
        Model = "模型3: 全变量调整",
        HR = round(var_result$estimate[i], 2),
        CI_lower = round(var_result$conf.low[i], 2),
        CI_upper = round(var_result$conf.high[i], 2),
        P_value = var_result$p.value[i],
        Variable_Type = get_var_type(var),
        Category = get_var_category(var),
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # 仅对PA变量进行Benjamini-Hochberg校正
  cat("对身体活动变量执行FDR校正...\n")
  pa_vars <- covariates_pa[covariates_pa %in% all_vars]
  pa_results <- results %>% filter(Variable %in% pa_vars | grepl(paste(paste0("^", pa_vars), collapse="|"), Variable))
  non_pa_results <- results %>% filter(!(Variable %in% pa_vars | grepl(paste(paste0("^", pa_vars), collapse="|"), Variable)))
  pa_results_corrected <- pa_results %>%
    group_by(Model) %>%
    mutate(
      P_BH = p.adjust(P_value, method = "BH"),
      Significance = case_when(
        P_BH < 0.05 ~ "显著 (BH校正后)",
        P_value < 0.05 ~ "原始P值显著",
        TRUE ~ "不显著"
      )
    ) %>% ungroup()
  non_pa_results <- non_pa_results %>%
    mutate(
      P_BH = NA,
      Significance = case_when(
        P_value < 0.05 ~ "原始P值显著",
        TRUE ~ "不显著"
      )
    )
  results_corrected <- bind_rows(pa_results_corrected, non_pa_results) %>%
    mutate(
      CI = paste0(HR, " (", CI_lower, "-", CI_upper, ")"),
      P_value_formatted = case_when(
        P_value < 0.001 ~ "<0.001",
        TRUE ~ sprintf("%.3f", P_value)
      ),
      P_BH_formatted = case_when(
        is.na(P_BH) ~ "NA",
        P_BH < 0.001 ~ "<0.001",
        TRUE ~ sprintf("%.3f", P_BH)
      )
    )
  
  # 按变量分组
  cat("生成按变量分组的结果...\n")
  results_by_var <- results_corrected %>%
    select(Category, Variable, Variable_Type, Model, CI, P_value_formatted, P_BH_formatted, Significance) %>%
    pivot_wider(names_from = Model, values_from = c(CI, P_value_formatted, P_BH_formatted, Significance))
  
  # 按模型分组
  cat("生成按分析模型分组的结果...\n")
  results_by_model <- results_corrected %>%
    arrange(Model, Category, Variable) %>%
    select(Model, Category, Variable, Variable_Type, HR, CI_lower, CI_upper, P_value_formatted, P_BH_formatted, Significance)
  
  # 保存结果
  cat("保存结果文件...\n")
  write.csv(results_corrected, file.path(output_dir, "cox_results_all_corrected.csv"), row.names = FALSE)
  write.csv(results_by_var, file.path(output_dir, "cox_results_by_variable_corrected.csv"), row.names = FALSE)
  write.csv(results_by_model, file.path(output_dir, "cox_results_by_model_corrected.csv"), row.names = FALSE)
  
  cat(paste0("分析完成！结果已保存到目录: ", output_dir, "\n"))
}

# 执行批量处理
input_directory <- "F:/Research_Project/EP_ukb/Revision_sleep/sleep_cox/"
output_directory <- "F:/Research_Project/EP_ukb/Revision_sleep/sleep_coxresults/"
process_csv_files(input_directory, output_directory)