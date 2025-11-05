# 加载必要的包
library(survival)
library(dplyr)
library(broom)
library(knitr)
library(tidyr)
library(stringr)

# 主函数：批量处理CSV文件
process_csv_files <- function(input_dir, output_dir) {
  # 获取目录中的所有CSV文件
  csv_files <- list.files(path = input_dir, pattern = "\\.csv$", full.names = TRUE)
  
  # 如果没有找到文件，给出提示并退出
  if(length(csv_files) == 0) {
    cat("错误：指定目录中没有找到CSV文件\n")
    return()
  }
  
  # 创建主输出目录（如果不存在）
  if(!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
    cat(paste0("创建输出目录: ", output_dir, "\n"))
  }
  
  # 显示找到的文件总数
  cat(paste0("找到 ", length(csv_files), " 个CSV文件需要处理\n"))
  
  # 遍历每个CSV文件并进行分析
  for(i in 1:length(csv_files)) {
    csv_path <- csv_files[i]
    csv_name <- tools::file_path_sans_ext(basename(csv_path))
    
    # 创建该CSV文件的输出目录
    file_output_dir <- file.path(output_dir, csv_name)
    if(!dir.exists(file_output_dir)) {
      dir.create(file_output_dir, recursive = TRUE)
    }
    
    # 输出当前处理信息
    cat(paste0("\n[", i, "/", length(csv_files), "] 处理文件: ", csv_name, "\n"))
    
    # 处理当前CSV文件
    process_single_csv(csv_path, file_output_dir)
  }
  
  cat("\n全部分析完成！每个文件的结果已保存到相应的子目录中\n")
}

# 用于处理单个CSV文件的函数
process_single_csv <- function(file_path, output_dir) {
  # 读取数据
  cat("读取数据...\n")
  data <- read.csv(file_path)
  
  # 定义协变量组
  covariates_demo <- c("gender", "education_level", "smoking", "alcohol_use", 
                       "ethic", "BMI", "Townsend_index", "enrollment_age")
  covariates_health <- c("Disable_long", "charlson_score", "overall_health")
  covariates_pa <- c("Light_Total_Hour", "MVPA_Total_Hour", "Sedentary_Total_Hour")
  
  # 合并所有协变量
  all_vars <- c(covariates_demo, covariates_health, covariates_pa)
  
  # 检查数据中实际存在的变量
  existing_vars <- all_vars[all_vars %in% colnames(data)]
  existing_demo <- covariates_demo[covariates_demo %in% colnames(data)]
  existing_health <- covariates_health[covariates_health %in% colnames(data)]
  existing_pa <- covariates_pa[covariates_pa %in% colnames(data)]
  
  if(length(existing_vars) < length(all_vars)) {
    missing_vars <- all_vars[!all_vars %in% colnames(data)]
    cat(paste0("警告: 以下变量在数据中不存在: ", paste(missing_vars, collapse=", "), "\n"))
    all_vars <- existing_vars
  }
  
  # 函数：执行Cox模型并提取结果
  run_cox_model <- function(data, var_name, adjust_vars = NULL, time_var = "Followingyears_birthday", status_var = "is_dead") {
    # 确保时间和状态变量存在
    if(!all(c(time_var, status_var) %in% colnames(data))) {
      cat(paste0("错误: 生存时间变量('", time_var, "')或事件状态变量('", status_var, "')不存在\n"))
      return(NULL)
    }
    
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
    tryCatch({
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
    }, error = function(e) {
      cat(paste0("错误: 变量 '", var_name, "' 的Cox模型计算失败: ", e$message, "\n"))
      return(NULL)
    })
  }
  
  # 初始化结果数据框
  results <- data.frame(Variable = character(),
                        Model = character(),
                        HR = numeric(),
                        CI_lower = numeric(),
                        CI_upper = numeric(),
                        P_value = numeric(),
                        stringsAsFactors = FALSE)
  
  # 执行单变量分析（模型1：无调整）
  cat("执行单变量Cox分析（模型1）...\n")
  for (var in all_vars) {
    # 跳过有太多缺失值的变量
    if (sum(is.na(data[[var]])) / nrow(data) > 0.5) {
      cat(paste0("跳过变量 '", var, "' (>50% 缺失值)\n"))
      next
    }
    
    var_result <- run_cox_model(data, var)
    if(is.null(var_result)) next
    
    # 添加到结果数据框
    for (i in 1:nrow(var_result)) {
      results <- rbind(results, data.frame(
        Variable = ifelse(nrow(var_result) > 1, var_result$term[i], var),
        Model = "模型1: 单变量",
        HR = round(var_result$estimate[i], 2),
        CI_lower = round(var_result$conf.low[i], 2),
        CI_upper = round(var_result$conf.high[i], 2),
        P_value = var_result$p.value[i],
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # 执行人口学调整分析（模型2：调整人口学因素）
  cat("执行人口学调整Cox分析（模型2）...\n")
  for (var in all_vars) {
    # 跳过有太多缺失值的变量
    if (sum(is.na(data[[var]])) / nrow(data) > 0.5) {
      cat(paste0("跳过变量 '", var, "' (>50% 缺失值)\n"))
      next
    }
    
    # 对于每个变量，调整人口学变量
    var_result <- run_cox_model(data, var, existing_demo)
    if(is.null(var_result)) next
    
    # 添加到结果数据框
    for (i in 1:nrow(var_result)) {
      results <- rbind(results, data.frame(
        Variable = ifelse(nrow(var_result) > 1, var_result$term[i], var),
        Model = "模型2: 人口学调整",
        HR = round(var_result$estimate[i], 2),
        CI_lower = round(var_result$conf.low[i], 2),
        CI_upper = round(var_result$conf.high[i], 2),
        P_value = var_result$p.value[i],
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # 执行全变量调整分析（模型3：调整所有因素）
  cat("执行全变量调整Cox分析（模型3）...\n")
  for (var in all_vars) {
    # 跳过有太多缺失值的变量
    if (sum(is.na(data[[var]])) / nrow(data) > 0.5) {
      cat(paste0("跳过变量 '", var, "' (>50% 缺失值)\n"))
      next
    }
    
    # 对于每个变量，调整所有其他变量
    var_result <- run_cox_model(data, var, all_vars)
    if(is.null(var_result)) next
    
    # 添加到结果数据框
    for (i in 1:nrow(var_result)) {
      results <- rbind(results, data.frame(
        Variable = ifelse(nrow(var_result) > 1, var_result$term[i], var),
        Model = "模型3: 全变量调整",
        HR = round(var_result$estimate[i], 2),
        CI_lower = round(var_result$conf.low[i], 2),
        CI_upper = round(var_result$conf.high[i], 2),
        P_value = var_result$p.value[i],
        stringsAsFactors = FALSE
      ))
    }
  }
  
  # 仅对PA变量进行Benjamini-Hochberg校正
  cat("对身体活动变量执行FDR校正...\n")
  
  # 将结果分为PA变量和非PA变量
  existing_pa_vars <- covariates_pa[covariates_pa %in% all_vars]
  pa_results <- results %>%
    dplyr::filter(Variable %in% existing_pa_vars | 
                    grepl(paste(paste0("^", existing_pa_vars), collapse="|"), Variable))
  
  non_pa_results <- results %>%
    dplyr::filter(!(Variable %in% existing_pa_vars | 
                      grepl(paste(paste0("^", existing_pa_vars), collapse="|"), Variable)))
  
  # 只对PA变量进行Benjamini-Hochberg校正（FDR），按模型分组进行校正
  pa_results_corrected <- pa_results %>%
    dplyr::group_by(Model) %>%
    dplyr::mutate(
      # Benjamini-Hochberg校正（FDR）
      P_BH = p.adjust(P_value, method = "BH"),
      
      # 标记显著性
      Significance = dplyr::case_when(
        P_BH < 0.05 ~ "显著 (BH校正后)",
        P_value < 0.05 ~ "原始P值显著",
        TRUE ~ "不显著"
      )
    ) %>%
    dplyr::ungroup()
  
  # 为非PA变量保留原始P值，不进行校正
  non_pa_results <- non_pa_results %>%
    dplyr::mutate(
      P_BH = NA,
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
      P_BH_formatted = dplyr::case_when(
        is.na(P_BH) ~ "NA",
        P_BH < 0.001 ~ "<0.001",
        TRUE ~ sprintf("%.3f", P_BH)
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
  
  # 按不同方式整理结果
  # 1. 按变量分组
  cat("生成按变量分组的结果...\n")
  results_by_var <- results_corrected %>%
    dplyr::select(Category, Variable, Model, CI, P_value_formatted, P_BH_formatted, Significance) %>%
    tidyr::pivot_wider(names_from = Model, values_from = c(CI, P_value_formatted, P_BH_formatted, Significance))
  
  # 2. 按分析模型分组
  cat("生成按分析模型分组的结果...\n")
  results_by_model <- results_corrected %>%
    dplyr::arrange(Model, Category, Variable) %>%
    dplyr::select(Model, Category, Variable, HR, CI_lower, CI_upper, P_value_formatted, P_BH_formatted, Significance)
  
  # 保存结果
  cat("保存结果文件...\n")
  write.csv(results_corrected, file.path(output_dir, "cox_results_all_corrected.csv"), row.names = FALSE)
  write.csv(results_by_var, file.path(output_dir, "cox_results_by_variable_corrected.csv"), row.names = FALSE)
  write.csv(results_by_model, file.path(output_dir, "cox_results_by_model_corrected.csv"), row.names = FALSE)
  
  # 为身体活动变量创建详细的结果表（只含FDR校正）
  pa_table_detailed <- pa_results_corrected %>%
    dplyr::arrange(Model, Variable) %>%
    dplyr::mutate(
      CI = paste0(HR, " [", CI_lower, ", ", CI_upper, "]"),
      Significant_Original = ifelse(P_value < 0.05, "是", "否"),
      Significant_BH = ifelse(P_BH < 0.05, "是", "否"),
      # 增加边缘显著性标记
      Marginal_Significant = dplyr::case_when(
        P_value < 0.05 ~ "显著 (p<0.05)",
        P_value < 0.10 ~ "边缘显著 (p<0.10)",
        TRUE ~ "不显著"
      )
    ) %>%
    dplyr::select(
      Variable, Model, HR, CI, 
      P_value, P_BH,
      Significant_Original, Significant_BH,
      Marginal_Significant
    )
  
  # 保存PA变量详细结果
  write.csv(pa_table_detailed, file.path(output_dir, "physical_activity_cox_results_detailed.csv"), row.names = FALSE)
  
  # 创建一个简洁版表格用于发表
  pa_table_publication <- pa_table_detailed %>%
    dplyr::select(Variable, Model, HR, CI, P_value, P_BH) %>%
    dplyr::mutate(
      P_value = ifelse(P_value < 0.001, "<0.001", sprintf("%.3f", P_value)),
      P_BH = ifelse(P_BH < 0.001, "<0.001", sprintf("%.3f", P_BH))
    )
  
  # 保存简洁版表格
  write.csv(pa_table_publication, file.path(output_dir, "physical_activity_cox_results_publication.csv"), row.names = FALSE)
  
  # 保存边缘显著性分析结果
  marginal_table <- pa_table_detailed %>%
    dplyr::select(Variable, Model, HR, CI, P_value, P_BH, Marginal_Significant) %>%
    dplyr::mutate(
      P_value = ifelse(P_value < 0.001, "<0.001", sprintf("%.3f", P_value)),
      P_BH = ifelse(P_BH < 0.001, "<0.001", sprintf("%.3f", P_BH))
    )
  
  write.csv(marginal_table, file.path(output_dir, "physical_activity_marginal_significance.csv"), row.names = FALSE)
  
  cat(paste0("分析完成！结果已保存到目录: ", output_dir, "\n"))
}

# 执行批量处理
# 请设置输入和输出目录路径
input_directory <- "/input_files/"
output_directory <- "/results"

# 执行分析
process_csv_files(input_directory, output_directory)
