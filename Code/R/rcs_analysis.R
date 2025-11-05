# 加载必要的包
library(rcssci)
library(dplyr)
library(survival)
library(tools)

# 批量处理文件夹下所有CSV文件的函数
process_rcs_for_all_files <- function(input_dir, output_base_dir) {
  csv_files <- list.files(path = input_dir, pattern = "\\.csv$", full.names = TRUE)
  if(length(csv_files) == 0) {
    cat("错误：指定目录中没有找到CSV文件\n")
    return()
  }
  cat(paste0("找到 ", length(csv_files), " 个CSV文件需要处理\n"))
  for(i in seq_along(csv_files)) {
    csv_path <- csv_files[i]
    csv_name <- file_path_sans_ext(basename(csv_path))
    file_output_dir <- file.path(output_base_dir, csv_name)
    if(!dir.exists(file_output_dir)) {
      dir.create(file_output_dir, recursive = TRUE)
      cat(paste0("创建输出目录: ", file_output_dir, "\n"))
    }
    cat(paste0("\n[", i, "/", length(csv_files), "] 处理文件: ", csv_name, "\n"))
    process_single_file_rcs(csv_path, file_output_dir)
  }
  cat("\n全部分析完成！每个文件的RCS结果已保存到相应的子目录中\n")
}

# 处理单个CSV文件的RCS分析
process_single_file_rcs <- function(file_path, output_dir) {
  cat("读取数据...\n")
  data <- read.csv(file_path)
  
  # 定义协变量组
  covariates_demo <- c("gender", "education_level", "smoking", "alcohol_use", 
                       "ethic", "BMI", "Townsend_index", "enrollment_age", "sleep_condition")
  covariates_health <- c("Disable_long", "elixhauser_score", "overall_health")
  covariates_pa <- c("Sedentary_Total_Hour")
  
  # 检查并提取实际存在的变量
  existing_demo <- covariates_demo[covariates_demo %in% colnames(data)]
  existing_health <- covariates_health[covariates_health %in% colnames(data)]
  existing_pa <- covariates_pa[covariates_pa %in% colnames(data)]
  covariates_all <- c(existing_demo, existing_health, existing_pa)
  exposure_vars <- existing_pa
  
  # 检查关键生存变量是否存在
  if(!all(c("Followingyears_birthday", "is_dead") %in% colnames(data))) {
    cat("错误：生存时间变量('Followingyears_birthday')或事件变量('is_dead')不存在\n")
    return()
  }
  
  for(exposure in exposure_vars) {
    cat(paste0("处理变量: ", exposure, "\n"))
    subdir <- file.path(output_dir, exposure)
    if(!dir.exists(subdir)) dir.create(subdir, recursive = TRUE)
    pa_covs <- setdiff(existing_pa, exposure)
    tryCatch({
      # 模型1：仅调整其它PA变量
      cat("  执行PA调整模型...\n")
      rcssci_cox(
        data = data, 
        time = "Followingyears_birthday", 
        y = "is_dead", 
        x = exposure, 
        cov = pa_covs, 
        prob = 0.1, 
        knots = 3,
        filepath = file.path(subdir, paste0(exposure, "_PA_adjusted"))
      )
      # 模型2：调整所有协变量
      cat("  执行全变量调整模型...\n")
      all_covs <- setdiff(covariates_all, exposure)
      rcssci_cox(
        data = data, 
        time = "Followingyears_birthday", 
        y = "is_dead", 
        x = exposure, 
        cov = all_covs, 
        prob = 0.1, 
        knots = 3,
        filepath = file.path(subdir, paste0(exposure, "_fully_adjusted"))
      )
    }, error = function(e) {
      cat(paste0("  错误: 变量 '", exposure, "' 的RCS分析失败: ", e$message, "\n"))
    })
  }
  cat(paste0("文件 '", basename(file_path), "' 的RCS分析完成\n"))
}

# 执行批量处理
input_directory <- "E:/phd_project/AF_PA_Mortality/RCS/For_rcs"
output_directory <- "E:/phd_project/AF_PA_Mortality/RCS/results"
if(!dir.exists(output_directory)) dir.create(output_directory, recursive = TRUE)
process_rcs_for_all_files(input_directory, output_directory)