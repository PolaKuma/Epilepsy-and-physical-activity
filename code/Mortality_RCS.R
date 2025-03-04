# 加载必要的包
library(rcssci)
library(dplyr)
library(survival)

# 读取数据
data <- read.csv("death_file.csv")

# 处理缺失值
# 对连续变量使用均值填充
continuous_vars <- c("Light_Total_Min", "MVPA_Total_Min", "Sedentary_Total_Min", "BMI", "Townsend_index", "age_ep_min", "enrollment_age")
for(var in continuous_vars) {
  data[[var]][is.na(data[[var]])] <- mean(data[[var]], na.rm = TRUE)
}


# 定义协变量组
covariates_demo <- c("gender", "education_level", "smoking", "alcohol_use", "ethic", "sleep_condition", "BMI", "Townsend_index", "enrollment_age")
covariates_health <- c("Disable_long", "cancer_report", "overall_health", "hpb_combined", "cvd_combined", "bd_combined", "zcd_combined", "diabetes_combined", "age_ep_min")
covariates_all <- c(covariates_demo, covariates_health)

# 定义暴露变量
exposure_vars <- c("Light_Total_Min", "MVPA_Total_Min", "Sedentary_Total_Min")

# 进行RCS分析
for(exposure in exposure_vars) {
  
  # 模型4：调整所有协变量
  rcssci_cox(data = data, time = "Followingyears_birthday_EP", y = "cancer_death", x = exposure, 
             adjust = covariates_all, prob = 0.1, knots = 3, ref = "median",
             filepath = paste0("E:/EP_ukb/Results/Figures/Figure4/", exposure, "_cancer_death_finalmodel"))
}

