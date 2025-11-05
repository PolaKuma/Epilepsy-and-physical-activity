library(comorbidity)
library(data.table)

# 读取数据
ukb_data <- fread("/41270_icd10.csv")
setnames(ukb_data, c("eid", "icd10_codes"))

# 高效的ICD-10提取函数
extract_icd10_fast <- function(dt) {
  # 过滤掉空值或无效格式
  dt <- dt[!is.na(icd10_codes) & 
             icd10_codes != "" & 
             grepl("\\[.*\\]", icd10_codes)]
  
  if(nrow(dt) == 0) return(data.table(id = integer(), code = character()))
  
  # 使用正则表达式直接提取代码部分
  result <- dt[, {
    # 提取所有ICD-10代码
    matches <- gregexpr("'([A-Z][0-9]{1,2}(\\.\\d{1,2})?)[^']*'", icd10_codes)
    all_matches <- regmatches(icd10_codes, matches)[[1]]
    
    # 只保留代码部分
    codes <- gsub("^'([A-Z][0-9]{1,2}(\\.\\d{1,2})?).*'$", "\\1", all_matches)
    
    # 返回ID和代码的数据表
    .(id = rep(eid, length(codes)), code = codes)
  }, by = seq_len(nrow(dt))]
  
  # 移除序列号列
  result[, seq_len := NULL]
  return(result)
}

# 分批处理数据
process_batches <- function(dt, batch_size = 50000) {
  total_rows <- nrow(dt)
  num_batches <- ceiling(total_rows / batch_size)
  all_results <- data.table(id = integer(), code = character())
  
  for(i in 1:num_batches) {
    start_idx <- (i-1) * batch_size + 1
    end_idx <- min(i * batch_size, total_rows)
    
    cat(sprintf("处理批次 %d/%d (行 %d 到 %d)...\n", 
                i, num_batches, start_idx, end_idx))
    
    batch_result <- extract_icd10_fast(dt[start_idx:end_idx])
    all_results <- rbindlist(list(all_results, batch_result))
  }
  
  return(all_results)
}

# 开始提取ICD-10代码
cat("开始提取ICD-10代码...\n")
system.time({
  icd10_expanded <- process_batches(ukb_data, batch_size = 50000)
})

# 显示提取结果
cat("提取的ICD-10代码数量:", nrow(icd10_expanded), "\n")

# 创建完整的患者ID列表
all_patients <- data.table(id = unique(ukb_data$eid))
cat("总患者数量:", nrow(all_patients), "\n")

# 考虑采用分块计算CCI和EI以降低内存使用
chunk_compute_indices <- function(codes_dt, all_ids, chunk_size = 250000) {
  total_patients <- length(all_ids)
  cat("总共有", total_patients, "名患者需要处理\n")
  
  # 分组计算
  num_chunks <- ceiling(length(all_ids) / chunk_size)
  
  # 初始化结果数据框
  cci_combined <- data.table()
  ei_combined <- data.table()
  
  for(i in 1:num_chunks) {
    start_idx <- (i-1) * chunk_size + 1
    end_idx <- min(i * chunk_size, length(all_ids))
    chunk_ids <- all_ids[start_idx:end_idx]
    
    cat(sprintf("处理第 %d/%d 组患者 (ID %d 到 %d)...\n", 
                i, num_chunks, start_idx, end_idx))
    
    # 提取当前组的代码
    chunk_codes <- codes_dt[id %in% chunk_ids]
    
    # 处理没有代码的情况
    if(nrow(chunk_codes) == 0) {
      cat("  - 该组患者没有ICD-10代码，分配默认值0\n")
      cci_dt <- data.table(id = chunk_ids)
      cci_dt[, `:=`(
        charlson_score = 0,
        charlson_weighted = 0
      )]
      
      ei_dt <- data.table(id = chunk_ids)
      ei_dt[, `:=`(
        elixhauser_score = 0,
        elixhauser_weighted = 0
      )]
    } else {
      # 计算Charlson指数
      cat("  - 计算Charlson共病指数...\n")
      cci_scores <- comorbidity(
        x = as.data.frame(chunk_codes),
        id = "id",
        code = "code",
        map = "charlson_icd10_quan",
        assign0 = TRUE
      )
      
      # 添加分数
      cci_dt <- as.data.table(cci_scores)
      cci_dt[, `:=`(
        charlson_score = score(cci_scores, weights = NULL, assign0 = TRUE),
        charlson_weighted = score(cci_scores, weights = "charlson", assign0 = TRUE)
      )]
      
      # 只保留ID和分数列
      cci_dt <- cci_dt[, .(id, charlson_score, charlson_weighted)]
      
      # 确保该组所有ID都在结果中
      missing_ids <- setdiff(chunk_ids, cci_dt$id)
      if(length(missing_ids) > 0) {
        missing_dt <- data.table(
          id = missing_ids,
          charlson_score = 0,
          charlson_weighted = 0
        )
        cci_dt <- rbindlist(list(cci_dt, missing_dt))
      }
      
      # 计算Elixhauser指数
      cat("  - 计算Elixhauser指数...\n")
      ei_scores <- comorbidity(
        x = as.data.frame(chunk_codes),
        id = "id",
        code = "code",
        map = "elixhauser_icd10_quan",
        assign0 = TRUE
      )
      
      # 添加分数
      ei_dt <- as.data.table(ei_scores)
      ei_dt[, `:=`(
        elixhauser_score = score(ei_scores, weights = NULL, assign0 = TRUE),
        elixhauser_weighted = score(ei_scores, weights = "vw", assign0 = TRUE)
      )]
      
      # 只保留ID和分数列
      ei_dt <- ei_dt[, .(id, elixhauser_score, elixhauser_weighted)]
      
      # 确保该组所有ID都在结果中
      missing_ids <- setdiff(chunk_ids, ei_dt$id)
      if(length(missing_ids) > 0) {
        missing_dt <- data.table(
          id = missing_ids,
          elixhauser_score = 0,
          elixhauser_weighted = 0
        )
        ei_dt <- rbindlist(list(ei_dt, missing_dt))
      }
    }
    
    # 合并结果
    cci_combined <- rbindlist(list(cci_combined, cci_dt))
    ei_combined <- rbindlist(list(ei_combined, ei_dt))
    
    # 释放内存
    if(exists("chunk_codes")) rm(chunk_codes)
    if(exists("cci_scores")) rm(cci_scores)
    if(exists("ei_scores")) rm(ei_scores)
    gc()
  }
  
  return(list(cci = cci_combined, ei = ei_combined))
}

# 获取所有唯一患者ID
all_unique_ids <- unique(ukb_data$eid)

# 计算共病指数
cat("开始分块计算共病指数...\n")
system.time({
  indices <- chunk_compute_indices(icd10_expanded, all_unique_ids, chunk_size = 100000)
})

# 合并结果
result <- indices$cci[
  indices$ei, 
  on = "id"
]

# 检查结果数量
cat("结果中的患者数量:", nrow(result), "\n")
cat("应有的患者数量:", length(all_unique_ids), "\n")

# 确保所有患者都包含在结果中
if(nrow(result) != length(all_unique_ids)) {
  warning("结果患者数量与总患者数量不一致")
  # 查找缺失的患者
  missing_ids <- setdiff(all_unique_ids, result$id)
  cat("缺失患者数量:", length(missing_ids), "\n")
  
  # 为缺失患者添加记录
  if(length(missing_ids) > 0) {
    missing_dt <- data.table(
      id = missing_ids,
      charlson_score = 0,
      charlson_weighted = 0,
      elixhauser_score = 0,
      elixhauser_weighted = 0
    )
    result <- rbindlist(list(result, missing_dt))
  }
}

# 重命名id列为eid
setnames(result, "id", "eid")

# 查看前几行结果
print(head(result))

# 保存结果
fwrite(result, "/ukb_comorbidity_scores.csv")

# 基本统计信息
cat("\nCCI统计:\n")
print(summary(result$charlson_score))
cat("CCI得分为0的患者比例:", sum(result$charlson_score == 0) / nrow(result), "\n")

cat("\nCCI加权统计:\n")
print(summary(result$charlson_weighted))
cat("CCI加权得分为0的患者比例:", sum(result$charlson_weighted == 0) / nrow(result), "\n")

cat("\nEI统计:\n")
print(summary(result$elixhauser_score))
cat("EI得分为0的患者比例:", sum(result$elixhauser_score == 0) / nrow(result), "\n")

cat("\nEI加权统计:\n")
print(summary(result$elixhauser_weighted))
cat("EI加权得分为0的患者比例:", sum(result$elixhauser_weighted == 0) / nrow(result), "\n")

