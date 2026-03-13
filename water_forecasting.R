# ============================================================
# 城市用水量预测：经典统计学 (SARIMA) vs 工业级模型 (Prophet)
# ============================================================

if (!require("pacman")) install.packages("pacman")
pacman::p_load(tidyverse, forecast, prophet, lubridate, Metrics, ggplot2, tseries, readxl)
# 1. 数据读取与防冲突处理
print(">>> 1. 数据加载与预处理...")
df_raw <- read_xlsx("Water consumption.xlsx")
colnames(df_raw)[1:2] <- c("YearCol", "MonthCol")

df <- df_raw %>%
  mutate(
    date_str = paste(YearCol, MonthCol, "01", sep = "-"),
    ds = ymd(date_str),
    y = as.numeric(`totalWaterConsumption (m3)`)
  ) %>%
  filter(!is.na(ds) & !is.na(y)) %>%
  arrange(ds) %>% select(ds, y)

# 2. 样本外切分 (Out-of-sample Split)
test_len <- 24 
train_df <- head(df, nrow(df) - test_len)
test_df  <- tail(df, test_len)

start_yr <- as.numeric(format(min(train_df$ds), "%Y"))
start_mo <- as.numeric(format(min(train_df$ds), "%m"))
train_ts <- ts(train_df$y, start = c(start_yr, start_mo), frequency = 12)

# ============================================================
#  建模前传：严谨的统计学假设检验 (EDA & Testing)
# ============================================================
print(">>> 2. 正在进行时间序列平稳性检验 (ADF Test)...")

# ADF 检验 (Augmented Dickey-Fuller Test)
# 原假设 (H0): 时间序列存在单位根，是非平稳的。
adf_result <- adf.test(train_ts, alternative = "stationary")
print(adf_result)

if(adf_result$p.value > 0.05) {
  cat("⚠️ 结论: P值 > 0.05，无法拒绝原假设。数据是【非平稳】的！\n")
  cat("💡 动作: 必须在 SARIMA 中使用差分 (d > 0) 来消除趋势。\n\n")
} else {
  cat("✅ 结论: P值 <= 0.05，拒绝原假设。数据是平稳的。\n\n")
}

# ============================================================
# 4. 算法博弈：SARIMA vs Prophet
# ============================================================
print(">>> 3. 正在训练经典统计学模型: SARIMA...")
# auto.arima 会参考前一步的结论，自动执行所需阶数的差分
fit_sarima <- auto.arima(train_ts, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
forecast_sarima <- forecast(fit_sarima, h = test_len)

print(">>> 4. 正在训练现代工业级模型: Prophet...")
fit_prophet <- prophet(train_df, yearly.seasonality = TRUE, weekly.seasonality = FALSE, daily.seasonality = FALSE)
future_prophet <- make_future_dataframe(fit_prophet, periods = test_len, freq = "month")
res_prophet <- tail(predict(fit_prophet, future_prophet)$yhat, test_len)

# ============================================================
# 建模后传：残差诊断 (Residual Diagnostics)
# ============================================================
print(">>> 5. 正在进行 SARIMA 模型残差白噪声检验 (Ljung-Box Test)...")

# Ljung-Box 检验
# 原假设 (H0): 残差序列是纯随机的白噪声 (即模型已经提取了所有有用信息)。
lb_test <- Box.test(fit_sarima$residuals, type = "Ljung-Box")
print(lb_test)

if(lb_test$p.value > 0.05) {
  cat("✅ 结论: P值 > 0.05，无法拒绝原假设。残差是【白噪声】！\n")
  cat("💡 动作: 模型极度健康，没有遗漏任何时间序列规律，可以直接用于预测。\n\n")
} else {
  cat("⚠️ 结论: P值 <= 0.05，残差仍有自相关性，模型可能需要优化。\n\n")
}

# 6. 精度对决 (样本外 MAPE)
mape_sarima <- mape(test_df$y, as.numeric(forecast_sarima$mean)) * 100
mape_prophet <- mape(test_df$y, res_prophet) * 100

cat("====================================\n")
cat("🎯 样本外盲测 24个月误差对决：\n")
cat(sprintf("🏆 [SARIMA]  MAPE: %.2f%%\n", mape_sarima))
cat(sprintf("🏆 [Prophet] MAPE: %.2f%%\n", mape_prophet))
cat("====================================\n")


# ------------------------------------------------------------
#  工业级可视化大屏
# ------------------------------------------------------------
print(">>> 6. 正在生成高对比度业务图表...")

# 构建用于画图的数据框
plot_data <- bind_rows(
  train_df %>% mutate(Type = "Historical Data"),
  test_df %>% mutate(Type = "Actual Truth"),
  data.frame(ds = test_df$ds, y = as.numeric(forecast_sarima$mean), Type = "SARIMA Forecast"),
  data.frame(ds = test_df$ds, y = res_prophet, Type = "Prophet Forecast")
)

# 使用 ggplot2 绘制极其专业的对比图
p <- ggplot() +
  # 历史数据 (灰色)
  geom_line(data = filter(plot_data, Type == "Historical Data"), aes(x = ds, y = y), color = "gray70", size = 0.8) +
  # 真实测试集数据 (黑色实体)
  geom_line(data = filter(plot_data, Type == "Actual Truth"), aes(x = ds, y = y), color = "black", size = 1.2) +
  # SARIMA 预测 (蓝色虚线)
  geom_line(data = filter(plot_data, Type == "SARIMA Forecast"), aes(x = ds, y = y), color = "blue", linetype = "dashed", size = 1) +
  # Prophet 预测 (红色点划线)
  geom_line(data = filter(plot_data, Type == "Prophet Forecast"), aes(x = ds, y = y), color = "red", linetype = "dotdash", size = 1) +
  labs(
    title = "澳门城市用水量预测：SARIMA 与 Prophet 样本外盲测对比",
    subtitle = sprintf("冠军模型: %s", ifelse(mape_sarima < mape_prophet, "SARIMA", "Prophet")),
    x = "年份",
    y = "总用水量 (m3)"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(size = 16, face = "bold"),
    plot.subtitle = element_text(size = 12, color = "darkgray"),
    axis.title = element_text(size = 12, face = "bold")
  )

print(p)

# ------------------------------------------------------------
# 7. 终极业务产出：未来 24 个月的进阶调度表 (全量数据训练)
# ------------------------------------------------------------
print(">>> 6. 正在生成未来 24 个月的最终业务排期表 (使用冠军模型 SARIMA)...")

# 构建全量数据的 ts 对象（从第一天到最后一天）
start_yr_full <- as.numeric(format(min(df$ds), "%Y"))
start_mo_full <- as.numeric(format(min(df$ds), "%m"))
full_ts <- ts(df$y, start = c(start_yr_full, start_mo_full), frequency = 12)

# 使用 auto.arima 对全量数据进行终极拟合
final_sarima <- auto.arima(full_ts, seasonal = TRUE, stepwise = FALSE, approximation = FALSE)
final_forecast <- forecast(final_sarima, h = 24)

# 生成未来 24 个月的日期序列
last_date <- max(df$ds)
future_dates <- seq(last_date %m+% months(1), by = "month", length.out = 24)

# 提取 SARIMA 预测的中值、以及 95% 置信区间的下界和上界
# forecast 函数的 lower/upper 默认第一列是 80%，第二列是 95%
output_plan <- data.frame(
  Date = future_dates,
  Forecast_m3 = round(as.numeric(final_forecast$mean), 0),
  Lower_95_CI = round(as.numeric(final_forecast$lower[, 2]), 0),
  Upper_95_CI = round(as.numeric(final_forecast$upper[, 2]), 0)
)

# 导出 CSV 供供应链/水务部门使用
write.csv(output_plan, "Macau_Water_24Months_Plan.csv", row.names = FALSE)
print("✅ 完美收工！由 SARIMA 驱动的结果报表已保存为 'Macau_Water_24Months_Plan.csv'。")