library(readxl)
library(dplyr)
library(caret)
library(ggplot2)
library(knitr)

set.seed(123)

#=============================================================
# 4.1 PENGUMPULAN DATA
#=============================================================
data_raw <- read_excel(
  "C:/Users/grace/Downloads/data gabungan grace.xlsx"
)

cat("Jumlah Observasi :", nrow(data_raw), "\n")
cat("Jumlah Variabel  :", ncol(data_raw), "\n")

#=============================================================
# 4.2 PEMBENTUKAN VARIABEL DEPENDEN
#=============================================================
result_label <- data.frame(
  Model = c("Springate","Grover","Zmijewski","Label"),
  
  Class_0 = c(
    sum(data_raw$`Label Springate` == 0, na.rm=TRUE),
    sum(data_raw$`Label Grover` == 0, na.rm=TRUE),
    sum(data_raw$`Label Zmijewski` == 0, na.rm=TRUE),
    sum(data_raw$Label == 0, na.rm=TRUE)
  ),
  
  Class_1 = c(
    sum(data_raw$`Label Springate` == 1, na.rm=TRUE),
    sum(data_raw$`Label Grover` == 1, na.rm=TRUE),
    sum(data_raw$`Label Zmijewski` == 1, na.rm=TRUE),
    sum(data_raw$Label == 1, na.rm=TRUE)
  ),
  
  Missing = c(
    sum(is.na(data_raw$`Label Springate`)),
    sum(is.na(data_raw$`Label Grover`)),
    sum(is.na(data_raw$`Label Zmijewski`)),
    sum(is.na(data_raw$Label))
  )
)

kable(
  result_label,
  caption = "Hasil Pembentukan Variabel Dependen"
)

#=============================================================
# PILIH VARIABEL PENELITIAN
#=============================================================
data_model <- data_raw %>%
  select(
    ROA, ROE, EBITTA, DER, DAR, ICR, CR,
    `TA Turnover`, `WC/TA`, OCF,
    WC, TA, NBT, Sales, `Net Income`,
    TD, CA, CL, EBIT, RETA,
    Label
  )

names(data_model) <- make.names(names(data_model))

#=============================================================
# DATA CLEANING
#=============================================================
data_model[data_model == ""]   <- NA
data_model[data_model == "NA"] <- NA

data_model <- data_model %>%
  mutate(
    across(
      -Label,
      ~as.numeric(gsub(",", ".", as.character(.)))
    )
  )

data_model <- data_model %>%
  filter(!is.na(Label))

data_model$Label <- factor(
  data_model$Label,
  levels = c(0,1)
)

num_cols <- setdiff(names(data_model), "Label")

#=============================================================
# 4.3 PENANGANAN MISSING VALUE
#=============================================================
missing_before <- data.frame(
  Variabel = num_cols,
  Missing_Before = colSums(is.na(data_model[, num_cols]))
)

kable(
  missing_before,
  caption = "Jumlah Missing Value Sebelum Imputasi"
)

#=============================================================
# 4.4 PEMBAGIAN DATA TRAINING DAN TESTING
#=============================================================
train_index <- createDataPartition(
  data_model$Label,
  p = 0.8,
  list = FALSE
)

train_data <- data_model[train_index, ]
test_data  <- data_model[-train_index, ]

dist_split <- data.frame(
  Dataset = c("Training","Training","Testing","Testing"),
  Kelas   = c("0","1","0","1"),
  Jumlah  = c(
    sum(train_data$Label=="0"),
    sum(train_data$Label=="1"),
    sum(test_data$Label=="0"),
    sum(test_data$Label=="1")
  )
)

kable(
  dist_split,
  caption = "Distribusi Data Training dan Testing"
)

#=============================================================
# IMPUTASI MEDIAN (TRAIN ONLY)
#=============================================================
median_train <- sapply(
  train_data[, num_cols],
  function(x){
    if(all(is.na(x))) 0 else median(x, na.rm=TRUE)
  }
)

for(v in num_cols){
  
  train_data[[v]][is.na(train_data[[v]])] <- median_train[v]
  test_data[[v]][is.na(test_data[[v]])]   <- median_train[v]
  
}

missing_after <- data.frame(
  Variabel = num_cols,
  Missing_Train = colSums(is.na(train_data[, num_cols])),
  Missing_Test  = colSums(is.na(test_data[, num_cols]))
)

kable(
  missing_after,
  caption = "Jumlah Missing Value Setelah Imputasi"
)

#=============================================================
# STATISTIK DESKRIPTIF (SEBELUM STANDARDISASI)
#=============================================================
stat_desc <- data.frame(
  Variabel = num_cols,
  Minimum  = round(sapply(train_data[, num_cols], min),4),
  Median   = round(sapply(train_data[, num_cols], median),4),
  Mean     = round(sapply(train_data[, num_cols], mean),4),
  Maksimum = round(sapply(train_data[, num_cols], max),4)
)

kable(
  stat_desc,
  caption = "Statistik Deskriptif Variabel Independen Sebelum Standarisasi"
)

#=============================================================
# 4.5 PENGUJIAN DATA
# UJI BEDA MEAN (TRAINING DATA)
#=============================================================
result_ttest <- data.frame()

for(v in num_cols){
  
  x0 <- train_data[[v]][train_data$Label=="0"]
  x1 <- train_data[[v]][train_data$Label=="1"]
  
  test <- tryCatch(
    t.test(x0, x1),
    error = function(e) list(
      statistic = NA,
      p.value   = NA
    )
  )
  
  result_ttest <- rbind(
    result_ttest,
    data.frame(
      Variabel    = v,
      Mean_Label0 = round(mean(x0, na.rm=TRUE),4),
      Mean_Label1 = round(mean(x1, na.rm=TRUE),4),
      T_Statistic = round(as.numeric(test$statistic),4),
      P_Value     = round(test$p.value,6),
      Signifikan  = ifelse(
        !is.na(test$p.value) & test$p.value < 0.05,
        "Ya","Tidak"
      )
    )
  )
}

kable(
  result_ttest,
  caption = "Hasil Uji Beda Mean Antar Kelas"
)

#=============================================================
# 4.6 STANDARISASI DATA
#=============================================================
mean_train <- sapply(train_data[, num_cols], mean)
sd_train   <- sapply(train_data[, num_cols], sd)

sd_train[sd_train == 0] <- 1

train_data[, num_cols] <- scale(
  train_data[, num_cols],
  center = mean_train,
  scale  = sd_train
)

test_data[, num_cols] <- scale(
  test_data[, num_cols],
  center = mean_train,
  scale  = sd_train
)

train_data <- as.data.frame(train_data)
test_data  <- as.data.frame(test_data)

train_data$Label <- factor(train_data$Label, levels=c(0,1))
test_data$Label  <- factor(test_data$Label, levels=c(0,1))

stat_std <- data.frame(
  Variabel = num_cols,
  Minimum  = round(sapply(train_data[, num_cols], min),4),
  Median   = round(sapply(train_data[, num_cols], median),4),
  Mean     = round(sapply(train_data[, num_cols], mean),4),
  Maksimum = round(sapply(train_data[, num_cols], max),4)
)

kable(
  stat_std,
  caption = "Distribusi Data Variabel Independen Setelah Standarisasi"
)

#=============================================================
# 4.7 VISUALISASI DATA
#=============================================================
for (v in num_cols) {
  
  p <- ggplot(train_data, aes_string(x = "Label", y = v, fill = "Label")) +
    geom_boxplot(alpha = 0.6) +
    theme_minimal() +
    labs(
      title = paste("Boxplot Variabel", v, "Berdasarkan Kelas"),
      x = "Label",
      y = "Nilai Standarisasi"
    ) +
    scale_fill_manual(values = c("0" = "skyblue", "1" = "pink")) +
    theme(legend.position = "none")
  
  print(p)
}

#=============================================================
# 4.8 PERSIAPAN MODEL PREDIKSI FINANCIAL DISTRESS
#=============================================================
cat("\nData Training Siap Digunakan :", nrow(train_data))
cat("\nData Testing Siap Digunakan  :", nrow(test_data))

#=============================================================
# FINAL CHECK
#=============================================================
cat("\n\nMissing Value Train:\n")
print(colSums(is.na(train_data)))

cat("\nMissing Value Test:\n")
print(colSums(is.na(test_data)))

cat("\nDimensi Train :", dim(train_data))
cat("\nDimensi Test  :", dim(test_data))

cat("\n\nDATA PREPARATION SELESAI\n")