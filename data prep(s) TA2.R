library(dplyr)

set.seed(123)

#-------------------------------------------------------------
# 1. T-TEST SELEKSI VARIABEL — DARI TRAIN ONLY
#-------------------------------------------------------------
num_cols <- setdiff(names(train_data), "Label")

result_ttest_sel <- data.frame()

for (v in num_cols) {
  x0   <- train_data[[v]][train_data$Label == 0]
  x1   <- train_data[[v]][train_data$Label == 1]
  tval <- tryCatch(t.test(x0, x1), error = function(e) list(p.value = NA))
  
  result_ttest_sel <- rbind(result_ttest_sel, data.frame(
    Variabel         = v,
    Mean_NonDistress = round(mean(x0, na.rm = TRUE), 4),
    Mean_Distress    = round(mean(x1, na.rm = TRUE), 4),
    P_value          = round(tval$p.value, 4),
    Signifikan       = ifelse(!is.na(tval$p.value) & tval$p.value < 0.05,
                              "YA", "TIDAK")
  ))
}

cat("=== Hasil T-Test Seleksi Variabel (Train Data) ===\n")
print(result_ttest_sel)

#-------------------------------------------------------------
# 2. DAFTAR VARIABEL TERPILIH (p < 0.05)
#-------------------------------------------------------------
selected_vars <- result_ttest_sel$Variabel[
  !is.na(result_ttest_sel$P_value) & result_ttest_sel$P_value < 0.05
]

cat("\nVariabel terpilih (", length(selected_vars), " variabel):\n", sep = "")
print(selected_vars)

#-------------------------------------------------------------
# 3. BUAT SUBSET DATA DENGAN VARIABEL TERPILIH
#-------------------------------------------------------------
all_selected <- c(selected_vars, "Label")

train_sel <- train_data[, all_selected]
test_sel  <- test_data[, all_selected]

train_sel$Label <- factor(train_sel$Label, levels = c(0, 1))
test_sel$Label  <- factor(test_sel$Label,  levels = c(0, 1))

#-------------------------------------------------------------
# 4. CEK AKHIR
#-------------------------------------------------------------
cat("\n=== Distribusi Label (Data Seleksi) ===\n")
cat("Train:\n"); print(table(train_sel$Label))
cat("Test:\n");  print(table(test_sel$Label))

cat("\nDimensi Train Sel:", dim(train_sel), "\n")
cat("Dimensi Test Sel :", dim(test_sel),  "\n")