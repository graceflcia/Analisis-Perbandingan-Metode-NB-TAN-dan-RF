library(randomForest)
library(caret)
library(PRROC)
library(pROC)
library(ggplot2)
library(dplyr)
library(knitr)

set.seed(123)

#-------------------------------------------------------------
# 1. SIAPKAN DATA
#-------------------------------------------------------------
train_rf_sel <- train_sel
test_rf_sel  <- test_sel

train_rf_sel$Label <- factor(train_rf_sel$Label, levels = c(0, 1))
test_rf_sel$Label  <- factor(test_rf_sel$Label, levels = c(0, 1))

p_rf_sel <- ncol(train_rf_sel) - 1

cat("Variabel yang digunakan (", p_rf_sel, " variabel):\n", sep = "")
print(setdiff(names(train_rf_sel), "Label"))

#-------------------------------------------------------------
# 2. GRID SEARCH — ntree x mtry
#-------------------------------------------------------------
ntree_vals_sel <- c(100, 200, 300)
mtry_vals_sel  <- unique(floor(c(sqrt(p_rf_sel), p_rf_sel / 3, p_rf_sel / 2)))

model_list_rf_sel  <- list()
result_grid_rf_sel <- data.frame()

cat("\n=== Grid Search Random Forest (Seleksi Variabel) ===\n")
cat("Jumlah variabel:", p_rf_sel,
    "| mtry dicoba:", paste(mtry_vals_sel, collapse = ", "), "\n\n")

for (n in ntree_vals_sel) {
  for (m in mtry_vals_sel) {
    
    set.seed(123)
    model_tmp <- randomForest(
      Label ~ .,
      data       = train_rf_sel,
      ntree      = n,
      mtry       = m,
      importance = TRUE
    )
    
    pred_prob_tmp  <- predict(model_tmp, test_rf_sel, type = "prob")[, "1"]
    pred_class_tmp <- predict(model_tmp, test_rf_sel)
    
    cm_tmp <- confusionMatrix(
      pred_class_tmp,
      test_rf_sel$Label,
      positive = "1"
    )
    
    pr_tmp <- pr.curve(
      scores.class0 = pred_prob_tmp[test_rf_sel$Label == "1"],
      scores.class1 = pred_prob_tmp[test_rf_sel$Label == "0"],
      curve = FALSE
    )
    
    roc_tmp <- roc(
      test_rf_sel$Label,
      pred_prob_tmp,
      levels = c("0", "1"),
      direction = "<",
      quiet = TRUE
    )
    
    auc_roc_tmp <- as.numeric(auc(roc_tmp))
    
    model_name <- paste0("ntree", n, "_mtry", m)
    model_list_rf_sel[[model_name]] <- model_tmp
    
    result_grid_rf_sel <- rbind(
      result_grid_rf_sel,
      data.frame(
        Model    = model_name,
        Ntree    = n,
        Mtry     = m,
        Accuracy = round(as.numeric(cm_tmp$overall["Accuracy"]), 4),
        AUC_PR   = round(pr_tmp$auc.integral, 4),
        AUC_ROC  = round(auc_roc_tmp, 4)
      )
    )
    
    cat(
      "ntree =", n,
      "| mtry =", m,
      "| AUC-PR =", round(pr_tmp$auc.integral, 4),
      "| AUC-ROC =", round(auc_roc_tmp, 4), "\n"
    )
  }
}

cat("\n")
print(kable(
  result_grid_rf_sel,
  caption = "Hasil Grid Search Random Forest (Seleksi Variabel)"
))

#-------------------------------------------------------------
# 3. PILIH MODEL TERBAIK BERDASARKAN AUC-PR TERTINGGI
#-------------------------------------------------------------
best_idx_sel  <- which.max(result_grid_rf_sel$AUC_PR)
best_name_sel <- result_grid_rf_sel$Model[best_idx_sel]
rf_model_sel  <- model_list_rf_sel[[best_name_sel]]

cat("\n=== Model Terbaik ===\n")
cat("Nama  :", best_name_sel, "\n")
cat("Ntree :", result_grid_rf_sel$Ntree[best_idx_sel], "\n")
cat("Mtry  :", result_grid_rf_sel$Mtry[best_idx_sel], "\n")
cat("AUC-PR:", result_grid_rf_sel$AUC_PR[best_idx_sel], "\n")

#-------------------------------------------------------------
# 4A. PREDIKSI DATA TRAINING
#-------------------------------------------------------------
train_pred_rf_sel <- predict(rf_model_sel, train_rf_sel)
cm_train_rf_sel   <- confusionMatrix(
  train_pred_rf_sel,
  train_rf_sel$Label,
  positive = "1"
)

train_acc_rf_sel <- as.numeric(cm_train_rf_sel$overall["Accuracy"])

cat("\n=== Confusion Matrix Training ===\n")
print(cm_train_rf_sel$table)
cat("Accuracy Training:", round(train_acc_rf_sel, 4), "\n")

#-------------------------------------------------------------
# 4B. METRIK EVALUASI DATA TRAINING
#-------------------------------------------------------------
rf_train_prob_sel <- predict(rf_model_sel, train_rf_sel, type = "prob")[, "1"]

tab_train <- cm_train_rf_sel$table
TN_train  <- tab_train[1, 1]
FN_train  <- tab_train[1, 2]
FP_train  <- tab_train[2, 1]
TP_train  <- tab_train[2, 2]

accuracy_train_rf_sel <- (TP_train + TN_train) / sum(tab_train)

# Kelas 1
precision_1_train_rf_sel <- ifelse((TP_train + FP_train) == 0, 0,
                                   TP_train / (TP_train + FP_train))
recall_1_train_rf_sel    <- ifelse((TP_train + FN_train) == 0, 0,
                                   TP_train / (TP_train + FN_train))
f1_1_train_rf_sel        <- ifelse(
  (precision_1_train_rf_sel + recall_1_train_rf_sel) == 0, 0,
  2 * precision_1_train_rf_sel * recall_1_train_rf_sel /
    (precision_1_train_rf_sel + recall_1_train_rf_sel)
)

# Kelas 0
precision_0_train_rf_sel <- ifelse((TN_train + FN_train) == 0, 0,
                                   TN_train / (TN_train + FN_train))
recall_0_train_rf_sel    <- ifelse((TN_train + FP_train) == 0, 0,
                                   TN_train / (TN_train + FP_train))
f1_0_train_rf_sel        <- ifelse(
  (precision_0_train_rf_sel + recall_0_train_rf_sel) == 0, 0,
  2 * precision_0_train_rf_sel * recall_0_train_rf_sel /
    (precision_0_train_rf_sel + recall_0_train_rf_sel)
)

roc_train_rf_sel <- roc(
  train_rf_sel$Label,
  rf_train_prob_sel,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

auc_roc_train_rf_sel <- as.numeric(auc(roc_train_rf_sel))

pr_train_rf_sel <- pr.curve(
  scores.class0 = rf_train_prob_sel[train_rf_sel$Label == "1"],
  scores.class1 = rf_train_prob_sel[train_rf_sel$Label == "0"],
  curve = TRUE
)

auc_pr_train_rf_sel <- pr_train_rf_sel$auc.integral

hasil_train_rf_sel <- data.frame(
  Kelas     = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy  = round(c(accuracy_train_rf_sel, accuracy_train_rf_sel), 4),
  Precision = round(c(precision_0_train_rf_sel, precision_1_train_rf_sel), 4),
  Recall    = round(c(recall_0_train_rf_sel, recall_1_train_rf_sel), 4),
  F1_Score  = round(c(f1_0_train_rf_sel, f1_1_train_rf_sel), 4),
  AUC_PR    = round(c(auc_pr_train_rf_sel, auc_pr_train_rf_sel), 4),
  AUC_ROC   = round(c(auc_roc_train_rf_sel, auc_roc_train_rf_sel), 4)
)

cat("\n=== Tabel Performa Training Random Forest (Seleksi Variabel) ===\n")
print(kable(
  hasil_train_rf_sel,
  caption = "Performa Training Model Random Forest (Seleksi Variabel)"
))

#-------------------------------------------------------------
# 5. PREDIKSI DATA TESTING
#-------------------------------------------------------------
rf_pred_class_sel <- predict(rf_model_sel, test_rf_sel)
rf_pred_prob_sel  <- predict(rf_model_sel, test_rf_sel, type = "prob")[, "1"]

#-------------------------------------------------------------
# 6. CONFUSION MATRIX TEST
#-------------------------------------------------------------
cm_rf_sel <- confusionMatrix(
  rf_pred_class_sel,
  test_rf_sel$Label,
  positive = "1"
)

cat("\n=== Confusion Matrix Testing ===\n")
print(cm_rf_sel$table)

tab <- cm_rf_sel$table
TN  <- tab[1, 1]
FN  <- tab[1, 2]
FP  <- tab[2, 1]
TP  <- tab[2, 2]

#-------------------------------------------------------------
# 7. METRIK EVALUASI
#-------------------------------------------------------------
accuracy_rf_sel <- (TP + TN) / sum(tab)

precision_1_rf_sel <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
recall_1_rf_sel    <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
f1_1_rf_sel        <- ifelse((precision_1_rf_sel + recall_1_rf_sel) == 0, 0,
                             2 * precision_1_rf_sel * recall_1_rf_sel /
                               (precision_1_rf_sel + recall_1_rf_sel))

precision_0_rf_sel <- ifelse((TN + FN) == 0, 0, TN / (TN + FN))
recall_0_rf_sel    <- ifelse((TN + FP) == 0, 0, TN / (TN + FP))
f1_0_rf_sel        <- ifelse((precision_0_rf_sel + recall_0_rf_sel) == 0, 0,
                             2 * precision_0_rf_sel * recall_0_rf_sel /
                               (precision_0_rf_sel + recall_0_rf_sel))

#-------------------------------------------------------------
# 8. AUC-ROC
#-------------------------------------------------------------
roc_rf_sel <- roc(
  test_rf_sel$Label,
  rf_pred_prob_sel,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

auc_roc_rf_sel <- as.numeric(auc(roc_rf_sel))

#-------------------------------------------------------------
# 9. AUC-PR
#-------------------------------------------------------------
pr_rf_sel <- pr.curve(
  scores.class0 = rf_pred_prob_sel[test_rf_sel$Label == "1"],
  scores.class1 = rf_pred_prob_sel[test_rf_sel$Label == "0"],
  curve = TRUE
)

auc_pr_rf_sel <- pr_rf_sel$auc.integral

#-------------------------------------------------------------
# 10. OVERFITTING GAP
#-------------------------------------------------------------
gap_rf_sel <- accuracy_train_rf_sel - accuracy_rf_sel

cat("\n=== Overfitting Check ===\n")
cat("Accuracy Training :", round(accuracy_train_rf_sel, 4), "\n")
cat("Accuracy Testing  :", round(accuracy_rf_sel, 4), "\n")
cat("Gap (Train - Test):", round(gap_rf_sel, 4), "\n")

if (abs(gap_rf_sel) <= 0.05) {
  cat("Status: Stabil (tidak ada indikasi overfitting)\n")
} else {
  cat("Status: Indikasi overfitting — perlu evaluasi lebih lanjut\n")
}

#-------------------------------------------------------------
# 11. TABEL RINGKASAN PERFORMA
#-------------------------------------------------------------
hasil_rf_sel <- data.frame(
  Kelas     = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy  = round(c(accuracy_rf_sel, accuracy_rf_sel), 4),
  Precision = round(c(precision_0_rf_sel, precision_1_rf_sel), 4),
  Recall    = round(c(recall_0_rf_sel, recall_1_rf_sel), 4),
  F1_Score  = round(c(f1_0_rf_sel, f1_1_rf_sel), 4),
  AUC_PR    = round(c(auc_pr_rf_sel, auc_pr_rf_sel), 4),
  AUC_ROC   = round(c(auc_roc_rf_sel, auc_roc_rf_sel), 4)
)

cat("\n=== Tabel Performa Random Forest (Seleksi Variabel) ===\n")
print(kable(
  hasil_rf_sel,
  caption = "Performa Model Random Forest (Seleksi Variabel)"
))

#-------------------------------------------------------------
# 12. VARIABLE IMPORTANCE
#-------------------------------------------------------------
imp_rf_sel <- importance(rf_model_sel, type = 1)

imp_df_rf_sel <- data.frame(
  Variabel   = rownames(imp_rf_sel),
  MeanDecAcc = round(imp_rf_sel[, 1], 4)
) %>% arrange(desc(MeanDecAcc))

cat("\n=== Variable Importance (Mean Decrease Accuracy) ===\n")
print(kable(
  imp_df_rf_sel,
  caption = "Pentingnya Variabel — RF (Seleksi Variabel)"
))

varImpPlot(
  rf_model_sel,
  type = 1,
  main = "Variable Importance — Random Forest (Seleksi Variabel)",
  col = "steelblue",
  pch = 16
)

#-------------------------------------------------------------
# 13. ROC CURVE
#-------------------------------------------------------------
roc_df_rf_sel <- data.frame(
  FPR = 1 - roc_rf_sel$specificities,
  TPR = roc_rf_sel$sensitivities
)

ggplot(roc_df_rf_sel, aes(FPR, TPR)) +
  geom_line(linewidth = 1, color = "blue") +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "gray") +
  annotate("text", x = 0.7, y = 0.2,
           label = paste0("AUC = ", round(auc_roc_rf_sel, 4)), size = 4) +
  labs(
    title = "ROC Curve — Random Forest (Seleksi Variabel)",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal()

#-------------------------------------------------------------
# 14. PR CURVE
#-------------------------------------------------------------
pr_df_rf_sel <- data.frame(
  Recall    = pr_rf_sel$curve[, 1],
  Precision = pr_rf_sel$curve[, 2]
)

ggplot(pr_df_rf_sel, aes(Recall, Precision)) +
  geom_line(linewidth = 1, color = "red") +
  annotate("text", x = 0.3, y = 0.2,
           label = paste0("AUC-PR = ", round(auc_pr_rf_sel, 4)), size = 4) +
  labs(
    title = "Precision-Recall Curve — Random Forest (Seleksi Variabel)",
    x = "Recall",
    y = "Precision"
  ) +
  theme_minimal()

#-------------------------------------------------------------
# 15. ACTUAL VS PREDICTED PLOT
#-------------------------------------------------------------
plot_rf_sel <- data.frame(
  Index    = 1:nrow(test_rf_sel),
  Aktual   = as.numeric(as.character(test_rf_sel$Label)),
  Prediksi = as.numeric(as.character(rf_pred_class_sel))
)

ggplot(plot_rf_sel, aes(x = Index)) +
  geom_point(aes(y = Aktual, color = "Actual"),
             size = 2, shape = 16) +
  geom_line(aes(y = Prediksi, color = "Predicted"),
            linewidth = 0.5) +
  scale_color_manual(
    values = c(
      "Actual" = "red",
      "Predicted" = "blue"
    )
  ) +
  scale_y_continuous(
    breaks = c(0, 1),
    limits = c(-0.05, 1.05)
  ) +
  labs(
    title = "Actual vs Predicted - Random Forest (Seleksi Variabel)",
    x = "",
    y = "Label",
    color = "Keterangan"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "right"
  )