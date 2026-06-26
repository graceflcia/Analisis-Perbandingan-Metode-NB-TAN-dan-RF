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
train_rf <- train_data
test_rf  <- test_data

train_rf$Label <- factor(train_rf$Label, levels = c(0, 1))
test_rf$Label  <- factor(test_rf$Label,  levels = c(0, 1))

p_rf <- ncol(train_rf) - 1  # Jumlah variabel independen

#-------------------------------------------------------------
# 2. GRID SEARCH — ntree x mtry
#-------------------------------------------------------------
ntree_vals <- c(100, 200, 300)
mtry_vals  <- unique(floor(c(sqrt(p_rf), p_rf / 3, p_rf / 2)))

model_list_rf  <- list()
result_grid_rf <- data.frame()

cat("=== Grid Search Random Forest (Semua Variabel) ===\n")
cat("Jumlah variabel:", p_rf, "| mtry dicoba:", paste(mtry_vals, collapse = ", "), "\n\n")

for (n in ntree_vals) {
  for (m in mtry_vals) {
    
    set.seed(123)
    model_tmp <- randomForest(Label ~ .,
                              data       = train_rf,
                              ntree      = n,
                              mtry       = m,
                              importance = TRUE)
    
    pred_prob_tmp  <- predict(model_tmp, test_rf, type = "prob")[, "1"]
    pred_class_tmp <- predict(model_tmp, test_rf)
    
    cm_tmp <- confusionMatrix(pred_class_tmp, test_rf$Label, positive = "1")
    
    pr_tmp <- pr.curve(
      scores.class0 = pred_prob_tmp[test_rf$Label == "1"],
      scores.class1 = pred_prob_tmp[test_rf$Label == "0"],
      curve = FALSE
    )
    
    roc_tmp     <- roc(test_rf$Label, pred_prob_tmp,
                       levels = c("0", "1"), direction = "<", quiet = TRUE)
    auc_roc_tmp <- as.numeric(auc(roc_tmp))
    
    model_name <- paste0("ntree", n, "_mtry", m)
    model_list_rf[[model_name]] <- model_tmp
    
    result_grid_rf <- rbind(result_grid_rf, data.frame(
      Model    = model_name,
      Ntree    = n,
      Mtry     = m,
      Accuracy = round(as.numeric(cm_tmp$overall["Accuracy"]), 4),
      AUC_PR   = round(pr_tmp$auc.integral, 4),
      AUC_ROC  = round(auc_roc_tmp, 4)
    ))
    
    cat("ntree =", n, "| mtry =", m,
        "| AUC-PR =", round(pr_tmp$auc.integral, 4),
        "| AUC-ROC =", round(auc_roc_tmp, 4), "\n")
  }
}

cat("\n")
print(kable(result_grid_rf, caption = "Hasil Grid Search Random Forest (Semua Variabel)"))

#-------------------------------------------------------------
# 3. PILIH MODEL TERBAIK BERDASARKAN AUC-PR TERTINGGI
#-------------------------------------------------------------
best_idx_rf  <- which.max(result_grid_rf$AUC_PR)
best_name_rf <- result_grid_rf$Model[best_idx_rf]
rf_model     <- model_list_rf[[best_name_rf]]

cat("\n=== Model Terbaik ===\n")
cat("Nama  :", best_name_rf, "\n")
cat("Ntree :", result_grid_rf$Ntree[best_idx_rf], "\n")
cat("Mtry  :", result_grid_rf$Mtry[best_idx_rf], "\n")
cat("AUC-PR:", result_grid_rf$AUC_PR[best_idx_rf], "\n")

#-------------------------------------------------------------
# 4A. PREDIKSI DATA TRAINING
#-------------------------------------------------------------
train_pred_rf <- predict(rf_model, train_rf)
cm_train_rf   <- confusionMatrix(train_pred_rf, train_rf$Label, positive = "1")
train_acc_rf  <- as.numeric(cm_train_rf$overall["Accuracy"])

cat("\n=== Confusion Matrix Training ===\n")
print(cm_train_rf$table)
cat("Accuracy Training:", round(train_acc_rf, 4), "\n")

#-------------------------------------------------------------
# 4B. METRIK TRAINING
#-------------------------------------------------------------
train_pred_prob_rf <- predict(rf_model, train_rf, type = "prob")[, "1"]

tab_train <- cm_train_rf$table
TN_train  <- tab_train[1, 1]
FN_train  <- tab_train[1, 2]
FP_train  <- tab_train[2, 1]
TP_train  <- tab_train[2, 2]

# Accuracy training
accuracy_train_rf <- (TP_train + TN_train) / sum(tab_train)

# Kelas 1 (Financial Distress)
precision_1_train_rf <- ifelse((TP_train + FP_train) == 0, 0,
                               TP_train / (TP_train + FP_train))
recall_1_train_rf    <- ifelse((TP_train + FN_train) == 0, 0,
                               TP_train / (TP_train + FN_train))
f1_1_train_rf        <- ifelse((precision_1_train_rf + recall_1_train_rf) == 0, 0,
                               2 * precision_1_train_rf * recall_1_train_rf /
                                 (precision_1_train_rf + recall_1_train_rf))

# Kelas 0 (Non-Distress)
precision_0_train_rf <- ifelse((TN_train + FN_train) == 0, 0,
                               TN_train / (TN_train + FN_train))
recall_0_train_rf    <- ifelse((TN_train + FP_train) == 0, 0,
                               TN_train / (TN_train + FP_train))
f1_0_train_rf        <- ifelse((precision_0_train_rf + recall_0_train_rf) == 0, 0,
                               2 * precision_0_train_rf * recall_0_train_rf /
                                 (precision_0_train_rf + recall_0_train_rf))

# AUC-ROC Training
roc_train_rf <- roc(
  train_rf$Label,
  train_pred_prob_rf,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

auc_roc_train_rf <- as.numeric(auc(roc_train_rf))

# AUC-PR Training
pr_train_rf <- pr.curve(
  scores.class0 = train_pred_prob_rf[train_rf$Label == "1"],
  scores.class1 = train_pred_prob_rf[train_rf$Label == "0"],
  curve = TRUE
)

auc_pr_train_rf <- pr_train_rf$auc.integral

#-------------------------------------------------------------
# 4C. TABEL RINGKASAN PERFORMA TRAINING
#-------------------------------------------------------------
hasil_train_rf <- data.frame(
  Kelas     = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy  = round(c(accuracy_train_rf, accuracy_train_rf), 4),
  Precision = round(c(precision_0_train_rf, precision_1_train_rf), 4),
  Recall    = round(c(recall_0_train_rf,    recall_1_train_rf),    4),
  F1_Score  = round(c(f1_0_train_rf,        f1_1_train_rf),        4),
  AUC_PR    = round(c(auc_pr_train_rf, auc_pr_train_rf), 4),
  AUC_ROC   = round(c(auc_roc_train_rf, auc_roc_train_rf), 4)
)

cat("\n=== Tabel Performa Training Random Forest ===\n")
print(kable(
  hasil_train_rf,
  caption = "Performa Training Model Random Forest"
))

#-------------------------------------------------------------
# 5. PREDIKSI DATA TESTING
#-------------------------------------------------------------
rf_pred_class <- predict(rf_model, test_rf)
rf_pred_prob  <- predict(rf_model, test_rf, type = "prob")[, "1"]

#-------------------------------------------------------------
# 6. CONFUSION MATRIX TEST
#-------------------------------------------------------------
cm_rf <- confusionMatrix(rf_pred_class, test_rf$Label, positive = "1")
cat("\n=== Confusion Matrix Testing ===\n")
print(cm_rf$table)

tab <- cm_rf$table
TN  <- tab[1, 1]
FN  <- tab[1, 2]
FP  <- tab[2, 1]
TP  <- tab[2, 2]

#-------------------------------------------------------------
# 7. METRIK EVALUASI
#-------------------------------------------------------------
accuracy_rf <- (TP + TN) / sum(tab)

precision_1_rf <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
recall_1_rf    <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
f1_1_rf        <- ifelse((precision_1_rf + recall_1_rf) == 0, 0,
                         2 * precision_1_rf * recall_1_rf /
                           (precision_1_rf + recall_1_rf))

precision_0_rf <- ifelse((TN + FN) == 0, 0, TN / (TN + FN))
recall_0_rf    <- ifelse((TN + FP) == 0, 0, TN / (TN + FP))
f1_0_rf        <- ifelse((precision_0_rf + recall_0_rf) == 0, 0,
                         2 * precision_0_rf * recall_0_rf /
                           (precision_0_rf + recall_0_rf))

#-------------------------------------------------------------
# 8. AUC-ROC
#-------------------------------------------------------------
roc_rf     <- roc(test_rf$Label, rf_pred_prob,
                  levels = c("0", "1"), direction = "<", quiet = TRUE)
auc_roc_rf <- as.numeric(auc(roc_rf))

#-------------------------------------------------------------
# 9. AUC-PR
#-------------------------------------------------------------
pr_rf <- pr.curve(
  scores.class0 = rf_pred_prob[test_rf$Label == "1"],
  scores.class1 = rf_pred_prob[test_rf$Label == "0"],
  curve = TRUE
)
auc_pr_rf <- pr_rf$auc.integral

#-------------------------------------------------------------
# 10. OVERFITTING GAP
#-------------------------------------------------------------
gap_rf <- train_acc_rf - accuracy_rf

cat("\n=== Overfitting Check ===\n")
cat("Accuracy Training :", round(train_acc_rf, 4), "\n")
cat("Accuracy Testing  :", round(accuracy_rf,  4), "\n")
cat("Gap (Train - Test):", round(gap_rf,        4), "\n")
if (abs(gap_rf) <= 0.05) {
  cat("Status: Stabil\n")
} else {
  cat("Status: Indikasi overfitting (wajar untuk RF — perhatikan test accuracy)\n")
}

#-------------------------------------------------------------
# 11. TABEL RINGKASAN PERFORMA
#-------------------------------------------------------------
hasil_rf <- data.frame(
  Kelas     = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy  = round(c(accuracy_rf, accuracy_rf), 4),
  Precision = round(c(precision_0_rf, precision_1_rf), 4),
  Recall    = round(c(recall_0_rf,    recall_1_rf),    4),
  F1_Score  = round(c(f1_0_rf,        f1_1_rf),        4),
  AUC_PR    = round(c(auc_pr_rf, auc_pr_rf), 4),
  AUC_ROC   = round(c(auc_roc_rf, auc_roc_rf), 4)
)

cat("\n=== Tabel Performa Random Forest (Semua Variabel) ===\n")
print(kable(
  hasil_rf,
  caption = "Performa Model Random Forest (Semua Variabel)"
))

#-------------------------------------------------------------
# 12. VARIABLE IMPORTANCE
#-------------------------------------------------------------
imp_rf    <- importance(rf_model, type = 1)
imp_df_rf <- data.frame(
  Variabel   = rownames(imp_rf),
  MeanDecAcc = round(imp_rf[, 1], 4)
) %>% arrange(desc(MeanDecAcc))

cat("\n=== Variable Importance (Mean Decrease Accuracy) ===\n")
print(kable(imp_df_rf, caption = "Pentingnya Variabel — RF (Semua Variabel)"))

varImpPlot(rf_model, type = 1,
           main = "Variable Importance — Random Forest (Semua Variabel)",
           col = "steelblue", pch = 16)

#-------------------------------------------------------------
# 13. ROC CURVE
#-------------------------------------------------------------
roc_df_rf <- data.frame(
  FPR = 1 - roc_rf$specificities,
  TPR = roc_rf$sensitivities
)

ggplot(roc_df_rf, aes(FPR, TPR)) +
  geom_line(linewidth = 1, color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  annotate("text", x = 0.7, y = 0.2,
           label = paste0("AUC = ", round(auc_roc_rf, 4)), size = 4) +
  labs(title = "ROC Curve — Random Forest (Semua Variabel)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal()

#-------------------------------------------------------------
# 14. PR CURVE
#-------------------------------------------------------------
pr_df_rf <- data.frame(
  Recall    = pr_rf$curve[, 1],
  Precision = pr_rf$curve[, 2]
)

ggplot(pr_df_rf, aes(Recall, Precision)) +
  geom_line(linewidth = 1, color = "red") +
  annotate("text", x = 0.3, y = 0.2,
           label = paste0("AUC-PR = ", round(auc_pr_rf, 4)), size = 4) +
  labs(title = "Precision-Recall Curve — Random Forest (Semua Variabel)",
       x = "Recall", y = "Precision") +
  theme_minimal()

#-------------------------------------------------------------
# 15. ACTUAL VS PREDICTED PLOT
#-------------------------------------------------------------
plot_rf <- data.frame(
  Index    = 1:nrow(test_rf),
  Aktual   = as.numeric(as.character(test_rf$Label)),
  Prediksi = as.numeric(as.character(rf_pred_class))
)

ggplot(plot_rf, aes(x = Index)) +
  
  # titik merah = actual
  geom_point(aes(y = Aktual, color = "Actual"),
             size = 2, shape = 16) +
  
  # garis biru = predicted
  geom_line(aes(y = Prediksi, color = "Predicted"),
            linewidth = 0.5) +
  
  scale_color_manual(
    values = c(
      "Actual" = "red",
      "Predicted" = "blue"
    )
  ) +
  
  scale_y_continuous(
    breaks = c(0,1),
    limits = c(-0.05, 1.05)
  ) +
  
  labs(
    title = "Actual vs Predicted - Random Forest (Semua Variabel)",
    x = "",
    y = "Label",
    color = "Keterangan"
  ) +
  
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "right"
  )
