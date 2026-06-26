library(e1071)
library(caret)
library(PRROC)
library(pROC)
library(ggplot2)
library(knitr)

set.seed(123)

#-------------------------------------------------------------
# 1. SIAPKAN DATA
#-------------------------------------------------------------
train_nb_sel <- train_sel
test_nb_sel  <- test_sel

train_nb_sel$Label <- factor(train_nb_sel$Label, levels = c(0, 1))
test_nb_sel$Label  <- factor(test_nb_sel$Label, levels = c(0, 1))

cat("Variabel yang digunakan (", length(selected_vars), " variabel):\n", sep = "")
print(selected_vars)

#-------------------------------------------------------------
# 2. TRAINING MODEL
#-------------------------------------------------------------
nb_model_sel <- naiveBayes(Label ~ ., data = train_nb_sel, laplace = 1)

#-------------------------------------------------------------
# 3A. PREDIKSI DATA TRAINING
#-------------------------------------------------------------
train_pred_nb_sel <- predict(nb_model_sel, train_nb_sel)
cm_train_nb_sel   <- confusionMatrix(
  train_pred_nb_sel,
  train_nb_sel$Label,
  positive = "1"
)

train_acc_nb_sel <- as.numeric(cm_train_nb_sel$overall["Accuracy"])

cat("=== Confusion Matrix Training ===\n")
print(cm_train_nb_sel$table)
cat("Accuracy Training:", round(train_acc_nb_sel, 4), "\n")

#-------------------------------------------------------------
# 3B. METRIK EVALUASI DATA TRAINING
#-------------------------------------------------------------
tab_train <- cm_train_nb_sel$table
TN_train  <- tab_train[1,1]
FN_train  <- tab_train[1,2]
FP_train  <- tab_train[2,1]
TP_train  <- tab_train[2,2]

# Accuracy training
accuracy_train_nb_sel <- (TP_train + TN_train) / sum(tab_train)

# Kelas 1 (Financial Distress)
precision_1_train_nb_sel <- ifelse((TP_train + FP_train) == 0, 0,
                                   TP_train / (TP_train + FP_train))
recall_1_train_nb_sel <- ifelse((TP_train + FN_train) == 0, 0,
                                TP_train / (TP_train + FN_train))
f1_1_train_nb_sel <- ifelse((precision_1_train_nb_sel + recall_1_train_nb_sel) == 0, 0,
                            2 * precision_1_train_nb_sel * recall_1_train_nb_sel /
                              (precision_1_train_nb_sel + recall_1_train_nb_sel))

# Kelas 0 (Non-Distress)
precision_0_train_nb_sel <- ifelse((TN_train + FN_train) == 0, 0,
                                   TN_train / (TN_train + FN_train))
recall_0_train_nb_sel <- ifelse((TN_train + FP_train) == 0, 0,
                                TN_train / (TN_train + FP_train))
f1_0_train_nb_sel <- ifelse((precision_0_train_nb_sel + recall_0_train_nb_sel) == 0, 0,
                            2 * precision_0_train_nb_sel * recall_0_train_nb_sel /
                              (precision_0_train_nb_sel + recall_0_train_nb_sel))

#-------------------------------------------------------------
# 3C. AUC-ROC & AUC-PR DATA TRAINING
#-------------------------------------------------------------
train_pred_prob_nb_sel <- predict(nb_model_sel, train_nb_sel, type = "raw")[, "1"]

roc_train_nb_sel <- roc(
  train_nb_sel$Label,
  train_pred_prob_nb_sel,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

auc_roc_train_nb_sel <- as.numeric(auc(roc_train_nb_sel))

pr_train_nb_sel <- pr.curve(
  scores.class0 = train_pred_prob_nb_sel[train_nb_sel$Label == "1"],
  scores.class1 = train_pred_prob_nb_sel[train_nb_sel$Label == "0"],
  curve = TRUE
)

auc_pr_train_nb_sel <- pr_train_nb_sel$auc.integral

#-------------------------------------------------------------
# 3D. TABEL RINGKASAN PERFORMA TRAINING
#-------------------------------------------------------------
hasil_train_nb_sel <- data.frame(
  Kelas     = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy  = round(c(accuracy_train_nb_sel, accuracy_train_nb_sel), 4),
  Precision = round(c(precision_0_train_nb_sel, precision_1_train_nb_sel), 4),
  Recall    = round(c(recall_0_train_nb_sel, recall_1_train_nb_sel), 4),
  F1_Score  = round(c(f1_0_train_nb_sel, f1_1_train_nb_sel), 4),
  AUC_PR    = round(c(auc_pr_train_nb_sel, auc_pr_train_nb_sel), 4),
  AUC_ROC   = round(c(auc_roc_train_nb_sel, auc_roc_train_nb_sel), 4)
)

cat("\n=== Tabel Performa Training Naive Bayes (Seleksi Variabel) ===\n")
print(kable(
  hasil_train_nb_sel,
  caption = "Performa Training Model Naive Bayes (Seleksi Variabel)"
))

#-------------------------------------------------------------
# 4. PREDIKSI DATA TESTING
#-------------------------------------------------------------
nb_pred_class_sel <- predict(nb_model_sel, test_nb_sel)
nb_pred_prob_sel  <- predict(nb_model_sel, test_nb_sel, type = "raw")[, "1"]

#-------------------------------------------------------------
# 5. CONFUSION MATRIX TEST
#-------------------------------------------------------------
cm_nb_sel <- confusionMatrix(
  nb_pred_class_sel,
  test_nb_sel$Label,
  positive = "1"
)

cat("\n=== Confusion Matrix Testing ===\n")
print(cm_nb_sel$table)

tab <- cm_nb_sel$table
TN  <- tab[1,1]
FN  <- tab[1,2]
FP  <- tab[2,1]
TP  <- tab[2,2]

#-------------------------------------------------------------
# 6. METRIK EVALUASI
#-------------------------------------------------------------
accuracy_nb_sel <- (TP + TN) / sum(tab)

# Kelas 1 (Financial Distress)
precision_1_nb_sel <- ifelse((TP + FP) == 0, 0,
                             TP / (TP + FP))
recall_1_nb_sel <- ifelse((TP + FN) == 0, 0,
                          TP / (TP + FN))
f1_1_nb_sel <- ifelse((precision_1_nb_sel + recall_1_nb_sel) == 0, 0,
                      2 * precision_1_nb_sel * recall_1_nb_sel /
                        (precision_1_nb_sel + recall_1_nb_sel))

# Kelas 0 (Non-Distress)
precision_0_nb_sel <- ifelse((TN + FN) == 0, 0,
                             TN / (TN + FN))
recall_0_nb_sel <- ifelse((TN + FP) == 0, 0,
                          TN / (TN + FP))
f1_0_nb_sel <- ifelse((precision_0_nb_sel + recall_0_nb_sel) == 0, 0,
                      2 * precision_0_nb_sel * recall_0_nb_sel /
                        (precision_0_nb_sel + recall_0_nb_sel))

#-------------------------------------------------------------
# 7. AUC-ROC
#-------------------------------------------------------------
roc_nb_sel <- roc(
  test_nb_sel$Label,
  nb_pred_prob_sel,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

auc_roc_nb_sel <- as.numeric(auc(roc_nb_sel))

#-------------------------------------------------------------
# 8. AUC-PR
#-------------------------------------------------------------
pr_nb_sel <- pr.curve(
  scores.class0 = nb_pred_prob_sel[test_nb_sel$Label == "1"],
  scores.class1 = nb_pred_prob_sel[test_nb_sel$Label == "0"],
  curve = TRUE
)

auc_pr_nb_sel <- pr_nb_sel$auc.integral

#-------------------------------------------------------------
# 9. OVERFITTING GAP
#-------------------------------------------------------------
gap_nb_sel <- train_acc_nb_sel - accuracy_nb_sel

cat("\n=== Overfitting Check ===\n")
cat("Accuracy Training :", round(train_acc_nb_sel, 4), "\n")
cat("Accuracy Testing  :", round(accuracy_nb_sel, 4), "\n")
cat("Gap (Train - Test):", round(gap_nb_sel, 4), "\n")

if (abs(gap_nb_sel) <= 0.05) {
  cat("Status: Stabil (tidak ada indikasi overfitting)\n")
} else {
  cat("Status: Indikasi overfitting — perlu evaluasi lebih lanjut\n")
}

#-------------------------------------------------------------
# 10. TABEL RINGKASAN PERFORMA TESTING
#-------------------------------------------------------------
hasil_nb_sel <- data.frame(
  Kelas     = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy  = round(c(accuracy_nb_sel, accuracy_nb_sel), 4),
  Precision = round(c(precision_0_nb_sel, precision_1_nb_sel), 4),
  Recall    = round(c(recall_0_nb_sel, recall_1_nb_sel), 4),
  F1_Score  = round(c(f1_0_nb_sel, f1_1_nb_sel), 4),
  AUC_PR    = round(c(auc_pr_nb_sel, auc_pr_nb_sel), 4),
  AUC_ROC   = round(c(auc_roc_nb_sel, auc_roc_nb_sel), 4)
)

cat("\n=== Tabel Performa Naive Bayes (Seleksi Variabel) ===\n")
print(kable(
  hasil_nb_sel,
  caption = "Performa Model Naive Bayes (Seleksi Variabel)"
))

#-------------------------------------------------------------
# 11. ROC CURVE
#-------------------------------------------------------------
roc_df_nb_sel <- data.frame(
  FPR = 1 - roc_nb_sel$specificities,
  TPR = roc_nb_sel$sensitivities
)

ggplot(roc_df_nb_sel, aes(FPR, TPR)) +
  geom_line(linewidth = 1, color = "blue") +
  geom_abline(slope = 1, intercept = 0,
              linetype = "dashed", color = "gray") +
  annotate("text", x = 0.7, y = 0.2,
           label = paste0("AUC = ", round(auc_roc_nb_sel, 4)),
           size = 4) +
  labs(
    title = "ROC Curve — Naive Bayes (Seleksi Variabel)",
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal()

#-------------------------------------------------------------
# 12. PR CURVE
#-------------------------------------------------------------
pr_df_nb_sel <- data.frame(
  Recall = pr_nb_sel$curve[,1],
  Precision = pr_nb_sel$curve[,2]
)

ggplot(pr_df_nb_sel, aes(Recall, Precision)) +
  geom_line(linewidth = 1, color = "red") +
  annotate("text", x = 0.3, y = 0.2,
           label = paste0("AUC-PR = ", round(auc_pr_nb_sel, 4)),
           size = 4) +
  labs(
    title = "Precision-Recall Curve — Naive Bayes (Seleksi Variabel)",
    x = "Recall",
    y = "Precision"
  ) +
  theme_minimal()

#-------------------------------------------------------------
# 13. ACTUAL VS PREDICTED PLOT
#-------------------------------------------------------------
plot_nb_sel <- data.frame(
  Index    = 1:nrow(test_nb_sel),
  Aktual   = as.numeric(as.character(test_nb_sel$Label)),
  Prediksi = as.numeric(as.character(nb_pred_class_sel))
)

ggplot(plot_nb_sel, aes(x = Index)) +
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
    breaks = c(0,1),
    limits = c(-0.05, 1.05)
  ) +
  labs(
    title = "Actual vs Predicted - Naive Bayes (Seleksi Variabel)",
    x = "",
    y = "Label",
    color = "Keterangan"
  ) +
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "right"
  )