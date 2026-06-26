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
train_nb <- train_data
test_nb  <- test_data

train_nb$Label <- factor(train_nb$Label, levels = c(0, 1))
test_nb$Label  <- factor(test_nb$Label,  levels = c(0, 1))

#-------------------------------------------------------------
# 2. TRAINING MODEL
#-------------------------------------------------------------
nb_model <- naiveBayes(Label ~ ., data = train_nb, laplace = 1)

#-------------------------------------------------------------
# 3A. PREDIKSI DATA TRAINING
#-------------------------------------------------------------
train_pred_nb <- predict(nb_model, train_nb)
cm_train_nb   <- confusionMatrix(train_pred_nb, train_nb$Label, positive = "1")
train_acc_nb  <- as.numeric(cm_train_nb$overall["Accuracy"])

cat("=== Confusion Matrix Training ===\n")
print(cm_train_nb$table)
cat("Accuracy Training:", round(train_acc_nb, 4), "\n")

#-------------------------------------------------------------
# 3B. METRIK EVALUASI DATA TRAINING
#-------------------------------------------------------------
tab_train <- cm_train_nb$table
TN_train  <- tab_train[1, 1]
FN_train  <- tab_train[1, 2]
FP_train  <- tab_train[2, 1]
TP_train  <- tab_train[2, 2]

# Accuracy training
accuracy_train_nb <- (TP_train + TN_train) / sum(tab_train)

# Kelas 1 (Financial Distress)
precision_1_train_nb <- ifelse((TP_train + FP_train) == 0, 0,
                               TP_train / (TP_train + FP_train))
recall_1_train_nb    <- ifelse((TP_train + FN_train) == 0, 0,
                               TP_train / (TP_train + FN_train))
f1_1_train_nb        <- ifelse((precision_1_train_nb + recall_1_train_nb) == 0, 0,
                               2 * precision_1_train_nb * recall_1_train_nb /
                                 (precision_1_train_nb + recall_1_train_nb))

# Kelas 0 (Non-Distress)
precision_0_train_nb <- ifelse((TN_train + FN_train) == 0, 0,
                               TN_train / (TN_train + FN_train))
recall_0_train_nb    <- ifelse((TN_train + FP_train) == 0, 0,
                               TN_train / (TN_train + FP_train))
f1_0_train_nb        <- ifelse((precision_0_train_nb + recall_0_train_nb) == 0, 0,
                               2 * precision_0_train_nb * recall_0_train_nb /
                                 (precision_0_train_nb + recall_0_train_nb))

#-------------------------------------------------------------
# 3C. AUC-ROC & AUC-PR DATA TRAINING
#-------------------------------------------------------------
train_pred_prob_nb <- predict(nb_model, train_nb, type = "raw")[, "1"]

roc_train_nb <- roc(
  train_nb$Label,
  train_pred_prob_nb,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

auc_roc_train_nb <- as.numeric(auc(roc_train_nb))

pr_train_nb <- pr.curve(
  scores.class0 = train_pred_prob_nb[train_nb$Label == "1"],
  scores.class1 = train_pred_prob_nb[train_nb$Label == "0"],
  curve = TRUE
)

auc_pr_train_nb <- pr_train_nb$auc.integral

#-------------------------------------------------------------
# 3D. TABEL RINGKASAN PERFORMA TRAINING
#-------------------------------------------------------------
hasil_train_nb <- data.frame(
  Kelas     = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy  = round(c(accuracy_train_nb, accuracy_train_nb), 4),
  Precision = round(c(precision_0_train_nb, precision_1_train_nb), 4),
  Recall    = round(c(recall_0_train_nb,    recall_1_train_nb),    4),
  F1_Score  = round(c(f1_0_train_nb,        f1_1_train_nb),        4),
  AUC_PR    = round(c(auc_pr_train_nb, auc_pr_train_nb), 4),
  AUC_ROC   = round(c(auc_roc_train_nb, auc_roc_train_nb), 4)
)

cat("\n=== Tabel Performa Training Naive Bayes ===\n")
print(kable(
  hasil_train_nb,
  caption = "Performa Training Model Naive Bayes"
))

#-------------------------------------------------------------
# 4. PREDIKSI DATA TESTING
#-------------------------------------------------------------
nb_pred_class <- predict(nb_model, test_nb)
nb_pred_prob  <- predict(nb_model, test_nb, type = "raw")[, "1"]
# nb_pred_prob: probabilitas prediksi kelas 1 (Financial Distress)

#-------------------------------------------------------------
# 5. CONFUSION MATRIX TEST
#-------------------------------------------------------------
cm_nb <- confusionMatrix(nb_pred_class, test_nb$Label, positive = "1")
cat("\n=== Confusion Matrix Testing ===\n")
print(cm_nb$table)

tab <- cm_nb$table
TN  <- tab[1, 1]   # True Negative  (prediksi 0, aktual 0)
FN  <- tab[1, 2]   # False Negative (prediksi 0, aktual 1)
FP  <- tab[2, 1]   # False Positive (prediksi 1, aktual 0)
TP  <- tab[2, 2]   # True Positive  (prediksi 1, aktual 1)

#-------------------------------------------------------------
# 6. METRIK EVALUASI
#-------------------------------------------------------------
accuracy_nb <- (TP + TN) / sum(tab)

# Kelas 1 (Financial Distress) — kelas positif
precision_1_nb <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
recall_1_nb    <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
f1_1_nb        <- ifelse((precision_1_nb + recall_1_nb) == 0, 0,
                         2 * precision_1_nb * recall_1_nb /
                           (precision_1_nb + recall_1_nb))

# Kelas 0 (Non-Distress) — kelas negatif
precision_0_nb <- ifelse((TN + FN) == 0, 0, TN / (TN + FN))
recall_0_nb    <- ifelse((TN + FP) == 0, 0, TN / (TN + FP))
f1_0_nb        <- ifelse((precision_0_nb + recall_0_nb) == 0, 0,
                         2 * precision_0_nb * recall_0_nb /
                           (precision_0_nb + recall_0_nb))

#-------------------------------------------------------------
# 7. AUC-ROC
#-------------------------------------------------------------
roc_nb     <- roc(test_nb$Label, nb_pred_prob,
                  levels = c("0", "1"), direction = "<", quiet = TRUE)
auc_roc_nb <- as.numeric(auc(roc_nb))

#-------------------------------------------------------------
# 8. AUC-PR (Precision-Recall)
#-------------------------------------------------------------
pr_nb <- pr.curve(
  scores.class0 = nb_pred_prob[test_nb$Label == "1"],
  scores.class1 = nb_pred_prob[test_nb$Label == "0"],
  curve = TRUE
)
auc_pr_nb <- pr_nb$auc.integral

#-------------------------------------------------------------
# 9. OVERFITTING GAP
#-------------------------------------------------------------
test_acc_nb <- accuracy_nb
gap_nb      <- train_acc_nb - test_acc_nb

cat("\n=== Overfitting Check ===\n")
cat("Accuracy Training :", round(train_acc_nb, 4), "\n")
cat("Accuracy Testing  :", round(test_acc_nb,  4), "\n")
cat("Gap (Train - Test):", round(gap_nb,       4), "\n")
if (abs(gap_nb) <= 0.05) {
  cat("Status: Stabil (tidak ada indikasi overfitting)\n")
} else {
  cat("Status: Indikasi overfitting — perlu evaluasi lebih lanjut\n")
}

#-------------------------------------------------------------
# 10. TABEL RINGKASAN PERFORMA TESTING
#-------------------------------------------------------------
hasil_nb <- data.frame(
  Kelas     = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy  = round(c(accuracy_nb, accuracy_nb), 4),
  Precision = round(c(precision_0_nb, precision_1_nb), 4),
  Recall    = round(c(recall_0_nb,    recall_1_nb),    4),
  F1_Score  = round(c(f1_0_nb,        f1_1_nb),        4),
  AUC_PR    = round(c(auc_pr_nb, auc_pr_nb), 4),
  AUC_ROC   = round(c(auc_roc_nb, auc_roc_nb), 4)
)

cat("\n=== Tabel Performa Testing Naive Bayes ===\n")
print(kable(
  hasil_nb,
  caption = "Performa Testing Model Naive Bayes"
))

#-------------------------------------------------------------
# 11. ROC CURVE
#-------------------------------------------------------------
roc_df_nb <- data.frame(
  FPR = 1 - roc_nb$specificities,
  TPR = roc_nb$sensitivities
)

ggplot(roc_df_nb, aes(FPR, TPR)) +
  geom_line(linewidth = 1, color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  annotate("text", x = 0.7, y = 0.2,
           label = paste0("AUC = ", round(auc_roc_nb, 4)), size = 4) +
  labs(title = "ROC Curve — Naive Bayes (Semua Variabel)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal()

#-------------------------------------------------------------
# 12. PR CURVE
#-------------------------------------------------------------
pr_df_nb <- data.frame(
  Recall    = pr_nb$curve[, 1],
  Precision = pr_nb$curve[, 2]
)

ggplot(pr_df_nb, aes(Recall, Precision)) +
  geom_line(linewidth = 1, color = "red") +
  annotate("text", x = 0.3, y = 0.2,
           label = paste0("AUC-PR = ", round(auc_pr_nb, 4)), size = 4) +
  labs(title = "Precision-Recall Curve — Naive Bayes (Semua Variabel)",
       x = "Recall", y = "Precision") +
  theme_minimal()

#-------------------------------------------------------------
# 13. ACTUAL VS PREDICTED PLOT
#-------------------------------------------------------------
plot_nb <- data.frame(
  Index    = 1:nrow(test_nb),
  Aktual   = as.numeric(as.character(test_nb$Label)),
  Prediksi = as.numeric(as.character(nb_pred_class))
)

ggplot(plot_nb, aes(x = Index)) +
  
  # titik merah = aktual
  geom_point(aes(y = Aktual, color = "Actual"),
             size = 2, shape = 16) +
  
  # garis biru = prediksi
  geom_line(aes(y = Prediksi, color = "Predicted"),
            linewidth = 0.5) +
  
  scale_color_manual(
    values = c("Actual" = "red",
               "Predicted" = "blue")
  ) +
  
  scale_y_continuous(
    breaks = c(0, 1),
    limits = c(-0.05, 1.05)
  ) +
  
  labs(
    title = "Actual vs Predicted - Naive Bayes",
    x = "",
    y = "Label",
    color = "Keterangan"
  ) +
  
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "right"
  )
