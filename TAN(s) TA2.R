library(infotheo)
library(igraph)
library(caret)
library(PRROC)
library(pROC)
library(ggplot2)
library(reshape2)
library(knitr)

set.seed(123)

#-------------------------------------------------------------
# 1. SIAPKAN DATA
#-------------------------------------------------------------
train_tan_sel <- train_sel
test_tan_sel  <- test_sel

train_tan_sel$Label <- factor(train_tan_sel$Label, levels = c(0, 1))
test_tan_sel$Label  <- factor(test_tan_sel$Label,  levels = c(0, 1))

vars_tan_sel <- setdiff(names(train_tan_sel), "Label")

cat("Variabel yang digunakan (", length(vars_tan_sel), " variabel):\n", sep = "")
print(vars_tan_sel)

#-------------------------------------------------------------
# 2. DISKRETISASI
#-------------------------------------------------------------
train_disc_sel <- train_tan_sel
test_disc_sel  <- test_tan_sel

for (v in vars_tan_sel) {
  q <- quantile(train_tan_sel[[v]], probs = c(0, 1/3, 2/3, 1), na.rm = TRUE)
  q <- unique(q)
  
  if (length(q) < 3) {
    q <- quantile(train_tan_sel[[v]], probs = c(0, 0.5, 1), na.rm = TRUE)
    q <- unique(q)
  }
  
  breaks_v <- c(-Inf, q[-c(1, length(q))], Inf)
  
  train_disc_sel[[v]] <- cut(train_tan_sel[[v]], breaks = breaks_v,
                             labels = FALSE, include.lowest = TRUE)
  test_disc_sel[[v]]  <- cut(test_tan_sel[[v]], breaks = breaks_v,
                             labels = FALSE, include.lowest = TRUE)
}

train_disc_sel <- data.frame(lapply(train_disc_sel, as.factor))
test_disc_sel  <- data.frame(lapply(test_disc_sel, as.factor))

#-------------------------------------------------------------
# 3. HEATMAP KORELASI PEARSON
#-------------------------------------------------------------
cor_matrix_sel <- cor(
  train_tan_sel[, vars_tan_sel],
  use = "pairwise.complete.obs"
)

cor_melt_sel <- melt(cor_matrix_sel)

ggplot(cor_melt_sel, aes(x = Var1, y = Var2, fill = value)) +
  geom_tile() +
  scale_fill_gradient2(
    low = "blue",
    high = "red",
    mid = "white",
    midpoint = 0,
    limits = c(-1, 1)
  ) +
  theme_minimal() +
  coord_fixed() +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, size = 8),
    axis.text.y = element_text(size = 8),
    plot.title = element_text(hjust = 0.5)
  ) +
  labs(
    title = "Heatmap Korelasi Seleksi Variabel",
    x = "",
    y = "",
    fill = "Korelasi"
  )

#-------------------------------------------------------------
# 4. CONDITIONAL MUTUAL INFORMATION (CMI)
#-------------------------------------------------------------
p_sel <- length(vars_tan_sel)

mi_matrix_sel <- matrix(
  0, p_sel, p_sel,
  dimnames = list(vars_tan_sel, vars_tan_sel)
)

for (i in 1:p_sel) {
  for (j in 1:p_sel) {
    if (i != j) {
      mi_matrix_sel[i, j] <- condinformation(
        train_disc_sel[[vars_tan_sel[i]]],
        train_disc_sel[[vars_tan_sel[j]]],
        train_disc_sel$Label
      )
    }
  }
}

#-------------------------------------------------------------
# 5. MAXIMUM SPANNING TREE + DIRECTED TREE
#-------------------------------------------------------------
g_tan_sel <- graph_from_adjacency_matrix(
  mi_matrix_sel,
  mode = "undirected",
  weighted = TRUE,
  diag = FALSE
)

mst_tan_sel <- mst(g_tan_sel, weights = E(g_tan_sel)$weight)

plot(mst_tan_sel,
     layout = layout_with_fr(mst_tan_sel),
     vertex.size = 22,
     vertex.color = "skyblue",
     vertex.label.cex = 0.8,
     edge.color = "gray40",
     main = "Maximum Spanning Tree — TAN (Seleksi Variabel)")

root_sel  <- vars_tan_sel[which.max(rowSums(mi_matrix_sel))]
bfs_sel   <- bfs(mst_tan_sel, root = root_sel, dist = TRUE)
edges_sel <- as_edgelist(mst_tan_sel)

directed_edges_sel <- data.frame(
  From = character(),
  To   = character(),
  stringsAsFactors = FALSE
)

for (i in 1:nrow(edges_sel)) {
  a  <- edges_sel[i,1]
  b  <- edges_sel[i,2]
  da <- bfs_sel$dist[which(vars_tan_sel == a)]
  db <- bfs_sel$dist[which(vars_tan_sel == b)]
  
  if (da < db) {
    directed_edges_sel <- rbind(directed_edges_sel, data.frame(From = a, To = b))
  } else {
    directed_edges_sel <- rbind(directed_edges_sel, data.frame(From = b, To = a))
  }
}

g_dir_sel <- graph_from_data_frame(directed_edges_sel, directed = TRUE)

plot(g_dir_sel,
     layout = layout_as_tree(g_dir_sel, root = root_sel),
     vertex.size = 22,
     vertex.color = "lightgreen",
     vertex.label.cex = 0.8,
     edge.arrow.size = 0.5,
     main = "Struktur TAN Directed Tree (Seleksi Variabel)")

#-------------------------------------------------------------
# 6. FUNGSI PREDIKSI TAN
#-------------------------------------------------------------
predict_tan_fn <- function(test_df, train_df, vars, directed_edges) {
  classes  <- levels(train_df$Label)
  prior    <- prop.table(table(train_df$Label))
  pred_out <- rep(NA, nrow(test_df))
  prob_out <- rep(NA, nrow(test_df))
  
  for (i in 1:nrow(test_df)) {
    obs <- test_df[i, ]
    log_prob <- numeric(length(classes))
    
    for (ci in seq_along(classes)) {
      cl <- classes[ci]
      lp <- log(as.numeric(prior[cl]))
      
      for (v in vars) {
        parent <- directed_edges$From[directed_edges$To == v]
        
        if (length(parent) == 0) {
          num <- sum(train_df[[v]] == obs[[v]] &
                       train_df$Label == cl) + 1
          den <- sum(train_df$Label == cl) +
            length(unique(train_df[[v]]))
        } else {
          num <- sum(train_df[[v]] == obs[[v]] &
                       train_df[[parent]] == obs[[parent]] &
                       train_df$Label == cl) + 1
          den <- sum(train_df[[parent]] == obs[[parent]] &
                       train_df$Label == cl) +
            length(unique(train_df[[v]]))
        }
        
        lp <- lp + log(num / den)
      }
      
      log_prob[ci] <- lp
    }
    
    pexp <- exp(log_prob - max(log_prob))
    pexp <- pexp / sum(pexp)
    
    pred_out[i] <- classes[which.max(pexp)]
    prob_out[i] <- pexp[which(classes == "1")]
  }
  
  list(
    pred = factor(pred_out, levels = c("0", "1")),
    prob = prob_out
  )
}

#-------------------------------------------------------------
# 7A. PREDIKSI DATA TRAINING
#-------------------------------------------------------------
tan_train_res_sel <- predict_tan_fn(
  train_disc_sel,
  train_disc_sel,
  vars_tan_sel,
  directed_edges_sel
)

cm_train_tan_sel <- confusionMatrix(
  tan_train_res_sel$pred,
  train_disc_sel$Label,
  positive = "1"
)

train_acc_tan_sel <- as.numeric(cm_train_tan_sel$overall["Accuracy"])

cat("=== Confusion Matrix Training ===\n")
print(cm_train_tan_sel$table)
cat("Accuracy Training:", round(train_acc_tan_sel, 4), "\n")

#-------------------------------------------------------------
# 7B. METRIK EVALUASI DATA TRAINING
#-------------------------------------------------------------
tab_train <- cm_train_tan_sel$table
TN_train  <- tab_train[1,1]
FN_train  <- tab_train[1,2]
FP_train  <- tab_train[2,1]
TP_train  <- tab_train[2,2]

accuracy_train_tan_sel <- (TP_train + TN_train) / sum(tab_train)

precision_1_train_tan_sel <- TP_train / (TP_train + FP_train)
recall_1_train_tan_sel    <- TP_train / (TP_train + FN_train)
f1_1_train_tan_sel <- 2 * precision_1_train_tan_sel *
  recall_1_train_tan_sel /
  (precision_1_train_tan_sel + recall_1_train_tan_sel)

precision_0_train_tan_sel <- TN_train / (TN_train + FN_train)
recall_0_train_tan_sel    <- TN_train / (TN_train + FP_train)
f1_0_train_tan_sel <- 2 * precision_0_train_tan_sel *
  recall_0_train_tan_sel /
  (precision_0_train_tan_sel + recall_0_train_tan_sel)

roc_train_tan_sel <- roc(
  train_disc_sel$Label,
  tan_train_res_sel$prob,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

auc_roc_train_tan_sel <- as.numeric(auc(roc_train_tan_sel))

pr_train_tan_sel <- pr.curve(
  scores.class0 = tan_train_res_sel$prob[train_disc_sel$Label == "1"],
  scores.class1 = tan_train_res_sel$prob[train_disc_sel$Label == "0"],
  curve = TRUE
)

auc_pr_train_tan_sel <- pr_train_tan_sel$auc.integral

hasil_train_tan_sel <- data.frame(
  Kelas = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy = round(rep(accuracy_train_tan_sel, 2), 4),
  Precision = round(c(precision_0_train_tan_sel, precision_1_train_tan_sel), 4),
  Recall = round(c(recall_0_train_tan_sel, recall_1_train_tan_sel), 4),
  F1_Score = round(c(f1_0_train_tan_sel, f1_1_train_tan_sel), 4),
  AUC_PR = round(rep(auc_pr_train_tan_sel, 2), 4),
  AUC_ROC = round(rep(auc_roc_train_tan_sel, 2), 4)
)

print(kable(hasil_train_tan_sel,
            caption = "Performa Training Model TAN (Seleksi Variabel)"))

#-------------------------------------------------------------
# 8. PREDIKSI DATA TESTING
#-------------------------------------------------------------
tan_test_res_sel <- predict_tan_fn(
  test_disc_sel,
  train_disc_sel,
  vars_tan_sel,
  directed_edges_sel
)

tan_pred_sel <- tan_test_res_sel$pred
tan_prob_sel <- tan_test_res_sel$prob

#-------------------------------------------------------------
# 9. CONFUSION MATRIX TEST
#-------------------------------------------------------------
cm_tan_sel <- confusionMatrix(
  tan_pred_sel,
  test_disc_sel$Label,
  positive = "1"
)

cat("\n=== Confusion Matrix Testing ===\n")
print(cm_tan_sel$table)

tab <- cm_tan_sel$table
TN  <- tab[1,1]
FN  <- tab[1,2]
FP  <- tab[2,1]
TP  <- tab[2,2]

#-------------------------------------------------------------
# 10. METRIK EVALUASI
#-------------------------------------------------------------
accuracy_tan_sel <- (TP + TN) / sum(tab)

precision_1_tan_sel <- TP / (TP + FP)
recall_1_tan_sel    <- TP / (TP + FN)
f1_1_tan_sel <- 2 * precision_1_tan_sel *
  recall_1_tan_sel /
  (precision_1_tan_sel + recall_1_tan_sel)

precision_0_tan_sel <- TN / (TN + FN)
recall_0_tan_sel    <- TN / (TN + FP)
f1_0_tan_sel <- 2 * precision_0_tan_sel *
  recall_0_tan_sel /
  (precision_0_tan_sel + recall_0_tan_sel)

#-------------------------------------------------------------
# 11. AUC-ROC
#-------------------------------------------------------------
roc_tan_sel <- roc(
  test_disc_sel$Label,
  tan_prob_sel,
  levels = c("0", "1"),
  direction = "<",
  quiet = TRUE
)

auc_roc_tan_sel <- as.numeric(auc(roc_tan_sel))

#-------------------------------------------------------------
# 12. AUC-PR
#-------------------------------------------------------------
pr_tan_sel <- pr.curve(
  scores.class0 = tan_prob_sel[test_disc_sel$Label == "1"],
  scores.class1 = tan_prob_sel[test_disc_sel$Label == "0"],
  curve = TRUE
)

auc_pr_tan_sel <- pr_tan_sel$auc.integral

#-------------------------------------------------------------
# 13. OVERFITTING GAP
#-------------------------------------------------------------
gap_tan_sel <- train_acc_tan_sel - accuracy_tan_sel

cat("\n=== Overfitting Check ===\n")
cat("Accuracy Training :", round(train_acc_tan_sel, 4), "\n")
cat("Accuracy Testing  :", round(accuracy_tan_sel, 4), "\n")
cat("Gap (Train - Test):", round(gap_tan_sel, 4), "\n")

#-------------------------------------------------------------
# 14. TABEL RINGKASAN PERFORMA
#-------------------------------------------------------------
hasil_tan_sel <- data.frame(
  Kelas = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy = round(rep(accuracy_tan_sel, 2), 4),
  Precision = round(c(precision_0_tan_sel, precision_1_tan_sel), 4),
  Recall = round(c(recall_0_tan_sel, recall_1_tan_sel), 4),
  F1_Score = round(c(f1_0_tan_sel, f1_1_tan_sel), 4),
  AUC_PR = round(rep(auc_pr_tan_sel, 2), 4),
  AUC_ROC = round(rep(auc_roc_tan_sel, 2), 4)
)

print(kable(hasil_tan_sel,
            caption = "Performa Model TAN (Seleksi Variabel)"))

#-------------------------------------------------------------
# 15. ROC CURVE
#-------------------------------------------------------------
roc_df_tan_sel <- data.frame(
  FPR = 1 - roc_tan_sel$specificities,
  TPR = roc_tan_sel$sensitivities
)

ggplot(roc_df_tan_sel, aes(FPR, TPR)) +
  geom_line(linewidth = 1, color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  annotate("text", x = 0.7, y = 0.2,
           label = paste0("AUC = ", round(auc_roc_tan_sel, 4)), size = 4) +
  labs(title = "ROC Curve — TAN (Seleksi Variabel)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal()

#-------------------------------------------------------------
# 16. PR CURVE
#-------------------------------------------------------------
pr_df_tan_sel <- data.frame(
  Recall    = pr_tan_sel$curve[, 1],
  Precision = pr_tan_sel$curve[, 2]
)

ggplot(pr_df_tan_sel, aes(Recall, Precision)) +
  geom_line(linewidth = 1, color = "red") +
  annotate("text", x = 0.3, y = 0.2,
           label = paste0("AUC-PR = ", round(auc_pr_tan_sel, 4)), size = 4) +
  labs(title = "Precision-Recall Curve — TAN (Seleksi Variabel)",
       x = "Recall", y = "Precision") +
  theme_minimal()

#-------------------------------------------------------------
# 17. ACTUAL VS PREDICTED PLOT
#-------------------------------------------------------------
plot_tan_sel <- data.frame(
  Index    = 1:nrow(test_disc_sel),
  Aktual   = as.numeric(as.character(test_disc_sel$Label)),
  Prediksi = as.numeric(as.character(tan_pred_sel))
)

ggplot(plot_tan_sel, aes(x = Index)) +
  
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
    title = "Actual vs Predicted - TAN (Seleksi Variabel)",
    x = "",
    y = "Label",
    color = "Keterangan"
  ) +
  
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "right"
  )
