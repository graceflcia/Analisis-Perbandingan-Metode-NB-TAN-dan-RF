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
train_tan <- train_data
test_tan  <- test_data

train_tan$Label <- factor(train_tan$Label, levels = c(0, 1))
test_tan$Label  <- factor(test_tan$Label,  levels = c(0, 1))

vars_tan <- setdiff(names(train_tan), "Label")

#-------------------------------------------------------------
# 2. DISKRETISASI
#-------------------------------------------------------------
train_disc <- train_tan
test_disc  <- test_tan

for (v in vars_tan) {
  q <- quantile(train_tan[[v]], probs = c(0, 1/3, 2/3, 1), na.rm = TRUE)
  q <- unique(q)
  
  # Fallback ke median jika semua nilai sama (q tidak cukup unik)
  if (length(q) < 3) {
    q <- quantile(train_tan[[v]], probs = c(0, 0.5, 1), na.rm = TRUE)
    q <- unique(q)
  }
  
  breaks_v <- c(-Inf, q[-c(1, length(q))], Inf)
  
  train_disc[[v]] <- cut(train_tan[[v]], breaks = breaks_v,
                         labels = FALSE, include.lowest = TRUE)
  test_disc[[v]]  <- cut(test_tan[[v]],  breaks = breaks_v,
                         labels = FALSE, include.lowest = TRUE)
}

# Ubah semua kolom menjadi factor (dibutuhkan oleh infotheo & fungsi prediksi)
train_disc <- data.frame(lapply(train_disc, as.factor))
test_disc  <- data.frame(lapply(test_disc,  as.factor))

#-------------------------------------------------------------
# 3. HEATMAP KORELASI PEARSON
#-------------------------------------------------------------
cor_matrix <- cor(train_tan[, vars_tan], use = "pairwise.complete.obs")
cor_melt   <- melt(cor_matrix)

ggplot(cor_melt, aes(x = Var1, y = Var2, fill = value)) +
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
    axis.text.x = element_text(angle = 45, hjust = 1, size = 7),
    axis.text.y = element_text(size = 7),
    plot.title = element_text(hjust = 0.5) 
  ) +
  labs(
    title = "Heatmap Korelasi Antar Variabel",
    x = "",
    y = "",
    fill = "Korelasi"
  )

#-------------------------------------------------------------
# 4. CONDITIONAL MUTUAL INFORMATION (CMI)
#-------------------------------------------------------------
p_tan     <- length(vars_tan)
mi_matrix <- matrix(0, p_tan, p_tan,
                    dimnames = list(vars_tan, vars_tan))

for (i in 1:p_tan) {
  for (j in 1:p_tan) {
    if (i != j) {
      mi_matrix[i, j] <- condinformation(
        train_disc[[vars_tan[i]]],
        train_disc[[vars_tan[j]]],
        train_disc$Label
      )
    }
  }
}

#-------------------------------------------------------------
# 5. MAXIMUM SPANNING TREE (MST) + DIRECTED TREE (BFS)
#-------------------------------------------------------------
g_tan   <- graph_from_adjacency_matrix(mi_matrix, mode = "undirected",
                                       weighted = TRUE, diag = FALSE)
mst_tan <- mst(g_tan, weights = E(g_tan)$weight)
par(mar=c(1,1,2,1))
plot(mst_tan, layout = layout_with_fr(mst_tan),
     vertex.size = 22, vertex.color = "skyblue",
     vertex.label.cex = 0.7, edge.color = "gray40",
     main = "Maximum Spanning Tree — TAN (Semua Variabel)")

root_tan <- vars_tan[which.max(rowSums(mi_matrix))]
bfs_tan  <- bfs(mst_tan, root = root_tan, dist = TRUE)
edges_tan <- as_edgelist(mst_tan)

# Arahkan setiap edge dari node yang lebih dekat ke root
directed_edges_tan <- data.frame(From = character(), To = character(),
                                 stringsAsFactors = FALSE)
for (i in 1:nrow(edges_tan)) {
  a  <- edges_tan[i, 1]; b <- edges_tan[i, 2]
  da <- bfs_tan$dist[which(vars_tan == a)]
  db <- bfs_tan$dist[which(vars_tan == b)]
  if (da < db) {
    directed_edges_tan <- rbind(directed_edges_tan, data.frame(From = a, To = b))
  } else {
    directed_edges_tan <- rbind(directed_edges_tan, data.frame(From = b, To = a))
  }
}

g_dir_tan <- graph_from_data_frame(directed_edges_tan, directed = TRUE)
plot(g_dir_tan, layout = layout_as_tree(g_dir_tan, root = root_tan),
     vertex.size = 22, vertex.color = "lightgreen",
     vertex.label.cex = 0.7, edge.arrow.size = 0.5,
     main = "Struktur TAN Directed Tree (Semua Variabel)")

#-------------------------------------------------------------
# 6. FUNGSI PREDIKSI TAN
#-------------------------------------------------------------
predict_tan_fn <- function(test_df, train_df, vars, directed_edges) {
  classes  <- levels(train_df$Label)
  prior    <- prop.table(table(train_df$Label))
  pred_out <- rep(NA, nrow(test_df))
  prob_out <- rep(NA, nrow(test_df))
  
  for (i in 1:nrow(test_df)) {
    obs      <- test_df[i, ]
    log_prob <- numeric(length(classes))
    
    for (ci in seq_along(classes)) {
      cl <- classes[ci]
      lp <- log(as.numeric(prior[cl]))
      
      for (v in vars) {
        parent <- directed_edges$From[directed_edges$To == v]
        
        if (length(parent) == 0) {
          # Root node: P(Xi | Y) — conditional tanpa parent variabel
          num <- sum(train_df[[v]] == obs[[v]] & train_df$Label == cl) + 1
          den <- sum(train_df$Label == cl) + length(unique(train_df[[v]]))
        } else {
          # Non-root: P(Xi | Xparent, Y)
          num <- sum(train_df[[v]] == obs[[v]] &
                       train_df[[parent]] == obs[[parent]] &
                       train_df$Label == cl) + 1
          den <- sum(train_df[[parent]] == obs[[parent]] &
                       train_df$Label == cl) + length(unique(train_df[[v]]))
        }
        lp <- lp + log(num / den)
      }
      log_prob[ci] <- lp
    }
    
    # Normalisasi dengan log-sum-exp untuk stabilitas numerik
    pexp        <- exp(log_prob - max(log_prob))
    pexp        <- pexp / sum(pexp)
    pred_out[i] <- classes[which.max(pexp)]
    prob_out[i] <- pexp[which(classes == "1")]
  }
  
  list(pred = factor(pred_out, levels = c("0", "1")), prob = prob_out)
}

#-------------------------------------------------------------
# 7A. PREDIKSI DATA TRAINING
#-------------------------------------------------------------
tan_train_res <- predict_tan_fn(train_disc, train_disc, vars_tan, directed_edges_tan)
cm_train_tan  <- confusionMatrix(tan_train_res$pred, train_disc$Label, positive = "1")
train_acc_tan <- as.numeric(cm_train_tan$overall["Accuracy"])

cat("=== Confusion Matrix Training ===\n")
print(cm_train_tan$table)
cat("Accuracy Training:", round(train_acc_tan, 4), "\n")

#-------------------------------------------------------------
# 7B. TABEL RINGKASAN PERFORMA TRAINING
#-------------------------------------------------------------
hasil_train_tan <- data.frame(
  Kelas     = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy  = round(c(accuracy_train_tan, accuracy_train_tan), 4),
  Precision = round(c(precision_0_train_tan, precision_1_train_tan), 4),
  Recall    = round(c(recall_0_train_tan,    recall_1_train_tan),    4),
  F1_Score  = round(c(f1_0_train_tan,        f1_1_train_tan),        4),
  AUC_PR    = round(c(auc_pr_train_tan, auc_pr_train_tan), 4),
  AUC_ROC   = round(c(auc_roc_train_tan, auc_roc_train_tan), 4)
)

cat("\n=== Tabel Performa Training TAN ===\n")
print(kable(
  hasil_train_tan,
  caption = "Performa Training Model TAN"
))

#-------------------------------------------------------------
# 8. PREDIKSI DATA TESTING
#-------------------------------------------------------------
tan_test_res <- predict_tan_fn(test_disc, train_disc, vars_tan, directed_edges_tan)
tan_pred     <- tan_test_res$pred
tan_prob     <- tan_test_res$prob

#-------------------------------------------------------------
# 9. CONFUSION MATRIX TEST
#-------------------------------------------------------------
cm_tan <- confusionMatrix(tan_pred, test_disc$Label, positive = "1")
cat("\n=== Confusion Matrix Testing ===\n")
print(cm_tan$table)

tab <- cm_tan$table
TN  <- tab[1, 1]
FN  <- tab[1, 2]
FP  <- tab[2, 1]
TP  <- tab[2, 2]

#-------------------------------------------------------------
# 10. METRIK EVALUASI
#-------------------------------------------------------------
accuracy_tan <- (TP + TN) / sum(tab)

# Kelas 1 (Distress)
precision_1_tan <- ifelse((TP + FP) == 0, 0, TP / (TP + FP))
recall_1_tan    <- ifelse((TP + FN) == 0, 0, TP / (TP + FN))
f1_1_tan        <- ifelse((precision_1_tan + recall_1_tan) == 0, 0,
                          2 * precision_1_tan * recall_1_tan /
                            (precision_1_tan + recall_1_tan))

# Kelas 0 (Non-Distress)
precision_0_tan <- ifelse((TN + FN) == 0, 0, TN / (TN + FN))
recall_0_tan    <- ifelse((TN + FP) == 0, 0, TN / (TN + FP))
f1_0_tan        <- ifelse((precision_0_tan + recall_0_tan) == 0, 0,
                          2 * precision_0_tan * recall_0_tan /
                            (precision_0_tan + recall_0_tan))

#-------------------------------------------------------------
# 11. AUC-ROC
#-------------------------------------------------------------
roc_tan     <- roc(test_disc$Label, tan_prob,
                   levels = c("0", "1"), direction = "<", quiet = TRUE)
auc_roc_tan <- as.numeric(auc(roc_tan))

#-------------------------------------------------------------
# 12. AUC-PR
#-------------------------------------------------------------
pr_tan <- pr.curve(
  scores.class0 = tan_prob[test_disc$Label == "1"],
  scores.class1 = tan_prob[test_disc$Label == "0"],
  curve = TRUE
)
auc_pr_tan <- pr_tan$auc.integral

#-------------------------------------------------------------
# 13. OVERFITTING GAP
#-------------------------------------------------------------
gap_tan <- train_acc_tan - accuracy_tan

cat("\n=== Overfitting Check ===\n")
cat("Accuracy Training :", round(train_acc_tan, 4), "\n")
cat("Accuracy Testing  :", round(accuracy_tan,  4), "\n")
cat("Gap (Train - Test):", round(gap_tan,        4), "\n")
if (abs(gap_tan) <= 0.05) {
  cat("Status: Stabil\n")
} else {
  cat("Status: Indikasi overfitting\n")
}

#-------------------------------------------------------------
# 14. TABEL RINGKASAN PERFORMA
#-------------------------------------------------------------
hasil_tan <- data.frame(
  Kelas     = c("0 (Non-Distress)", "1 (Financial Distress)"),
  Accuracy  = round(c(accuracy_tan, accuracy_tan), 4),
  Precision = round(c(precision_0_tan, precision_1_tan), 4),
  Recall    = round(c(recall_0_tan,    recall_1_tan),    4),
  F1_Score  = round(c(f1_0_tan,        f1_1_tan),        4),
  AUC_PR    = round(c(auc_pr_tan, auc_pr_tan), 4),
  AUC_ROC   = round(c(auc_roc_tan, auc_roc_tan), 4)
)

cat("\n=== Tabel Performa TAN (Semua Variabel) ===\n")
print(kable(
  hasil_tan,
  caption = "Performa Model TAN (Semua Variabel)"
))

#-------------------------------------------------------------
# 15. ROC CURVE
#-------------------------------------------------------------
roc_df_tan <- data.frame(
  FPR = 1 - roc_tan$specificities,
  TPR = roc_tan$sensitivities
)

ggplot(roc_df_tan, aes(FPR, TPR)) +
  geom_line(linewidth = 1, color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray") +
  annotate("text", x = 0.7, y = 0.2,
           label = paste0("AUC = ", round(auc_roc_tan, 4)), size = 4) +
  labs(title = "ROC Curve — TAN (Semua Variabel)",
       x = "False Positive Rate (1 - Specificity)",
       y = "True Positive Rate (Sensitivity)") +
  theme_minimal()

#-------------------------------------------------------------
# 16. PR CURVE
#-------------------------------------------------------------
pr_df_tan <- data.frame(
  Recall    = pr_tan$curve[, 1],
  Precision = pr_tan$curve[, 2]
)

ggplot(pr_df_tan, aes(Recall, Precision)) +
  geom_line(linewidth = 1, color = "red") +
  annotate("text", x = 0.3, y = 0.2,
           label = paste0("AUC-PR = ", round(auc_pr_tan, 4)), size = 4) +
  labs(title = "Precision-Recall Curve — TAN (Semua Variabel)",
       x = "Recall", y = "Precision") +
  theme_minimal()

#-------------------------------------------------------------
# 17. ACTUAL VS PREDICTED PLOT
#-------------------------------------------------------------
plot_tan <- data.frame(
  Index    = 1:nrow(test_disc),
  Aktual   = as.numeric(as.character(test_disc$Label)),
  Prediksi = as.numeric(as.character(tan_pred))
)

ggplot(plot_tan, aes(x = Index)) +
  
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
    title = "Actual vs Predicted - TAN (Semua Variabel)",
    x = "",
    y = "Label",
    color = "Keterangan"
  ) +
  
  theme_minimal() +
  theme(
    plot.title = element_text(hjust = 0.5),
    legend.position = "right"
  )
