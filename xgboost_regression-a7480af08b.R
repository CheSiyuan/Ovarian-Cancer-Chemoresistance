# Load required libraries
library(xgboost)
library(pROC)
library(caret)
library(ggplot2)

# Read the data
resistant_data <- read.csv("D:\\csy\\301\\Raverage.csv")
sensitive_data <- read.csv("D:\\csy\\301\\Saverage.csv")

# Add labels: 1 for resistant, 0 for sensitive
resistant_data$label <- 1
sensitive_data$label <- 0

# Combine the data
bind_data <- rbind(resistant_data, sensitive_data)

# Remove ID column as it's not a feature
bind_data <- bind_data[, -which(colnames(bind_data) == "ID")]

# Split data into training and validation sets (7:3)
set.seed(123)
train_index <- createDataPartition(bind_data$label, p = 0.7, list = FALSE)
train_data <- bind_data[train_index, ]
val_data <- bind_data[-train_index, ]

# Separate features and labels
x_train <- as.matrix(train_data[, -which(colnames(train_data) == "label")])
y_train <- train_data$label

x_val <- as.matrix(val_data[, -which(colnames(val_data) == "label")])
y_val <- val_data$label

# Create xgboost DMatrix objects
dtrain <- xgb.DMatrix(data = x_train, label = y_train)
dval <- xgb.DMatrix(data = x_val, label = y_val)

# Set xgboost parameters
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  max_depth = 6,
  eta = 0.3,
  gamma = 0,
  subsample = 1,
  colsample_bytree = 1,
  min_child_weight = 1
)

# Train xgboost model
watchlist <- list(train = dtrain, eval = dval)
xgboost_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 100,
  watchlist = watchlist,
  print_every_n = 10,
  early_stopping_rounds = 10
)

# Predict on training and validation sets
prob_train <- predict(xgboost_model, dtrain)
prob_val <- predict(xgboost_model, dval)

# Calculate AUC and ROC
roc_train <- roc(y_train, prob_train)
roc_val <- roc(y_val, prob_val)

# Plot ROC curves with AUC annotations
pdf("roc_curves_xgboost.pdf")
plot(roc_train, col = "blue", main = "ROC Curves for XGBoost Model", lwd = 2)
plot(roc_val, col = "red", add = TRUE, lwd = 2)
# Add AUC value annotations
text(0.6, 0.4, paste("Training AUC:", round(auc(roc_train), 3)), col = "blue", cex = 0.8)
text(0.6, 0.35, paste("Validation AUC:", round(auc(roc_val), 3)), col = "red", cex = 0.8)
legend("bottomright", legend = c("Training Set", "Validation Set"), col = c("blue", "red"), lwd = 2)
dev.off()

# Save results to text file
results <- paste("Training AUC:", auc(roc_train), "\nValidation AUC:", auc(roc_val), "\nBest iteration:", xgboost_model$best_iteration, "\nBest score:", xgboost_model$best_score)
write(results, file = "xgboost_model_results.txt")

# Print results
cat(results, "\n")
cat("ROC curves saved as roc_curves_xgboost.pdf\n")
cat("Model results saved as xgboost_model_results.txt\n")