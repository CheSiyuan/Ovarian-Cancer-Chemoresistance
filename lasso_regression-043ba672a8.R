# Load required libraries
library(glmnet)
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
y_train <- as.factor(train_data$label)

x_val <- as.matrix(val_data[, -which(colnames(val_data) == "label")])
y_val <- as.factor(val_data$label)

# Train Lasso regression model
lasso_model <- cv.glmnet(x_train, y_train, family = "binomial", alpha = 1)

# Get optimal lambda
best_lambda <- lasso_model$lambda.min

# Train model with optimal lambda
final_model <- glmnet(x_train, y_train, family = "binomial", alpha = 1, lambda = best_lambda)

# Predict on training and validation sets
prob_train <- predict(final_model, newx = x_train, type = "response")
prob_val <- predict(final_model, newx = x_val, type = "response")

# Calculate AUC and ROC
roc_train <- roc(as.numeric(y_train), as.numeric(prob_train))
roc_val <- roc(as.numeric(y_val), as.numeric(prob_val))

# Plot ROC curves
pdf("roc_curves.pdf")
plot(roc_train, col = "blue", main = "ROC Curves for Lasso Regression Model", lwd = 2)
plot(roc_val, col = "red", add = TRUE, lwd = 2)
# Add AUC value annotations
text(0.6, 0.4, paste("Training AUC:", round(auc(roc_train), 3)), col = "blue", cex = 0.8)
text(0.6, 0.35, paste("Validation AUC:", round(auc(roc_val), 3)), col = "red", cex = 0.8)
legend("bottomright", legend = c("Training Set", "Validation Set"), col = c("blue", "red"), lwd = 2)
dev.off()

# Save results to text file
results <- paste("Training AUC:", auc(roc_train), "\nValidation AUC:", auc(roc_val), "\nBest Lambda:", best_lambda)
write(results, file = "model_results.txt")

# Print results
cat(results, "\n")
cat("ROC curves saved as roc_curves.pdf\n")
cat("Model results saved as model_results.txt\n")