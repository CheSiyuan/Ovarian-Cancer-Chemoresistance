# Load required libraries
library(randomForest)
library(pROC)
library(caret)
library(ggplot2)

# Read the data
resistant_data <- read.csv("D:\\csy\\301\\Raverage.csv")
sensitive_data <- read.csv("D:\\csy\\301\\Saverage.csv")

# Add labels: 1 for resistant, 0 for sensitive
resistant_data$label <- as.factor(1)
sensitive_data$label <- as.factor(0)

# Combine the data
bind_data <- rbind(resistant_data, sensitive_data)

# Remove ID column as it's not a feature
bind_data <- bind_data[, -which(colnames(bind_data) == "ID")]

# Split data into training and validation sets (7:3)
set.seed(123)
train_index <- createDataPartition(bind_data$label, p = 0.7, list = FALSE)
train_data <- bind_data[train_index, ]
val_data <- bind_data[-train_index, ]

# Train Random Forest model
randomforest_model <- randomForest(label ~ ., data = train_data, ntree = 100, importance = TRUE)

# Predict on training and validation sets with probabilities
prob_train <- predict(randomforest_model, newdata = train_data, type = "prob")
prob_val <- predict(randomforest_model, newdata = val_data, type = "prob")

# Extract probabilities for class 1
prob_train_class1 <- prob_train[, 2]
prob_val_class1 <- prob_val[, 2]

# Convert labels to numeric for ROC calculation
y_train_numeric <- as.numeric(levels(train_data$label))[train_data$label]
y_val_numeric <- as.numeric(levels(val_data$label))[val_data$label]

# Calculate AUC and ROC
roc_train <- roc(y_train_numeric, prob_train_class1)
roc_val <- roc(y_val_numeric, prob_val_class1)

# Plot ROC curves with AUC annotations
pdf("roc_curves_randomforest.pdf")
plot(roc_train, col = "blue", main = "ROC Curves for Random Forest Model", lwd = 2)
plot(roc_val, col = "red", add = TRUE, lwd = 2)
# Add AUC value annotations
text(0.6, 0.4, paste("Training AUC:", round(auc(roc_train), 3)), col = "blue", cex = 0.8)
text(0.6, 0.35, paste("Validation AUC:", round(auc(roc_val), 3)), col = "red", cex = 0.8)
legend("bottomright", legend = c("Training Set", "Validation Set"), col = c("blue", "red"), lwd = 2)
dev.off()

# Save results to text file
results <- paste("Training AUC:", auc(roc_train), "\nValidation AUC:", auc(roc_val), "\nNumber of trees:", 100)
write(results, file = "randomforest_model_results.txt")

# Print results
cat(results, "\n")
cat("ROC curves saved as roc_curves_randomforest.pdf\n")
cat("Model results saved as randomforest_model_results.txt\n")