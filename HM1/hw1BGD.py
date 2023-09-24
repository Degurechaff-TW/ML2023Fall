import pandas as pd
import numpy as np

# Load data
train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

# Data preprocessing: Removing outliers
train_data = train_data[train_data['PM2.5'] <= 100]

# Feature selection based on correlation analysis with threshold of 0.5
correlations = train_data.corr()['PM2.5'].abs()
selected_features_with_pm25 = correlations[correlations > 0.5].index.tolist()

# Extracting the selected features including PM2.5 from the training data
X_train_data_with_pm25 = train_data[selected_features_with_pm25].values

# Preparing training data
train_X_with_pm25 = []
train_y_with_pm25 = []
for i in range(len(X_train_data_with_pm25) - 8):
    train_X_with_pm25.append(X_train_data_with_pm25[i:i+8].flatten())
    train_y_with_pm25.append(X_train_data_with_pm25[i+8, -1])

train_X_with_pm25 = np.array(train_X_with_pm25)
train_y_with_pm25 = np.array(train_y_with_pm25)

# Data normalization
mean_train_with_pm25 = np.mean(train_X_with_pm25, axis=0)
std_train_with_pm25 = np.std(train_X_with_pm25, axis=0)

# Splitting the data into training and validation sets
split_ratio = 0.8
split_index = int(split_ratio * len(train_X_with_pm25))

X_train_split = train_X_with_pm25[:split_index]
y_train_split = train_y_with_pm25[:split_index]
X_val_split = train_X_with_pm25[split_index:]
y_val_split = train_y_with_pm25[split_index:]

# Normalizing the split data using the training data's mean and std
X_train_normalized = (X_train_split - mean_train_with_pm25) / (std_train_with_pm25 + 1e-8)
X_val_normalized = (X_val_split - mean_train_with_pm25) / (std_train_with_pm25 + 1e-8)

# Implementing Batch Gradient Descent for linear regression
learning_rate = 0.001
num_epochs = 5000

# Initialize weights and bias
W_batch = np.zeros(X_train_normalized.shape[1])
b_batch = 0
mse_list = []
for epoch in range(num_epochs):
    # Training prediction
    y_train_pred = np.dot(X_train_normalized, W_batch) + b_batch
    error_train = y_train_pred - y_train_split
    mse = np.mean(error_train**2)
    mse_list.append(mse)
    gradient_W = (2/len(X_train_normalized)) * np.dot(X_train_normalized.T, error_train)
    gradient_b = (2/len(X_train_normalized)) * error_train.sum()
    
    # Update weights and bias
    W_batch -= learning_rate * gradient_W
    b_batch -= learning_rate * gradient_b
min_mse_value = min(mse_list)
min_mse_epoch = mse_list.index(min_mse_value)
print(min_mse_value, min_mse_epoch)
# After training, predict on the validation set
y_val_pred_batch = np.dot(X_val_normalized, W_batch) + b_batch
error_val_batch = y_val_pred_batch - y_val_split
mse_val_batch = np.mean(error_val_batch ** 2)
print(mse_val_batch)

# Preparing the test data
X_test_data_with_pm25 = test_data[selected_features_with_pm25].values

test_X_pm25 = []
for i in range(0, len(X_test_data_with_pm25) - 7, 8):
    test_X_pm25.append(X_test_data_with_pm25[i:i+8].flatten())

test_X_pm25 = np.array(test_X_pm25)

# Normalize the test data using the mean and std from the training data
test_X_normalized_pm25 = (test_X_pm25 - mean_train_with_pm25) / (std_train_with_pm25 + 1e-8)

# Predicting the PM2.5 values for the test data
y_test_pred_batch = np.dot(test_X_normalized_pm25, W_batch) + b_batch

# Output the results to a CSV file
sample_submission = pd.read_csv("./sample_submission.csv")
sample_submission["Predicted"] = y_test_pred_batch
sample_submission.to_csv("batch_gradient_descent_submission.csv", index=False)
