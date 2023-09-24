import pandas as pd
import numpy as np

# Step 1: Load data
train_data = pd.read_csv("./train.csv")
test_data = pd.read_csv("./test.csv")

# Step 2: Data preprocessing - Removing outliers for PM2.5 in the training data
train_data = train_data[train_data['PM2.5'] <= 100]

# Step 3: Feature selection based on correlation analysis
correlations = train_data.corr()['PM2.5'].abs().sort_values(ascending=False)
selected_features_with_pm25 = correlations[correlations > 0].index.tolist()


# Extracting the selected features including PM2.5 from the training data
X_train_data_with_pm25 = train_data[selected_features_with_pm25].values


# Step 4: Preparing training data
train_X_with_pm25 = []
train_y_with_pm25 = []
for i in range(len(X_train_data_with_pm25) - 8):
    train_X_with_pm25.append(X_train_data_with_pm25[i:i+8].flatten())
    train_y_with_pm25.append(X_train_data_with_pm25[i+8, 0])

train_X_with_pm25 = np.array(train_X_with_pm25)
train_y_with_pm25 = np.array(train_y_with_pm25)

# Step 5: Data normalization
mean_train_with_pm25 = np.mean(train_X_with_pm25, axis=0)
std_train_with_pm25 = np.std(train_X_with_pm25, axis=0)
train_X_normalized_with_pm25 = (train_X_with_pm25 - mean_train_with_pm25) / (std_train_with_pm25 + 1e-8)

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

# Step 6: Implementing Adam optimizer for linear regression
alpha = 0.001
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
m_W = np.zeros(X_train_normalized.shape[1])
v_W = np.zeros(X_train_normalized.shape[1])
m_b = 0
v_b = 0
t = 0

W_adam_pm25 = np.zeros(X_train_normalized.shape[1])
b_adam_pm25 = 0

num_epochs = 500000
mse_list = []

for epoch in range(num_epochs):
    y_pred = np.dot(X_train_normalized, W_adam_pm25) + b_adam_pm25
    error = y_pred - y_train_split
    mse = np.mean(error**2)
    mse_list.append(mse)
    gradient_W = (2/len(X_train_normalized)) * np.dot(X_train_normalized.T, error)
    gradient_b = (2/len(X_train_normalized)) * error.sum()
    
    t += 1
    m_W = beta1 * m_W + (1 - beta1) * gradient_W
    v_W = beta2 * v_W + (1 - beta2) * (gradient_W ** 2)
    m_W_corrected = m_W / (1 - beta1 ** t)
    v_W_corrected = v_W / (1 - beta2 ** t)
    W_adam_pm25 -= alpha * m_W_corrected / (np.sqrt(v_W_corrected) + epsilon)

    m_b = beta1 * m_b + (1 - beta1) * gradient_b
    v_b = beta2 * v_b + (1 - beta2) * (gradient_b ** 2)
    m_b_corrected = m_b / (1 - beta1 ** t)
    v_b_corrected = v_b / (1 - beta2 ** t)
    b_adam_pm25 -= alpha * m_b_corrected / (np.sqrt(v_b_corrected) + epsilon)

min_mse_value = min(mse_list)
min_mse_epoch = mse_list.index(min_mse_value)
print(min_mse_value, min_mse_epoch)

# After training, we predict on the validation set
y_val_pred = np.dot(X_val_normalized, W_adam_pm25) + b_adam_pm25
error_val = y_val_pred - y_val_split
mse_val = np.mean(error_val ** 2)
print(mse_val)

# Step 7: Preparing the test data
X_test_data_with_pm25 = test_data[selected_features_with_pm25].values

test_X_pm25 = []
for i in range(0, len(X_test_data_with_pm25) - 7, 8):
    test_X_pm25.append(X_test_data_with_pm25[i:i+8].flatten())

test_X_pm25 = np.array(test_X_pm25)

# Normalize the test data using the mean and std from the training data
test_X_normalized_pm25 = (test_X_pm25 - mean_train_with_pm25) / (std_train_with_pm25 + 1e-8)

# Step 8: Predicting the PM2.5 values for the test data
y_test_pred_adam_pm25 = np.dot(test_X_normalized_pm25, W_adam_pm25) + b_adam_pm25

# Step 9: Output the results to a new CSV file
sample_submission = pd.read_csv("./sample_submission.csv")
submission_df = pd.DataFrame({
    "Id": sample_submission["Id"],
    "Predicted": y_test_pred_adam_pm25
})
output_file_path = "./predicted_submission.csv"
submission_df.to_csv(output_file_path, index=False)