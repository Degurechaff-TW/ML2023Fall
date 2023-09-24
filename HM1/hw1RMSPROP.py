import pandas as pd
import numpy as np

# Load data
train_data = pd.read_csv("./train.csv")

# Data preprocessing: Removing outliers
train_data = train_data[train_data['PM2.5'] <= 100]

# Feature selection based on correlation analysis
correlations = train_data.corr()['PM2.5'].abs()
selected_features_with_pm25 = correlations[correlations > 0].index.tolist()

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

X_train_normalized = (X_train_split - mean_train_with_pm25) / (std_train_with_pm25 + 1e-8)
X_val_normalized = (X_val_split - mean_train_with_pm25) / (std_train_with_pm25 + 1e-8)

# Implementing RMSprop for linear regression
alpha = 0.01
beta = 0.9
epsilon = 1e-8

W_rmsprop = np.zeros(X_train_normalized.shape[1])
b_rmsprop = 0
s_W = np.zeros(X_train_normalized.shape[1])
s_b = 0

num_epochs = 50000
mse_train_rmsprop_list = []

for epoch in range(num_epochs):
    y_train_pred = np.dot(X_train_normalized, W_rmsprop) + b_rmsprop
    error_train = y_train_pred - y_train_split
    mse_train = np.mean(error_train ** 2)
    mse_train_rmsprop_list.append(mse_train)
    
    gradient_W = (2/len(X_train_normalized)) * np.dot(X_train_normalized.T, error_train)
    gradient_b = (2/len(X_train_normalized)) * error_train.sum()
    
    s_W = beta * s_W + (1 - beta) * gradient_W ** 2
    s_b = beta * s_b + (1 - beta) * gradient_b ** 2
    
    W_rmsprop -= alpha * gradient_W / (np.sqrt(s_W) + epsilon)
    b_rmsprop -= alpha * gradient_b / (np.sqrt(s_b) + epsilon)

min_mse_value = min(mse_train_rmsprop_list)
min_mse_epoch = mse_train_rmsprop_list.index(min_mse_value)
print(min_mse_value, min_mse_epoch)
# Evaluating on validation set
y_val_pred_rmsprop = np.dot(X_val_normalized, W_rmsprop) + b_rmsprop
error_val_rmsprop = y_val_pred_rmsprop - y_val_split
mse_val_rmsprop = np.mean(error_val_rmsprop ** 2)

print(mse_val_rmsprop)
