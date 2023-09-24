import pandas as pd
import numpy as np

# Load data
train_data = pd.read_csv("/path/to/train.csv")
train_data = train_data[train_data['PM2.5'] <= 100]

# Feature selection based on correlation
correlations = train_data.corr()['PM2.5'].abs()
selected_features_with_pm25 = correlations[correlations > 0.5].index.tolist()

# Extracting features
X_train_data_with_pm25 = train_data[selected_features_with_pm25].values

train_X_with_pm25 = []
train_y_with_pm25 = []
for i in range(len(X_train_data_with_pm25) - 8):
    train_X_with_pm25.append(X_train_data_with_pm25[i:i+8].flatten())
    train_y_with_pm25.append(X_train_data_with_pm25[i+8, 0])

train_X_with_pm25 = np.array(train_X_with_pm25)
train_y_with_pm25 = np.array(train_y_with_pm25)

# Normalization
mean_train_with_pm25 = np.mean(train_X_with_pm25, axis=0)
std_train_with_pm25 = np.std(train_X_with_pm25, axis=0)

# Splitting the data
split_ratio = 0.8
split_index = int(split_ratio * len(train_X_with_pm25))

X_train_split = train_X_with_pm25[:split_index]
y_train_split = train_y_with_pm25[:split_index]
X_val_split = train_X_with_pm25[split_index:]
y_val_split = train_y_with_pm25[split_index:]

X_train_normalized = (X_train_split - mean_train_with_pm25) / (std_train_with_pm25 + 1e-8)
X_val_normalized = (X_val_split - mean_train_with_pm25) / (std_train_with_pm25 + 1e-8)

# Mini-Batch Gradient Descent
def generate_mini_batches(X, y, batch_size):
    m = X.shape[0]
    mini_batches = []
    
    # Shuffle
    permutation = list(np.random.permutation(m))
    shuffled_X = X[permutation, :]
    shuffled_y = y[permutation]
    
    num_complete_batches = m // batch_size
    for k in range(num_complete_batches):
        mini_batch_X = shuffled_X[k * batch_size:(k + 1) * batch_size, :]
        mini_batch_y = shuffled_y[k * batch_size:(k + 1) * batch_size]
        mini_batches.append((mini_batch_X, mini_batch_y))
        
    if m % batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_batches * batch_size:, :]
        mini_batch_y = shuffled_y[num_complete_batches * batch_size:]
        mini_batches.append((mini_batch_X, mini_batch_y))
    
    return mini_batches

alpha = 0.01
batch_size = 64
num_epochs = 1000

W_mbgd = np.zeros(X_train_normalized.shape[1])
b_mbgd = 0

for epoch in range(num_epochs):
    mini_batches = generate_mini_batches(X_train_normalized, y_train_split, batch_size)
    
    for mini_batch in mini_batches:
        (mini_batch_X, mini_batch_y) = mini_batch
        
        y_train_pred = np.dot(mini_batch_X, W_mbgd) + b_mbgd
        error_train = y_train_pred - mini_batch_y
        
        gradient_W = (2/len(mini_batch_X)) * np.dot(mini_batch_X.T, error_train)
        gradient_b = (2/len(mini_batch_X)) * error_train.sum()
        
        W_mbgd -= alpha * gradient_W
        b_mbgd -= alpha * gradient_b
