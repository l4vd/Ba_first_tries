from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pandas as pd
from scipy.sparse import hstack
from sklearn.preprocessing import OneHotEncoder
import torch
from torch import nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

dtype_dict = {
    'song_id': str,
    'song_name': str,
    'song_popularity': float,
    'explicit': bool,
    'song_type': str,
    'track_number': float,
    'num_artists': float,
    'num_available_markets': float,
    'release_date': str,  # Assuming it's a date, change to appropriate type if needed
    'duration_ms': float,
    'key': float,
    'mode': float,
    'time_signature': float,
    'acousticness': float,
    'danceability': float,
    'energy': float,
    'instrumentalness': float,
    'liveness': float,
    'loudness': float,
    'speechiness': float,
    'valence': float,
    'tempo': float,
    'hit': float,
    'nr_artists': float,
    'artist1_id': str,
    'artist2_id': str,
    'eigencentrality_x': float,
    'name_x': str,
    'eccentricity_x': float,
    'degree_x': float,
    'clustering_x': float,
    'closnesscentrality_x': float,
    'weighted_degree_x': float,
    'betweenesscentrality_x': float,
    'Cluster_x': float,
    'eigencentrality_y': float,
    'name_y': str,
    'eccentricity_y': float,
    'degree_y': float,
    'clustering_y': float,
    'closnesscentrality_y': float,
    'weighted_degree_y': float,
    'betweenesscentrality_y': float,
    'Cluster_y': float
}
data = pd.read_csv("data_basline_simple_feature_calc.csv", delimiter=",", dtype=dtype_dict, na_values=[''])

y = data["hit"]
X = data.drop(columns=["hit"])
# Define the custom data generator class
class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Define the preprocessing steps within a pipeline
numeric_features = X.select_dtypes(include=['number']).columns
categorical_features = X.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Preprocess the data and split into train and test sets
X_processed = preprocessor.fit_transform(X)
print("######PREPROCESSING DONE######")
X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.25, random_state=42, stratify=y)
print("######TRAIN TEST SPLIT DONE######")

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train.toarray(), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

# Define train and test datasets using custom data generator
train_dataset = CustomDataset(X_train_tensor, y_train_tensor)
test_dataset = CustomDataset(X_test_tensor, y_test_tensor)

# Define data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("######DATALOADERS DEFINED######")

# Check if GPU is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model architecture and move it to the GPU
class MLPRegressor(nn.Module):
    def __init__(self, input_shape):
        super(MLPRegressor, self).__init__()
        self.input_shape = input_shape
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[1], 128),  # First hidden layer with 128 neurons
            nn.ReLU(),
            nn.Linear(128, 64),  # Second hidden layer with 64 neurons
            nn.ReLU(),
            nn.Linear(64, 1)  # Output layer with 1 neuron for regression
        ).to(device)  # Move the model to the GPU

    def forward(self, x):
        return self.layers(x)
print("######NETWORK DEFINED######")

#define model
print(X_train.size())
model = MLPRegressor(X_train.size())

# Define loss function and optimizer (same as TensorFlow example)
loss_fn = nn.MSELoss()
loss_fn_mae = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
batch_size = 32  # Define your desired batch size
train_losses = []
for epoch in range(10):  # Adjust epochs as needed
    epoch_loss = 0.0
    num_batches = len(train_loader)

    for data, target in train_loader:
        # Move data and target to device
        data, target = data.to(device), target.to(device)

        # Forward pass
        y_pred = model(data)
        loss = loss_fn(y_pred, target)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / num_batches
    train_losses.append(avg_epoch_loss)
    print(f"Epoch [{epoch + 1}/10], Loss: {avg_epoch_loss:.4f}")

# Make predictions on new data (replace with your data)
# Assuming your test data is stored in X_test
predictions = model(X_test)

# Calculate and print MSE and MAE on test data
test_loss = loss_fn(predictions, y_test).item()
test_loss_mae = loss_fn_mae(predictions, y_test).item()
print(f"Test MSE: {test_loss:.4f}")
print(f"Test MAE: {test_loss_mae:.4f}")

# Plot the training loss
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch')
plt.show()
