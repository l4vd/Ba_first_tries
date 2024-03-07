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

data = pd.read_csv("data.csv", delimiter=",")

y = data["peak_position"]
X = data.drop(columns=["peak_position"])

def preprocess(df, exclude_cols=None):
    # Check for missing values in numerical features
    missing_numerical = df.select_dtypes(include=['number']).isnull().sum()
    # Fill missing values with mean for each numeric attribute
    imputer = SimpleImputer(strategy='mean')
    df_filled = df.copy()
    for col in missing_numerical.index:
        if missing_numerical[col] > 0:
            df_filled[col] = imputer.fit_transform(df[[col]])
    # Normalize numerical features into [0, 1] range with MinMaxScaler
    scaler = MinMaxScaler()
    if exclude_cols:
        numerical_cols = df_filled.select_dtypes(include=['number']).columns.difference(exclude_cols)
    else:
        numerical_cols = df_filled.select_dtypes(include=['number']).columns
    df_normalized = pd.DataFrame(scaler.fit_transform(df_filled[numerical_cols]),
                                 columns=numerical_cols)

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    if exclude_cols:
        categorical_cols = df.select_dtypes(include=['object']).columns.difference(exclude_cols)
    else:
        categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = encoder.fit_transform(df[categorical_cols])

    # Concatenate numerical and encoded categorical features
    df_processed = hstack([df_normalized.values, df_encoded])

    return df_processed


# Example usage:
X_prep = preprocess(X, exclude_cols=['name_x', 'name_y', 'artist1_id', 'artist2_id',"song_id", "song_name", "genres"])

# Assuming y is a 1D array
y_reshaped = y.values.reshape(-1, 1)

# Create the scaler
scaler = MinMaxScaler()

# Fit and transform the scaled array
y_scaled = scaler.fit_transform(y_reshaped)


# Assuming X is your feature dataset and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(X_prep, y_scaled, test_size=0.25, random_state=42, stratify=y_scaled)


# Check if GPU is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the input shape based on your data
input_shape = (4537,)  # Assuming 4537 columns based on max column index

# Define the model architecture and move it to the GPU
class MLPRegressor(nn.Module):
    def __init__(self):
        super(MLPRegressor, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[0], 128),  # First hidden layer with 128 neurons
            nn.ReLU(),
            nn.Linear(128, 64),  # Second hidden layer with 64 neurons
            nn.ReLU(),
            nn.Linear(64, 1)  # Output layer with 1 neuron for regression
        ).to(device)  # Move the model to the GPU

    def forward(self, x):
        return self.layers(x)

model = MLPRegressor()

# Move the data to the GPU if available
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

# Define loss function and optimizer (same as TensorFlow example)
loss_fn = nn.MSELoss()
loss_fn_mae = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())

# Training loop
train_losses = []
for epoch in range(10):  # Adjust epochs as needed
    epoch_loss = 0.0
    for i in range(len(X_train)):
        # Forward pass
        y_pred = model(X_train[i])
        loss = loss_fn(y_pred, y_train[i])

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Calculate average epoch loss
    avg_epoch_loss = epoch_loss / len(X_train)
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
