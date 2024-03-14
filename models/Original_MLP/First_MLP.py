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
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn import metrics
from sklearn.utils import resample
import scipy.sparse as sp


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
    'artist1_id': str,          #evtl ersätzen mit eintweder haswert oder count
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
data['date'] = pd.to_datetime(data['release_date'])
data['timestamp'] = data['date'].apply(lambda x: x.timestamp())
data.drop(columns=["song_id", "song_name", "artist1_id", "artist2_id", "name_x", "name_y", "date", "release_date"], inplace=True)

y = data["hit"]
X = data.drop(columns=["hit"])

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
    encoder = OneHotEncoder(handle_unknown='ignore')
    if exclude_cols:
        categorical_cols = df.select_dtypes(include=['object']).columns.difference(exclude_cols)
    else:
        categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = encoder.fit_transform(df[categorical_cols])

    # Concatenate numerical and encoded categorical features
    df_processed = hstack([df_normalized.values, df_encoded])

    return df_processed


# Example usage:
X_prep = preprocess(X, exclude_cols=['name_x', 'name_y', 'artist1_id', 'artist2_id',"song_id", "song_name"])

# Assuming y is a 1D array
y_reshaped = y.values.reshape(-1, 1)

# Create the scaler
scaler = MinMaxScaler()

# Fit and transform the scaled array
y_scaled = scaler.fit_transform(y_reshaped)
print("######PREPROCESSING DONE######")

# Assuming X is your feature dataset and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(X_prep, y_scaled, test_size=0.25, random_state=42, stratify=y_scaled, shuffle=True)
print("######TRAIN TEST SPLIT DONE######")


# Check if GPU is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the input shape based on your data
#input_shape = (4537,)  # Assuming 4537 columns based on max column index

# Define the model architecture and move it to the GPU
class MLPClassifier(nn.Module):
    def __init__(self, input_shape):
        super(MLPClassifier, self).__init__()
        self.input_shape = input_shape
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_shape[1], 128),  # First hidden layer with 128 neurons
            nn.ReLU(),
            nn.Linear(128, 64),  # Second hidden layer with 64 neurons
            nn.ReLU(),
            nn.Linear(64, 1)  # Output layer with 1 output neuron for classification
        ).to(device)  # Move the model to the GPU

    def forward(self, x):
        logits = self.layers(x)
        return torch.sigmoid(logits)

print("######NETWORK DEFINED######")

y_train_cpy = y_train.copy()
y_train_cpy = y_train_cpy#.to(device)

#y_train_int = y_train.to(torch.int64)
y_train_cpy = torch.tensor(y_train_cpy, dtype=torch.int64)
y_train_flat = y_train_cpy.flatten()

# Count the number of samples in each class
class_counts = torch.bincount(y_train_flat)#.to(device))
class_counts = np.array(class_counts)

# Determine the maximum class count
#max_class_count = class_counts.max().item()
#total_samples = class_counts.sum().item()
# Compute weights for each sample based on class imbalance
#weights = total_samples / class_counts
#weights = weights.numpy()
#print("weights for upsampling: ", weights)
print("class count: ", class_counts)



#upsampling:
difference = class_counts.max() - class_counts.min()
difference = 674301 - 8969
# Assuming `X_train` is your original CSR matrix
num_rows_to_duplicate = difference // sum(y_train)  # Number of rows to duplicate per positive instance

# Filter indices of positive instances
positive_indices = [i for i, label in enumerate(y_train) if label == 1]

# Get rows corresponding to positive instances
rows_to_duplicate = sp.csr_matrix(X_train.getrow(positive_indices[0]))

for i in positive_indices[1:]:
    row = sp.csr_matrix(X_train.getrow(i))
    rows_to_duplicate = sp.vstack([rows_to_duplicate, row])

#rows_to_duplicate = sp.vstack([X_train.getrow(i) for i in positive_indices])

# Duplicate rows

duplicated_rows = sp.vstack([rows_to_duplicate])
if num_rows_to_duplicate >= 2:
    #num_rows_to_duplicate -= 1
    for i in range(int(num_rows_to_duplicate[0]) - 1):
        duplicated_rows = sp.vstack([duplicated_rows, rows_to_duplicate])


# Stack duplicated rows with the original matrix
X_train_upsampled = sp.vstack([X_train, duplicated_rows])

x = int(num_rows_to_duplicate[0]) * len(positive_indices)

# Create an array of shape (x, 1) with all elements as 1
rows_of_ones = np.ones((x, 1))

# Append rows_of_ones to original_array
y_train_upsampled = np.concatenate((y_train, rows_of_ones), axis=0)
print("######UPSAMPLING DONE######")


# convert to Pytorch tensor
X_train = torch.tensor(X_train_upsampled.toarray(), dtype=torch.float32)
X_test = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_train = torch.tensor(y_train_upsampled, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
print("######CONVERSION TO TENSOR######")

# Move the data to the GPU if available
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

#define model
print(X_train.size())
model = MLPClassifier(X_train.size())

# Define loss function and optimizer (same as TensorFlow example)
loss_fn = nn.BCEWithLogitsLoss()#pos_weight=weights)#BCELoss(weights=weights)#nn.MSELoss()
loss_fn_mae = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())

# Create a WeightedRandomSampler to sample with replacement according to weights
#weights = list(weights) * len(X_train)
#print(type(weights))
#weights2 = weights.copy()#/(weights.max()+1)
#weights = weights2[::-1]
#print(weights)
#weights_tensor = torch.tensor(np.array([75.18129, 1]), dtype=torch.double).to(device)
#sampler = WeightedRandomSampler(weights_tensor, len(X_train))

#X_train_oversampled, y_train_oversampled = resample(X_train, y_train, replace=True,
#                                                    random_state=42,
#                                                    n_samples=int(len(X_train * 2)), weights=weights)

# Create DataLoader with oversampled data
dataset_train = TensorDataset(X_train, y_train)
trainloader = DataLoader(dataset_train, batch_size=32, shuffle=True)

def calculate_accuracy(output, labels):
    predictions = output.round()  # Rundet die Ausgabe auf 0 oder 1
    correct = (predictions == labels).float()  # Konvertiert in float für die Division
    accuracy = correct.sum() / len(correct)
    return accuracy


# Training loop
train_losses = []
val_losses = []
val_accs = []
for epoch in range(10):  # Adjust epochs as needed
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0

    # Initialize counts for each class
    class_counts = {0: 0, 1: 0}  # Assuming binary classification

    # Training phase
    model.train()  # Set model to training mode
    for X_batch, y_batch in trainloader:
        # Forward pass
        y_pred = model(X_batch)
        #print("y_batch: ", y_batch)
        #print("y_pred: ", y_pred)
        loss = loss_fn(y_pred, y_batch)

        for label in y_batch:
            #print(y_batch)
            class_counts[label.item()] += 1

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
    # Calculate average epoch training loss
    avg_epoch_train_loss = epoch_train_loss / len(trainloader)
    train_losses.append(avg_epoch_train_loss)

    print("Class counts during training: ", class_counts)

    # Validation phase
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        y_val_pred = model(X_test)  # Assuming X_val is your validation data
        val_loss = loss_fn(y_val_pred, y_test)  # Assuming y_val is your validation target
        #print("Prediction Data type: ", type(y_val_pred))
        #print("Pred: \n", y_val_pred)
        #print("Test Data type: ", type(y_test))
        #print("Test: \n", y_test)
        epoch_val_acc = calculate_accuracy(y_val_pred, y_test)
        epoch_val_loss = val_loss.item()
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        print(f"Epoch [{epoch + 1}/10], Training Loss: {avg_epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}")

print("######TRAINING DONE######")


# Make predictions on new data (replace with your data)
# Assuming your test data is stored in X_test
#predictions = model(X_test)

# Calculate and print MSE and MAE on test data
#test_loss = loss_fn(predictions, y_test).item()
#test_loss_mae = loss_fn_mae(predictions, y_test).item()
#print(f"Test MSE: {test_loss:.4f}")
#print(f"Test MAE: {test_loss_mae:.4f}")

# Plot the training loss
# Plot the training loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs. Epoch')
plt.legend()
plt.savefig("losses.png")
print("######LOSS PLOT DONE######")

# Calculate confusion matrix
output = model(X_test)
predictions = output.round().int().tolist()  # Converting tensor to list of integers
true_labels = y_test.int().tolist()  # Converting tensor to list of integers
#conf_matrix = confusion_matrix(true_labels, predictions)

confusion_matrix = metrics.confusion_matrix(true_labels, predictions)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.savefig("Confusion_Matrix.png")