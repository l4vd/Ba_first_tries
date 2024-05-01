from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
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
import random
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


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
    'weighted degree_x': float,
    'betweenesscentrality_x': float,
    'Cluster_x': float,
    'eigencentrality_y': float,
    'name_y': str,
    'eccentricity_y': float,
    'degree_y': float,
    'clustering_y': float,
    'closnesscentrality_y': float,
    'weighted degree_y': float,
    'betweenesscentrality_y': float,
    'Cluster_y': float
}
data = pd.read_csv("data_superstar_v1_0.csv", delimiter=",", dtype=dtype_dict, na_values=[''])
data['date'] = pd.to_datetime(data['release_date'])
data.sort_values(by="date", inplace=True)

# List of columns to keep
columns_to_keep = ['explicit', 'track_number', 'num_artists', 'num_available_markets', 'release_date',
                   'duration_ms', 'key', 'mode', 'time_signature', 'acousticness',
                   'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',
                   'speechiness', 'valence', 'tempo', 'years_on_charts', 'hit', "superstar_v1_x", "superstar_x"]                              #Collaboration Profile == CLuster????
#  'release_date', 'betweenesscentrality_x', 'closnesscentrality_x', 'clustering_x', 'Cluster_x',
                   # 'eccentricity_x', 'eigencentrality_x', 'weighted degree_x', "profile_x",
                   # 'betweenesscentrality_y', 'closnesscentrality_y', 'clustering_y', 'Cluster_y',
                   # 'eccentricity_y', 'eigencentrality_y', 'weighted degree_y', "profile_y", "hit"]                              #Collaboration Profile == CLuster????

# Drop columns not in the list
data = data[columns_to_keep]

def find_min_max(df):
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=['number'])

    # Find max and min values for each numeric column
    min_max_values = {}
    for col in numeric_cols.columns:
        min_value = df[col].min()
        max_value = df[col].max()
        min_max_values[col] = {'min': min_value, 'max': max_value}

    return min_max_values

min_max_val = find_min_max(data)

y = data["hit"]
X = data.drop(columns=["hit"])

def preprocess(df, min_max_values, exclude_cols=None):
    missing_numerical = df.select_dtypes(include=['number']).isnull().sum()
    # Fill missing values with mean for each numeric attribute
    imputer = SimpleImputer(strategy='mean')
    df_filled = df.copy()
    for col in missing_numerical.index:
        if missing_numerical[col] > 0:
            df_filled[col] = imputer.fit_transform(df[[col]])

    # Normalize numerical features into [0, 1] range with MinMaxScaler
    if exclude_cols:
        numerical_cols = df_filled.select_dtypes(include=['number']).columns.difference(exclude_cols)
    else:
        numerical_cols = df_filled.select_dtypes(include=['number']).columns
    
    #print("numerical columns:", numerical_cols)

    for column_name in numerical_cols:
        df_filled[column_name] = (df_filled[column_name] - min_max_values[column_name]["min"]) / (min_max_values[column_name]["max"] - min_max_values[column_name]["min"])

    df_normalized = pd.DataFrame(df_filled, columns=numerical_cols)

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    if exclude_cols:
        categorical_cols = df.select_dtypes(include=['object']).columns.difference(exclude_cols)
    else:
        categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = encoder.fit_transform(df[categorical_cols])

    #print(categorical_cols)

    # Convert the sparse matrix to dense array
    df_encoded_dense = df_encoded.toarray()

    # Concatenate numerical and encoded categorical features
    df_processed = np.hstack([df_normalized.values, df_encoded_dense])

    return df_processed

# Example usage:
# Assuming df is your DataFrame
#processed_data = preprocess_with_scaling(df)

# Assuming y is a 1D array
#y_reshaped = y.values.reshape(-1, 1)

# Create the scaler
#scaler = MinMaxScaler()

# Fit and transform the scaled array
#y_scaled = scaler.fit_transform(y_reshaped)
#print("######PREPROCESSING DONE######")

# Assuming X is your feature dataset and y is your target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)#random_state=42), stratify=y_scaled, shuffle=True) # try to do with ordered by date results are terrible:(, ..collab prof is missing
#X_train, y_train = shuffle(X_train, y_train, random_state=42)
print("######TRAIN TEST SPLIT DONE######")

def upsampling(X_train, y_train):
    # Convert y_train to a numpy array
    #y_train = y_train.to_numpy()
    X_train = X_train.to_numpy()

    # Count the number of samples in each class
    class_counts = np.bincount(y_train.flatten().astype(int))
    max_count = class_counts.max()

    # Find indices of positive instances
    positive_indices = np.where(y_train.flatten() == 1)[0]

    # Calculate how many times to duplicate positive samples
    difference = max_count - class_counts[1]

    # Randomly select indices from positive instances
    random_indices = np.random.choice(positive_indices, size=difference, replace=True)

    # Get rows corresponding to positive instances
    rows_to_duplicate = np.vstack([X_train[idx] for idx in random_indices])

    # Stack duplicated rows with the original matrix
    X_train_upsampled = np.vstack([X_train, rows_to_duplicate])

    # Create an array of shape (x, 1) with all elements as 1
    rows_of_ones = np.ones((difference, 1))

    # Append rows_of_ones to original_array
    y_train_upsampled = np.concatenate((y_train, rows_of_ones), axis=0)

    print("######UPSAMPLING DONE######")
    return X_train_upsampled, y_train_upsampled

y_reshaped = y_train.values.reshape(-1, 1)
#print(X_train.shape)
#print(y_reshaped.shape)
X_train_upsampled, y_train_upsampled = upsampling(X_train=X_train, y_train=y_reshaped)
# Assuming X_train, X_test, y_train, y_test are your training and testing data
#print("X_train_up type:", type(X_train_upsampled))
#print("y_train_up type:", type(y_train_upsampled))
#print("X_train_up shape:", X_train_upsampled.shape)
#print("y_train_up shape:", y_train_upsampled.shape)
#print(type(X_test))
#print(type(y_test))

# Count occurrences of each unique value
unique_values, counts = np.unique(y_train_upsampled, return_counts=True)

# Create a dictionary to store the counts of each value
value_counts = dict(zip(unique_values, counts))

print("Value counts:", value_counts)

# Convert arrays to DataFrames
X_train_upsampled_df = pd.DataFrame(X_train_upsampled, columns=X_train.columns)
y_train_upsampled_df = pd.DataFrame(y_train_upsampled, columns=['hit'])

# Concatenate y_train_upsampled as an extra column to X_train_upsampled_df
X_train_upsampled_with_y = pd.concat([X_train_upsampled_df, y_train_upsampled_df], axis=1)
X_train_upsampled_with_y['date'] = pd.to_datetime(X_train_upsampled_with_y['release_date'])
X_train_upsampled_with_y.sort_values(by="date", inplace=True)
X_train_upsampled_with_y.drop(columns=["release_date", "date"], inplace=True)

#print(X_train_upsampled_with_y.head())
#prepro:
y_train_upsampled_ordered = X_train_upsampled_with_y["hit"]
X_train_upsampled_ordered = X_train_upsampled_with_y.drop(columns="hit")

# Define data types for each column
dtype_dict = {
    'explicit': bool,
    'track_number': float,
    'num_artists': float,
    'num_available_markets': float,
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
    'years_on_charts': float,
    "superstar_v1_x": float,
    "superstar_x": int
    # 'betweenesscentrality_x': float,
    # 'closnesscentrality_x': float,
    # 'clustering_x': float,
    # 'Cluster_x': str,
    # 'eccentricity_x': float,
    # 'eigencentrality_x': float,
    # 'weighted degree_x': float,
    # 'profile_x': str,
    # 'betweenesscentrality_y': float,
    # 'closnesscentrality_y': float,
    # 'clustering_y': float,
    # 'Cluster_y': str,
    # 'eccentricity_y': float,
    # 'eigencentrality_y': float,
    # 'weighted degree_y': float,
    # 'profile_y': str,
}

# Use astype method to cast columns to the specified data types
X_train_upsampled_ordered = X_train_upsampled_ordered.astype(dtype_dict)
X_test.drop(columns="release_date", inplace=True)
X_test = X_test.astype(dtype_dict)

y_train_upsampled_ordered_reshaped = y_train_upsampled_ordered.values.reshape(-1, 1)
y_test_reshaped = y_test.values.reshape(-1, 1)

sep_index =  X_train_upsampled_ordered.shape[0]
concatenated_df = pd.concat([X_train_upsampled_ordered, X_test])
#print(min_max_val)
data_prepro = preprocess(concatenated_df, min_max_val)
X_train_upsampled_prepro = data_prepro[:sep_index]
X_test_prepro = data_prepro[sep_index:]

###EVTL MIN mAx SCALING AUF HIT (y)
## Create the scaler
#scaler = MinMaxScaler()
#
## Fit and transform the scaled array
#y_scaled = scaler.fit_transform(y_reshaped)
print("######PREPROCESSING DONE######")

# Check if GPU is available, otherwise fall back to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

# convert to Pytorch tensor
X_train = torch.tensor(X_train_upsampled_prepro, dtype=torch.float32)
X_test = torch.tensor(X_test_prepro, dtype=torch.float32)
y_train = torch.tensor(y_train_upsampled_ordered_reshaped, dtype=torch.float32)
y_test = torch.tensor(y_test_reshaped, dtype=torch.float32)
print("######CONVERSION TO TENSOR######")

# Move the data to the GPU if available
X_train = X_train.to(device)
y_train = y_train.to(device)
X_test = X_test.to(device)
y_test = y_test.to(device)

#define model
print(X_train.size())
model = MLPClassifier(X_train.size()).to(device)

# Define loss function and optimizer (same as TensorFlow example)
loss_fn = nn.BCELoss()   # alternative #BCELoss(weights=weights)#nn.MSELoss()
loss_fn_mae = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())

# Create DataLoader with oversampled data
dataset_train = TensorDataset(X_train, y_train)
trainloader = DataLoader(dataset_train, batch_size=32, shuffle=True)  #set shuffle false?

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

    # Training phase
    model.train()  # Set model to training mode
    for X_batch, y_batch in trainloader:
        # Forward pass
        y_pred = model(X_batch)
        #print("y_batch: ", y_batch)
        #print("y_pred: ", y_pred)
        loss = loss_fn(y_pred, y_batch)

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_train_loss += loss.item()
    # Calculate average epoch training loss
    avg_epoch_train_loss = epoch_train_loss / len(trainloader)
    train_losses.append(avg_epoch_train_loss)

    # Validation phase
    model.eval()  # Set model to evaluation mode
    with torch.no_grad():
        y_val_pred = model(X_test)  # Assuming X_val is your validation data
        val_loss = loss_fn(y_val_pred, y_test)  # Assuming y_val is your validation target
        epoch_val_acc = calculate_accuracy(y_val_pred, y_test)
        epoch_val_loss = val_loss.item()
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        print(f"Epoch [{epoch + 1}/200], Training Loss: {avg_epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}")

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
#print("output", output)

opt_thres = -1
opt_prec = 0
liste_thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
true_labels = y_test.int().tolist()
#print(output.tolist())
for i in liste_thresh:
    flattened_list = [item for sublist in output.tolist() for item in sublist]
    predictions = list(map(lambda x: int(x >= i), flattened_list))

    precision = metrics.precision_score(true_labels, predictions)

    # Recall
    recall = metrics.recall_score(true_labels, predictions)
    # F1-Score
    f1 = metrics.f1_score(true_labels, predictions)
    # ROC Curve and AUC
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
    roc_auc = metrics.auc(fpr, tpr)

    # print("Precision:", precision)
    # print("Recall:", recall)
    # print("F1-Score:", f1)
    # print("ROC AUC:", roc_auc)

    if precision > opt_prec:
        opt_thres = i
        opt_prec = precision
print(f"optimal threshold {opt_thres}, with precision {opt_prec}")

predictions = output.round().int().tolist()  # Converting tensor to list of integers
true_labels = y_test.int().tolist()  # Converting tensor to list of integers

confusion_matrix = metrics.confusion_matrix(true_labels, predictions)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.savefig("Confusion_Matrix.png")
print("######CONFUSION MATRIX PLOT DONE######")

# Extract TN, FP, TP values
TN = confusion_matrix[0, 0]  # True Negatives
FP = confusion_matrix[0, 1]  # False Positives
FN = confusion_matrix[1, 0]  # False Negatives
TP = confusion_matrix[1, 1]  # True Positives

# Print the results
print("True Negatives (TN):", TN)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)
print("True Positives (TP):", TP)

# Precision 
precision = metrics.precision_score(true_labels, predictions) 
# Recall 
recall = metrics.recall_score(true_labels, predictions) 
# F1-Score 
f1 = metrics.f1_score(true_labels, predictions) 
# ROC Curve and AUC 
fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions) 
roc_auc = metrics.auc(fpr, tpr) 
  
print("Precision:", precision) 
print("Recall:", recall) 
print("F1-Score:", f1) 
print("ROC AUC:", roc_auc) 

#print(output.device)
output_cpu = output.cpu().detach().numpy()

fpr, tpr, thresholds = metrics.roc_curve(y_test.tolist(), output_cpu.tolist())
roc_auc = metrics.auc(fpr, tpr) 

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("ROC_AUC.png")
print("######ROC-AUC PLOT DONE######")

#print(y_test.tolist())
#print(output_cpu.tolist())
# output_list = output_cpu.tolist()
# for i, elt in enumerate(output_list):
# 	output_list[i] = [int(elt[0])]


# Generate a classification report
class_report = classification_report(y_test.tolist(), predictions)
print("Classification Report:\n", class_report)

y_test_cpu = y_test.cpu()
y_test_list = y_test_cpu.tolist()

# Convert predictions to list
predictions_list = list(np.hstack(predictions))

y_test_series = pd.Series(list(np.hstack(y_test_list)))
count_occ = y_test_series.value_counts(normalize=True)

# Calculate the weighted accuracy
weighted_acc = (np.sum((y_test_series == 1) == predictions_list) * count_occ[0] + np.sum((y_test_series == 0) == predictions_list) * count_occ[1]) / len(y_test_list)

print("Weighted Accuracy:", weighted_acc)

macro_f1 = metrics.f1_score(true_labels, predictions, average='macro')

print("Macro F1 Score:", macro_f1)

#f1 opt 7
#auc opt 1