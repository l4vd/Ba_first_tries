import matplotlib
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from torch import nn

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn import metrics
import random
from sklearn.metrics import classification_report

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
    'release_date': str,
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
                   'speechiness', 'valence', 'tempo', "date", "years_on_charts",               #removeyoc
                   'hit']  # , "superstar_v1_x", "superstar_x"]                              #Collaboration Profile == CLuster????
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

    # print("numerical columns:", numerical_cols)

    for column_name in numerical_cols:
        df_filled[column_name] = (df_filled[column_name] - min_max_values[column_name]["min"]) / (
                min_max_values[column_name]["max"] - min_max_values[column_name]["min"])

    df_normalized = pd.DataFrame(df_filled, columns=numerical_cols)

    # One-hot encode categorical features
    encoder = OneHotEncoder(handle_unknown='ignore')
    if exclude_cols:
        categorical_cols = df.select_dtypes(include=['object']).columns.difference(exclude_cols)
    else:
        categorical_cols = df.select_dtypes(include=['object']).columns
    df_encoded = encoder.fit_transform(df[categorical_cols])

    # print(categorical_cols)

    # Convert the sparse matrix to dense array
    df_encoded_dense = df_encoded.toarray()

    # Concatenate numerical and encoded categorical features
    df_processed = np.hstack([df_normalized.values, df_encoded_dense])

    return df_processed


# Assuming X is your feature dataset and y is your target variable
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)#random_state=42), stratify=y_scaled, shuffle=True) # try to do with ordered by date results are terrible:(, ..collab prof is missing
split_day = X["date"].iloc[-1] - pd.DateOffset(years=1)
X_train = X[(X["date"] < split_day)].copy()

X_test = X[(X["date"] >= split_day)].copy()
sep_index = X_train.shape[0]
y_train = y.iloc[:sep_index].copy()
y_test = y.iloc[sep_index:].copy()

print("######TRAIN TEST SPLIT DONE######")


def upsampling(X_train, y_train):
    # Convert y_train to a numpy array
    # y_train = y_train.to_numpy()
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

X_train_upsampled, y_train_upsampled = upsampling(X_train=X_train, y_train=y_reshaped)


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

# prepro:
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
}
# "superstar_v1_x": float,
# "superstar_x": int
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
# }

# Use astype method to cast columns to the specified data types
X_train_upsampled_ordered = X_train_upsampled_ordered.astype(dtype_dict)
X_test.drop(columns=["release_date", "date"], inplace=True)
X_test = X_test.astype(dtype_dict)

y_train_upsampled_ordered_reshaped = y_train_upsampled_ordered.values.reshape(-1, 1)
y_test_reshaped = y_test.values.reshape(-1, 1)

sep_index = X_train_upsampled_ordered.shape[0]
concatenated_df = pd.concat([X_train_upsampled_ordered, X_test])
print(concatenated_df.columns)
data_prepro = preprocess(concatenated_df, min_max_val)
X_train_upsampled_prepro = data_prepro[:sep_index]
X_test_prepro = data_prepro[sep_index:]

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
            nn.Linear(input_shape[1], 128), 
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(128, 256),
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(256, 512),  
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(512, 256),  
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(256, 128),  
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(128, 64),  
            nn.ReLU(),
            #nn.Dropout(0.4),
            nn.Linear(64, 1)
        ).to(device) 
    
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

# define model
print(X_train.size())
model = MLPClassifier(X_train.size()).to(device)

# Define loss function and optimizer (same as TensorFlow example)
loss_fn = nn.BCELoss()
loss_fn_mae = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters())

# Create DataLoader with oversampled data
dataset_train = TensorDataset(X_train, y_train)
trainloader = DataLoader(dataset_train, batch_size=256, shuffle=True)#, num_workers=2, pin_memory=True) #last two are new look at later


def calculate_accuracy(output, labels):
    predictions = output.round()  # Rundet die Ausgabe auf 0 oder 1
    correct = (predictions == labels).float()  # Konvertiert in float für die Division
    accuracy = correct.sum() / len(correct)
    return accuracy


def calculate_precision(y_pred_val, y_actual):
    y_val_pred_rounded = y_pred_val.round().int().tolist()
    actual_labels = y_actual.int().tolist()
    return metrics.precision_score(actual_labels, y_val_pred_rounded)


# Training loop
train_losses = []
val_losses = []
val_accs = []
val_prec = []
epochs = 200
best_val_loss = 1e8
best_val_acc = 0
best_precision = 0
version = "v3_drop_04_yoc_rem"

for epoch in range(epochs):  # Adjust epochs as needed
    epoch_train_loss = 0.0
    epoch_val_loss = 0.0

    # Initialize counts for each class

    # Training phase
    model.train()  # Set model to training mode
    for X_batch, y_batch in trainloader:
        # Forward pass
        y_pred = model(X_batch)
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
        y_val_pred = model(X_test)
        val_loss = loss_fn(y_val_pred, y_test)
        epoch_val_acc = calculate_accuracy(y_val_pred, y_test)
        epoch_val_loss = val_loss.item()

        epoch_val_prec = calculate_precision(y_val_pred, y_test)

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_val_loss,
            }, f'best_torch_{version}_model_min_val_loss.pth')
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, f'best_torch_{version}_model_max_val_acc.pth')
        if epoch_val_prec > best_precision:
            best_precision = epoch_val_prec
            torch.save({
                'epoch': epoch+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'precision': best_precision,
            }, f"best_torch_{version}_model_max_val_prec.pth")

        val_prec.append(epoch_val_prec)
        val_losses.append(epoch_val_loss)
        val_accs.append(epoch_val_acc)
        print(
            f"Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_epoch_train_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_acc:.4f}, Validation Precision: {epoch_val_prec:.4f}")

print("######TRAINING DONE######")


def load_model(model, path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model {path} is loaded from epoch {checkpoint['epoch']} , loss {checkpoint['loss']}")
    return model


print("######LOAD MODEL######")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MLPClassifier(X_train.size())
model = load_model(model, f"best_torch_{version}_model_max_val_prec.pth")
model = model.to(device)
model.eval()

# Plot the training loss
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss vs. Epoch')
plt.legend()
plt.savefig(f"losses_pytorch_{version}.png")
print("######LOSS PLOT DONE######")

# Calculate confusion matrix
output = model(X_test)

predictions = output.round().int().tolist()  # Converting tensor to list of integers
true_labels = y_test.int().tolist()  # Converting tensor to list of integers

confusion_matrix = metrics.confusion_matrix(true_labels, predictions)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])

cm_display.plot()
plt.savefig(f"Confusion_Matrix_pytorch_{version}.png")
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

# print(output.device)
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
plt.savefig(f"ROC_AUC_pytorch_{version}.png")
print("######ROC-AUC PLOT DONE######")

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
weighted_acc = (np.sum((y_test_series == 1) == predictions_list) * count_occ[0] + np.sum(
    (y_test_series == 0) == predictions_list) * count_occ[1]) / len(y_test_list)

print("Weighted Accuracy:", weighted_acc)

macro_f1 = metrics.f1_score(true_labels, predictions, average='macro')

print("Macro F1 Score:", macro_f1)
