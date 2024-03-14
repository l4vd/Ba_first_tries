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


def upsampling(X_train, y_train):
    #upsampling:
    y_train_cpy = y_train.copy()

    y_train_cpy = torch.tensor(y_train_cpy, dtype=torch.int64)
    y_train_flat = y_train_cpy.flatten()

    # Count the number of samples in each class
    class_counts = torch.bincount(y_train_flat)
    class_counts = np.array(class_counts)


    print("class count: ", class_counts)

    difference = class_counts.max() - class_counts.min()
    # Assuming `X_train` is your original CSR matrix
    num_rows_to_duplicate = difference // sum(y_train)  # Number of rows to duplicate per positive instance

    # Filter indices of positive instances
    positive_indices = [i for i, label in enumerate(y_train) if label == 1]

    # Get rows corresponding to positive instances
    rows_to_duplicate = sp.csr_matrix(X_train.getrow(positive_indices[0]))

    for i in positive_indices[1:]:
        row = sp.csr_matrix(X_train.getrow(i))
        rows_to_duplicate = sp.vstack([rows_to_duplicate, row])

    # Duplicate rows
    duplicated_rows = sp.vstack([rows_to_duplicate])
    if num_rows_to_duplicate >= 2:
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
    return X_train_upsampled, y_train_upsampled

X_train_upsampled, y_train_upsampled = upsampling(X_train=X_train, y_train=y_train)
# Assuming X_train, X_test, y_train, y_test are your training and testing data

# Initialize the MLPClassifier
mlp_clf = MLPClassifier(verbose=True) #maxiter for interactive

# Train the model
history = mlp_clf.fit(X_train_upsampled, y_train_upsampled.flatten())

# Predictions on the test set
y_pred = mlp_clf.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


# Plot training loss and validation loss
train_loss = mlp_clf.loss_curve_
#val_loss = history['val_loss']

epochs = np.arange(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, label='Training Loss')
#plt.plot(epochs, val_loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig("Losses_sklearn.png")
print("######TRAIN VAL LOSS PLOT DONE######")

predictions = y_pred.round().astype(int).tolist()  # Converting array to list of integers
true_labels = y_test.astype(int).tolist()  # Converting array to list of integers

confusion_matrix = metrics.confusion_matrix(true_labels, predictions)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.savefig("Confusion_Matrix_sklearn.png")
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

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.savefig("ROC_AUC_sklearn.png")
print("######ROC-AUC PLOT DONE######")
