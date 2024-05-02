from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report


np.random.seed(42)

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
    'Cluster_y': float,
    "superstar_v1_x": float,
    "superstar_x": int
}
data = pd.read_csv("data_superstar_v1_0.csv", delimiter=",", dtype=dtype_dict, na_values=[''])
data['date'] = pd.to_datetime(data['release_date'])
data.sort_values(by="date", inplace=True)

# List of columns to keep
columns_to_keep = ['explicit', 'track_number', 'num_artists', 'num_available_markets', 'release_date',
                   'duration_ms', 'key', 'mode', 'time_signature', 'acousticness',
                   'danceability', 'energy', 'instrumentalness', 'liveness', 'loudness',
                   'speechiness', 'valence', 'tempo', 'years_on_charts', 'hit', "superstar_v1_x", "superstar_x", "date"]                              #Collaboration Profile == CLuster????

# Drop columns not in the list
data["explicit"] = data["explicit"].astype(int)
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
X_train_upsampled_with_y.drop(columns=["release_date"], inplace=True)

#print(X_train_upsampled_with_y.head())
#prepro:
y_train_upsampled_ordered = X_train_upsampled_with_y["hit"]
X_train_upsampled_ordered = X_train_upsampled_with_y.drop(columns="hit")

# Define data types for each column
dtype_dict = {
    'explicit': int,
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
    "superstar_x": int,
    "date": 'datetime64[ns]'
}

# Use astype method to cast columns to the specified data types
#print(X_test.columns)
print(X_test["release_date"].iloc[0])
print(X_test["release_date"].iloc[-1])
X_train_upsampled_ordered = X_train_upsampled_ordered.astype(dtype_dict)
X_test.drop(columns="release_date", inplace=True)
X_test = X_test.astype(dtype_dict)

y_train_upsampled_ordered_reshaped = y_train_upsampled_ordered.values.reshape(-1, 1)
y_test_reshaped = y_test.values.reshape(-1, 1)
y_test = y_test[:105500]
# print(y_test)
# print(y_test.shape)
# print(type(y_test))

sep_index = X_train_upsampled_ordered.shape[0]
concatenated_df = pd.concat([X_train_upsampled_ordered, X_test])
data_prepro = preprocess(concatenated_df, min_max_val, exclude_cols=["date"])              #richtig so?
X_train_upsampled_prepro = data_prepro[:sep_index]
X_test_prepro = data_prepro[sep_index: data_prepro.shape[0]-122257]

###EVTL MIN mAx SCALING AUF HIT (y)
## Create the scaler
#scaler = MinMaxScaler()
#
## Fit and transform the scaled array
#y_scaled = scaler.fit_transform(y_reshaped)
print("######PREPROCESSING DONE######")

# Initialize the MLPClassifier
mlp_clf = MLPClassifier(verbose=True, random_state=42)#, max_iter=10)#, shuffle=False, max_iter=5) #maxiter for interactive #shuffle False

# Train the model
history = mlp_clf.fit(X_train_upsampled_prepro, y_train_upsampled_ordered_reshaped.flatten())

# Predictions on the test set
y_pred = mlp_clf.predict(X_test_prepro) # nachsehen
#print(y_pred)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

y_pred_proba = mlp_clf.predict_proba(X_test_prepro)
# opt_thres = -1
# opt_prec = 0
# liste_thresh = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# true_labels = y_test.astype(int).tolist()
# predictions =[]
# #print(output.tolist())
# for i in liste_thresh:
#     predictions = list(map(lambda x: int(x >= i), y_pred_proba[:,1]))
#
#     precision = metrics.precision_score(true_labels, predictions)
#
#     # Recall
#     recall = metrics.recall_score(true_labels, predictions)
#     # F1-Score
#     f1 = metrics.f1_score(true_labels, predictions)
#     # ROC Curve and AUC
#     fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
#     roc_auc = metrics.auc(fpr, tpr)
#
#     #print("Precision:", precision)
#     #print("Recall:", recall)
#     #print("F1-Score:", f1)
#     #print("ROC AUC:", roc_auc)
#
#     if precision > opt_prec:
#         opt_thres = i
#         opt_prec = precision
# print(f"optimal threshold {opt_thres}, with precision {opt_prec}")

#
#confusion_matrix = metrics.confusion_matrix(true_labels, predictions)
#
## Extract TN, FP, TP values
#TN = confusion_matrix[0, 0]  # True Negatives
#FP = confusion_matrix[0, 1]  # False Positives
#FN = confusion_matrix[1, 0]  # False Negatives
#TP = confusion_matrix[1, 1]  # True Positives
#
## Print the results
#print("True Negatives (TN):", TN)
#print("False Positives (FP):", FP)
#print("False Negatives (FN):", FN)
#print("True Positives (TP):", TP)
#
#class_report = classification_report(y_test, predictions)
#print("Classification Report:\n", class_report)

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
plt.savefig("Losses_sklearn_collab.png")
print("######TRAIN VAL LOSS PLOT DONE######")

predictions = y_pred.round().astype(int).tolist()  # Converting array to list of integers
true_labels = y_test.astype(int).tolist()  # Converting array to list of integers

confusion_matrix = metrics.confusion_matrix(true_labels, predictions)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [False, True])

cm_display.plot()
plt.savefig("Confusion_Matrix_sklearn_collab.png")
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

fpr, tpr, thresholds = metrics.roc_curve(true_labels, predictions)
roc_auc = metrics.auc(fpr, tpr)

print("Precision:", precision)
print("Recall:", recall)
print("F1-Score:", f1)

y_pred_proba = mlp_clf.predict_proba(X_test_prepro)
#print(y_pred_proba)

fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred_proba[:,1])
roc_auc = metrics.auc(fpr, tpr)
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
plt.savefig("ROC_AUC_sklearn_collab_v2.png")
print("######ROC-AUC PLOT DONE######")


# Generate a classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

count_occ = y_test.value_counts(normalize=True)

# Calculate the weighted accuracy
weighted_acc = (np.sum((y_test == 1) == y_pred) * count_occ[0] + np.sum((y_test == 0) == y_pred) * count_occ[1]) / len(y_test)

print("Weighted Accuracy:", weighted_acc)

f1_scores = []
for label in np.unique(y_test):
    f1 = metrics.f1_score(y_test == label, y_pred == label)
    f1_scores.append(f1)

# Calculate the macro F1 score by taking the average
macro_f1 = np.mean(f1_scores)

print("Macro F1 Score:", macro_f1)
#%%
