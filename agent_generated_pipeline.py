import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset

# 1. Load the dataset
# The dataset is available on Hugging Face Hub under 'scikit-learn/breast-cancer-wisconsin'
dataset = load_dataset("scikit-learn/breast-cancer-wisconsin")

# Convert the dataset to a pandas DataFrame
# The 'train' split contains all the data
df = pd.DataFrame(dataset['train'])

# 2. Drop any completely empty structural columns like 'Unnamed: 32'
# The dataset info indicates 'Unnamed: 32' has 569 nulls out of 569 samples.
df = df.dropna(axis=1, how='all')

# Drop the 'id' column as it's an identifier and not a feature
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# 3. Set target 'y' to the 'diagnosis' column
# The 'diagnosis' column might be 'M'/'B' or already encoded.
# Let's ensure it's numeric (0 for Benign, 1 for Malignant)
if df['diagnosis'].dtype == 'object':
    le = LabelEncoder()
    df['diagnosis'] = le.fit_transform(df['diagnosis']) # M -> 1, B -> 0 typically

y = df['diagnosis']

# 4. Set features 'X' by dropping the 'diagnosis' column, keeping only numeric fields.
X = df.drop('diagnosis', axis=1)

# Ensure all feature columns are numeric.
# This step is generally good practice, though the dataset description implies they are.
X = X.select_dtypes(include=['number'])

# 5. Deal with any remaining missing data safely using X = X.fillna(X.mean())
# Based on the dataset info, there are no missing values in the feature columns,
# but this step ensures robustness.
X = X.fillna(X.mean(numeric_only=True))

# 6. Split data 80/20
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 7. Scale using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Train a LogisticRegression model
model = LogisticRegression(random_state=42, solver='liblinear') # 'liblinear' is good for small datasets
model.fit(X_train_scaled, y_train)

# 9. Print out final model accuracy score and classification_report
y_pred = model.predict(X_test_scaled)

print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))