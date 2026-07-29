import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

def run_everything():
    print("1. Downloading breast cancer dataset...")
    dataset = load_dataset("scikit-learn/breast-cancer-wisconsin")
    df = pd.DataFrame(dataset['train'])
    
    print("2. Preprocessing and cleaning data...")
    # Separate our target label
    y = df['diagnosis']
    
    # Get features by dropping the target column
    X = df.drop(columns=['diagnosis'])
    
    # 1. Drop columns that are completely empty (fixes the Unnamed: 32 issue)
    X = X.dropna(axis=1, how='all')
    
    # 2. Only keep numeric columns (ignores patient IDs or text columns)
    X = X.select_dtypes(include=[np.number])
    
    # 3. Fill any remaining random missing values with the column average
    X = X.fillna(X.mean())

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("3. Training the prediction model...")
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    print("4. Evaluating results...")
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)

    print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

if __name__ == "__main__":
    run_everything()
