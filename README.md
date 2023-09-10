# Titanic_Classification
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset with relevant features including socio-economic status, age, gender, etc.
# Replace 'data.csv' with the path to your dataset file.
data = pd.read_csv('data.csv')

# Define your target variable (whether a person is safe or not)
# Replace 'safe_column_name' with the actual column name in your dataset.
target = data['safe_column_name']

# Select relevant features (socio-economic status, age, gender, etc.)
# Replace 'feature_columns' with the actual feature column names in your dataset.
features = data[['socioeconomic_status', 'age', 'gender', ...]]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create a machine learning model (Random Forest in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model on the training data
model.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Get a classification report for more details



