import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

import seaborn as sns
# Load the dataset (replace 'your_dataset.csv' with your actual file)
data = pd.read_csv(r"C:\Users\Dhruv Sawant\Documents\Churn_Modelling.csv")

# Explore and understand the data
print(data.info())
print(data.describe())
# Split the data into features (X) and target variable (y)
X = data.drop(columns=['Exited'])
y = data['Exited']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Identify non-numeric columns
non_numeric_columns = X_train.select_dtypes(exclude=['number']).columns
# Encode both training and testing sets
X_train_encoded = pd.get_dummies(X_train, columns=non_numeric_columns)
X_test_encoded = pd.get_dummies(X_test, columns=non_numeric_columns)
# Align columns to make sure both sets have the same columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='outer', axis=1, fill_value=0)
# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)
# Example: Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
# Train the model
model.fit(X_train_scaled, y_train)
# Make predictions
y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]  # Probability estimates for the positive class
# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
classification_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy:.2f}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{classification_report}')
# Plot confusion matrix
plt.figure(figsize=(5, 5))
sns.heatmap(conf_matrix, annot=True, fmt=".0f", cmap="Blues", cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
# Plot ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
for i in range(20):
    print(f"Example {i + 1}:")
    print(f"Actual Label: {y_test.iloc[i]}, Predicted Label: {y_pred[i]}, Probability: {y_prob[i]:.4f}")
    print(f"Actual Data: {X_test.iloc[i]}\n")
