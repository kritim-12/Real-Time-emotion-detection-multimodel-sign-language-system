import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the extracted CSV
df = pd.read_csv('asl_landmarks.csv')

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Encode labels (e.g., 'A', 'B', ... -> 0, 1, ...)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train a random forest model
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)

# Check accuracy
accuracy = clf.score(X_test, y_test)
print(f"✅ Model trained! Accuracy: {accuracy * 100:.2f}%")

# Save model and encoder
joblib.dump(clf, 'asl_model.pkl')
joblib.dump(le, 'label_encoder.pkl')
print("✅ Model and label encoder saved.")
