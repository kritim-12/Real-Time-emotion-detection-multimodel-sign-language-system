import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset with geometric features
df = pd.read_csv('asl_landmarks_geometric.csv')

# Separate features and labels
X = df.drop('label', axis=1)
y = df['label']

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train classifier
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
accuracy = clf.score(X_test, y_test)
print(f"✅ Model trained! Accuracy: {accuracy * 100:.2f}%")

# Save model and label encoder
joblib.dump(clf, 'asl_model_geometric.pkl')
joblib.dump(le, 'label_encoder_geometric.pkl')
print("💾 Model and encoder saved.")
