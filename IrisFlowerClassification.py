import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
from sklearn.preprocessing import LabelEncoder

# Load the Iris dataset directly from Scikit-learn
iris = load_iris()
data = pd.DataFrame(data=iris.data, columns=iris.feature_names)

# Add the target column (species)
data['Species'] = iris.target

# Display the first few rows of the dataset
print(data.head())

# Encode the 'Species' column (though it's already numeric in this dataset)
encoder = LabelEncoder()
data['Species'] = encoder.fit_transform(data['Species'])

# Split the data into features (X) and target (y)
X = data.drop(columns=['Species'])  # Features: all columns except 'Species'
y = data['Species']                 # Target: 'Species'

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the trained model (optional)
import joblib
joblib.dump(model, "iris_model.pkl")

# Decode the numeric labels back to species names (if needed)
species_labels = encoder.inverse_transform(range(len(encoder.classes_)))
print("\nSpecies Labels Mapping:")
for idx, label in enumerate(species_labels):
    print(f"{idx}: {label}")
