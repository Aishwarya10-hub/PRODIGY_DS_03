# Step 1: Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import io

# Step 2: Load the dataset from UCI repo (ZIP)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank.zip"
response = urllib.request.urlopen(url)
zip_file = zipfile.ZipFile(io.BytesIO(response.read()))

# Step 3: Read 'bank.csv' from inside the ZIP file
with zip_file.open('bank.csv') as file:
    data = pd.read_csv(file, sep=';')

# Step 4: Preprocess data
# Convert categorical variables to dummy/indicator variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Step 5: Split into features and target
X = data_encoded.drop('y_yes', axis=1)
y = data_encoded['y_yes']

# Step 6: Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Train the Decision Tree model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = clf.predict(X_test)

# Step 9: Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 10: Plot the Decision Tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=["No", "Yes"], filled=True, max_depth=3)
plt.show()
