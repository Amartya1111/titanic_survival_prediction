import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
file_path = 'path_to_your_downloaded_file.csv'  # Change this to your file path
titanic_df = pd.read_csv(file_path)

# Data Preprocessing

# Handle missing values
# Fill missing Age values with the median age
imputer = SimpleImputer(strategy='median')
titanic_df['Age'] = imputer.fit_transform(titanic_df[['Age']])

# Fill missing Embarked values with the mode (most common value)
titanic_df['Embarked'].fillna(titanic_df['Embarked'].mode()[0], inplace=True)

# Fill missing Fare values with the median fare (if any)
titanic_df['Fare'] = imputer.fit_transform(titanic_df[['Fare']])

# Convert categorical features to numerical values
label_encoder = LabelEncoder()
titanic_df['Sex'] = label_encoder.fit_transform(titanic_df['Sex'])
titanic_df['Embarked'] = label_encoder.fit_transform(titanic_df['Embarked'])

# Drop unnecessary columns
titanic_df.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'], inplace=True)

# Split the data into features and target variable
X = titanic_df.drop(columns=['Survived'])
y = titanic_df['Survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Building

# Train a Logistic Regression model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
