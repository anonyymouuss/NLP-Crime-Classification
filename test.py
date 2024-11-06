import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Step 1: Load the full raw CSV data
data = pd.read_csv('D:/WEB/processed_data.csv')  # Provide the correct path to your file

# Step 2: Display the first few rows of data for verification (optional)
print("First few rows of the raw data:")
print(data.head())

# Step 3: Fill missing values in 'crimeaditionalinfo' column (if any)
data['crimeaditionalinfo'].fillna('Missing Info', inplace=True)

# Step 4: Check and display any remaining NaN values (optional)
print("Number of NaN values in 'crimeaditionalinfo' column after filling missing values:")
print(data['crimeaditionalinfo'].isnull().sum())

# Step 5: Define features (X) and target (y)
X = data['crimeaditionalinfo']  # Feature column containing the raw text
y = data['category']  # Target column for classification

# Step 6: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Convert text data to numerical features using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)  # Limit to 5000 features to prevent overfitting
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Step 8: Train a machine learning model (Logistic Regression as an example)
model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Step 9: Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Step 10: Generate classification report and accuracy
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Calculate overall accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Step 11: (Optional) If you want to predict the categories for the entire dataset
all_predictions = model.predict(vectorizer.transform(data['crimeaditionalinfo']))

# Add the predictions as a new column in the original dataframe
data['Predicted Category'] = all_predictions

# Step 12: Save the processed data with predictions to a new CSV or Excel file
output_file = 'processed_categorized_data1(test).csv'  # You can change the file format to .xlsx if you prefer
data.to_csv(output_file, index=False)

# Step 13: (Optional) To save as Excel, uncomment the following line:
# data.to_excel('processed_categorized_data.xlsx', index=False)

# Step 14: Display a sample of the processed data (for verification)
print("Processed data with predictions (saved in 'processed_categorized_data.csv'):")
print(data[['crimeaditionalinfo', 'category', 'Predicted Category']].head())

print(f"Full dataset has been saved as {output_file}.")
