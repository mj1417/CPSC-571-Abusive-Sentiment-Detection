import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load preprocessed data
df1 = pd.read_csv("processed_training_dataset.csv")
df2 = pd.read_csv("processed_testing_dataset.csv")

# Handling NaN values
df1['processed_comment'] = df1['processed_comment'].fillna('')
df2['processed_comment'] = df2['processed_comment'].fillna('')

# Convert labels to integers if needed
df1['label'] = df1['label'].astype(int)
df2['label'] = df2['label'].astype(int)

# Vectorize text data
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(df1['processed_comment'])
y_train = df1['label']
X_test = vectorizer.transform(df2['processed_comment'])
y_test = df2['label']

# Train Naive Bayes classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)

# Predict on test data
y_pred = nb_model.predict(X_test)

# Evaluate performance
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='binary')
recall = recall_score(y_test, y_pred, average='binary')
f1 = f1_score(y_test, y_pred, average='binary')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Create a DataFrame with the predictions
predictions_df = pd.DataFrame({
    'original_comment': df2['comment'],  # Original comment
    'processed_comment': df2['processed_comment'],  # Processed comment
    'actual_label': y_test,  # Actual label from the dataset
    'predicted_label': y_pred  # Predicted label by the model
})

# Save predictions to a new CSV file
predictions_df.to_csv("predicted_NaiveBayes.csv", index=False)
print("Predictions have been saved")
