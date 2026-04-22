import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA
# Using r'' to avoid the unicode error 
file_path = r'C:\Users\HP\OneDrive\Desktop\Sentiment_project\amazon_cells_labelled.txt'

# sep='\t' is crucial because your file uses tabs 
df = pd.read_csv(file_path, sep='\t', names=['review', 'sentiment'], header=None)

print(f"Dataset Loaded: {df.shape[0]} reviews found.")

# 2. DEFINE THE PIPELINE
# This combines the vectorizer and model into a single object
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('classifier', LogisticRegression())
])

# 3. TRAIN-TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['sentiment'], test_size=0.2, random_state=42
)

# 4. TRAIN THE MODEL
print("Training model...")
pipeline.fit(X_train, y_train)

# 5. EVALUATE
y_pred = pipeline.predict(X_test)
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 6. SAVE FOR DEPLOYMENT
# This saves the entire pipeline (including the TF-IDF logic)
joblib.dump(pipeline, 'sentiment_model.pkl')
print("\nPipeline saved as 'sentiment_model.pkl'")