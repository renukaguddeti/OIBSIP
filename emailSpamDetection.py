import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('email_detection.csv', usecols=['v1', 'v2'], encoding='latin-1')
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})  # Encode labels

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='label', hue='label', palette='viridis', dodge=False, legend=False)
plt.title("Distribution of Spam and Non-Spam Emails")
plt.xticks(ticks=[0, 1], labels=['Ham', 'Spam'])
plt.ylabel('Count')
plt.xlabel('Label')
plt.show()

X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(stop_words='english', max_features=3000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.title("Confusion Matrix")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

spam_words = X_train[y_train == 1]  
ham_words = X_train[y_train == 0] 

feature_names = vectorizer.get_feature_names_out()
spam_coefficients = model.feature_log_prob_[1]
ham_coefficients = model.feature_log_prob_[0]

importance_df = pd.DataFrame({
    'word': feature_names,
    'spam_score': spam_coefficients,
    'ham_score': ham_coefficients
})

top_spam_words = importance_df.nlargest(10, 'spam_score')
top_ham_words = importance_df.nlargest(10, 'ham_score')
