import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Sample dataset
data = {
    "label": [
        "ham", "ham", "ham", "ham", "ham", "ham", "ham",
        "spam", "spam", "spam", "spam", "spam", "spam", "spam"
    ],
    "text": [
        "Hey, how are you doing?",
        "Let's meet at the cafe tomorrow.",
        "Can you send the project files?",
        "Are we still on for the movie?",
        "I’ll call you in the evening.",
        "Don't forget to bring your notebook.",
        "Meeting has been rescheduled to 3 PM.",
        "Win a brand new car by clicking here!",
        "Limited offer: Get cheap meds now!",
        "Congratulations! You've won a prize.",
        "Claim your free gift card today!",
        "You have been selected for a lottery.",
        "Get rich quick with this scheme!",
        "Act now to secure your exclusive deal!"
    ]
}


df = pd.DataFrame(data)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.3, stratify=df["label"], random_state=42
)

# Convert text to feature vectors (Bag of Words)
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Predict on test data
y_pred = model.predict(X_test_vectorized)

# Evaluate the model
print(classification_report(y_test, y_pred))

def predict_spam(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Test with some new examples
# Test with some new examples
while True:
    user_input = input("Enter an email text (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    prediction = predict_spam(user_input)
    print(f"Prediction: {prediction}\n")


for email in new_emails:
    print(f"Text: {email}")
    print(f"Prediction: {predict_spam(email)}\n")


for email in new_emails:
    print(f"Text: {email}")
    print(f"Prediction: {predict_spam(email)}\n")
