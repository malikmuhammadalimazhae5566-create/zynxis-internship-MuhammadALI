from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


class SentimentModel:
    def __init__(self):
        self.vectorizer = None
        self.model = LogisticRegression(max_iter=1000)

    def fit(self, X_train, y_train):
        from src.vectorization import build_vectorizer

        self.vectorizer = build_vectorizer()
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_train_vec, y_train)
        return X_train_vec

    def predict(self, texts):
        if self.vectorizer is None:
            raise ValueError("The model must be fitted before prediction.")
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec)

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)
        return accuracy, report
