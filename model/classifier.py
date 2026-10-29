"""
Text Sentiment Classifier for SupportSense.

Uses a scikit-learn Pipeline with TfidfVectorizer and LogisticRegression.
Provides prediction, confidence probabilities, and top predictive n-gram analysis.
"""

import os
import joblib
from typing import Dict, Any, List, Tuple
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_PATH = os.path.join(MODEL_DIR, "sentiment_model.joblib")


class SupportSenseClassifier:
    """Wrapper around a TF-IDF + Logistic Regression classification pipeline."""

    def __init__(self, model_path: str = DEFAULT_MODEL_PATH):
        self.model_path = model_path
        self.pipeline: Pipeline = None
        self.classes_: List[str] = []

    def build_pipeline(self) -> Pipeline:
        """Constructs an un-fitted TF-IDF + Logistic Regression pipeline."""
        return Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        sublinear_tf=True,
                        strip_accents="unicode",
                        min_df=1,
                    ),
                ),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1500,
                        random_state=42,
                        C=3.0,
                    ),
                ),
            ]
        )

    def fit(self, texts: List[str], labels: List[str]) -> "SupportSenseClassifier":
        """Fits the pipeline on provided training texts and labels."""
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(texts, labels)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def save(self, path: str = None) -> str:
        """Serializes the fitted pipeline to disk."""
        target_path = path or self.model_path
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        joblib.dump(self.pipeline, target_path)
        return target_path

    def load(self, path: str = None) -> "SupportSenseClassifier":
        """Loads a pre-fitted pipeline from disk."""
        target_path = path or self.model_path
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Trained model not found at '{target_path}'. Please run train.py first.")
        self.pipeline = joblib.load(target_path)
        self.classes_ = list(self.pipeline.named_steps["clf"].classes_)
        return self

    def is_loaded(self) -> bool:
        """Checks if the classifier has an active fitted pipeline."""
        return self.pipeline is not None

    def predict(self, text: str) -> str:
        """Predicts the sentiment class ('negative', 'neutral', 'positive')."""
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call load() or fit() first.")
        return str(self.pipeline.predict([text])[0])

    def predict_proba(self, text: str) -> Dict[str, float]:
        """Returns a probability distribution across all sentiment classes."""
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call load() or fit() first.")
        probs = self.pipeline.predict_proba([text])[0]
        return {cls_name: round(float(prob), 4) for cls_name, prob in zip(self.classes_, probs)}

    def get_top_features(self, text: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Extracts the highest TF-IDF weighted terms from the input text to show
        users which words or phrases influenced the model's feature representation.
        """
        if not self.is_loaded():
            return []
        try:
            vectorizer: TfidfVectorizer = self.pipeline.named_steps["tfidf"]
            transformed = vectorizer.transform([text])
            feature_names = vectorizer.get_feature_names_out()
            row = transformed.tocoo()
            items = [(feature_names[col], float(val)) for col, val in zip(row.col, row.data)]
            items.sort(key=lambda x: x[1], reverse=True)
            return items[:top_k]
        except Exception:
            return []

    def classify(self, text: str) -> Dict[str, Any]:
        """
        Full inference endpoint helper.
        Returns predicted label, highest confidence score, all probabilities,
        and influential TF-IDF n-grams found in the input.
        """
        probabilities = self.predict_proba(text)
        predicted_sentiment = max(probabilities.items(), key=lambda kv: kv[1])[0]
        confidence = probabilities[predicted_sentiment]
        top_features = self.get_top_features(text, top_k=5)

        return {
            "sentiment": predicted_sentiment,
            "confidence": confidence,
            "probabilities": probabilities,
            "top_features": [feat for feat, _ in top_features],
        }
