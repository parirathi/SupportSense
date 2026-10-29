"""
Tests for SupportSense Sentiment Classifier Pipeline.
"""

import pytest
from model.classifier import SupportSenseClassifier, DEFAULT_MODEL_PATH


@pytest.fixture(scope="module")
def classifier():
    """Loads pre-trained model classifier."""
    clf = SupportSenseClassifier(DEFAULT_MODEL_PATH)
    clf.load()
    return clf


def test_model_is_loaded(classifier):
    """Classifier should be properly loaded with valid classes."""
    assert classifier.is_loaded() is True
    assert set(classifier.classes_) == {"negative", "neutral", "positive"}


def test_prediction_output_format(classifier):
    """Classifier should predict valid labels."""
    prediction = classifier.predict("This product is completely broken and useless!")
    assert prediction in {"negative", "neutral", "positive"}


def test_probability_distribution(classifier):
    """Probabilities must sum to approximately 1.0 across all 3 classes."""
    probs = classifier.predict_proba("Could you please tell me how to export customer reports?")
    assert isinstance(probs, dict)
    assert set(probs.keys()) == {"negative", "neutral", "positive"}
    total_prob = sum(probs.values())
    assert pytest.approx(total_prob, 0.01) == 1.0


def test_classify_endpoint_helper(classifier):
    """Classify method should return sentiment, confidence, probabilities, and top features."""
    result = classifier.classify("Thank you Sarah, your support was fast, wonderful and fantastic!")
    assert "sentiment" in result
    assert "confidence" in result
    assert "probabilities" in result
    assert "top_features" in result
    assert result["sentiment"] == "positive"
    assert result["confidence"] > 0.40
    assert isinstance(result["top_features"], list)


def test_unfitted_model_raises_runtime_error():
    """An un-fitted, unloaded classifier must raise RuntimeError on predict."""
    fresh_clf = SupportSenseClassifier("non_existent_path.joblib")
    with pytest.raises(RuntimeError):
        fresh_clf.predict("Testing text")
