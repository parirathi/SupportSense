"""
SupportSense - Customer Feedback Classification & Priority Triage Dashboard.

Flask web application integrating TF-IDF + Logistic Regression sentiment
classification with a transparent, rule-based urgency triage system.
"""

import os
import json
import datetime
from flask import Flask, request, jsonify, render_template, session

from model.classifier import SupportSenseClassifier, DEFAULT_MODEL_PATH
from model.priority import evaluate_priority
from train import train_and_evaluate, METRICS_PATH, DATA_PATH


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supportsense-demo-secret-key-3.14")

# Configuration constants
MAX_INPUT_LENGTH = 2000
MIN_INPUT_LENGTH = 3
MAX_SESSION_HISTORY = 50

# Global Classifier instance
classifier = SupportSenseClassifier(DEFAULT_MODEL_PATH)


def ensure_model_ready():
    """Ensures model is trained and loaded before serving requests."""
    if not os.path.exists(DEFAULT_MODEL_PATH) or not os.path.exists(METRICS_PATH):
        print("Model or metrics not found on disk. Initiating training...")
        train_and_evaluate(DATA_PATH, DEFAULT_MODEL_PATH, METRICS_PATH)
    if not classifier.is_loaded():
        classifier.load(DEFAULT_MODEL_PATH)


# Initialize model on startup
with app.app_context():
    try:
        ensure_model_ready()
    except Exception as exc:
        print(f"Warning during model initialization: {exc}")


def get_metrics_data():
    """Loads recorded evaluation metrics from disk."""
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return None


@app.route("/")
def index():
    """Renders the main SupportSense dashboard."""
    metrics = get_metrics_data()
    history = session.get("feedback_history", [])
    return render_template("index.html", metrics=metrics, session_history=history)


@app.route("/api/analyze", methods=["POST"])
def analyze():
    """
    Analyzes submitted feedback text:
    - Validates input format and length
    - Classifies sentiment with TF-IDF + Logistic Regression
    - Evaluates priority via transparent urgency heuristic rules
    - Stores the item in session history (non-persistent)
    """
    if not request.is_json:
        return (
            jsonify({"error": "Invalid request: Content-Type must be 'application/json'."}),
            400,
        )

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return (
            jsonify({"error": "Missing required field 'text' in JSON payload."}),
            400,
        )

    raw_text = data.get("text")
    if not isinstance(raw_text, str):
        return (
            jsonify({"error": "Input 'text' must be a valid string."}),
            400,
        )

    text = raw_text.strip()

    # Input validation
    if len(text) < MIN_INPUT_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Feedback message is too short. Please enter at least {MIN_INPUT_LENGTH} characters."
                }
            ),
            400,
        )

    if len(text) > MAX_INPUT_LENGTH:
        return (
            jsonify(
                {
                    "error": f"Feedback message exceeds maximum limit of {MAX_INPUT_LENGTH} characters (provided {len(text)})."
                }
            ),
            400,
        )

    # Inference execution
    try:
        if not classifier.is_loaded():
            ensure_model_ready()

        analysis = classifier.classify(text)
        sentiment = analysis["sentiment"]
        confidence = analysis["confidence"]
        probabilities = analysis["probabilities"]
        top_features = analysis["top_features"]

        # Transparent priority evaluation
        priority_info = evaluate_priority(text, sentiment, confidence)

        # Prepare response record
        record = {
            "id": f"fb-{datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')[:17]}",
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": text,
            "sentiment": sentiment,
            "confidence": confidence,
            "probabilities": probabilities,
            "top_features": top_features,
            "priority": priority_info["priority"],
            "is_urgent": priority_info["is_urgent"],
            "reasons": priority_info["reasons"],
            "matched_keywords": priority_info["matched_keywords"],
            "rule_type": priority_info["rule_type"],
            "suggested_action": priority_info["suggested_action"],
        }

        # Update session history (capped)
        history = session.get("feedback_history", [])
        history.insert(0, record)
        if len(history) > MAX_SESSION_HISTORY:
            history = history[:MAX_SESSION_HISTORY]
        session["feedback_history"] = history
        session.modified = True

        return jsonify({"status": "success", "result": record, "session_total": len(history)}), 200

    except Exception as exc:
        return (
            jsonify(
                {
                    "error": "An internal error occurred during prediction analysis.",
                    "details": str(exc),
                }
            ),
            500,
        )


@app.route("/api/clear-session", methods=["POST"])
def clear_session():
    """Clears non-persistent session feedback history."""
    session["feedback_history"] = []
    session.modified = True
    return jsonify({"status": "success", "message": "Session history cleared successfully."}), 200


@app.route("/api/metrics", methods=["GET"])
def get_metrics():
    """Returns stored test metrics and evaluation details."""
    metrics = get_metrics_data()
    if not metrics:
        return jsonify({"error": "Metrics record not found."}), 404
    return jsonify(metrics), 200


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return (
        jsonify(
            {
                "status": "healthy",
                "app": "SupportSense",
                "model_loaded": classifier.is_loaded(),
            }
        ),
        200,
    )


if __name__ == "__main__":
    # Standard development server for local testing
    app.run(host="127.0.0.1", port=5000, debug=False)
