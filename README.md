# SupportSense 🛡️
> A complete, beginner-friendly web application to help customer support teams classify incoming customer feedback and identify messages needing immediate human review.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-black.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)](https://scikit-learn.org/)
[![pytest](https://img.shields.io/badge/pytest-Passing-green.svg)](https://pytest.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents
1. [Overview & Purpose](#overview--purpose)
2. [Architecture & Design](#architecture--design)
3. [Model Training & Real Test Evaluation](#model-training--real-test-evaluation)
4. [Transparent Priority Triage Rule](#transparent-priority-triage-rule)
5. [Dataset & Known Limitations](#dataset--known-limitations)
6. [Getting Started on Windows](#getting-started-on-windows)
7. [Running Automated Tests](#running-automated-tests)
8. [API Endpoints](#api-endpoints)
9. [Project Structure](#project-structure)
10. [Third-Party Attribution & License](#third-party-attribution--license)

---

## Overview & Purpose

Support desks receive hundreds of incoming inquiries, bug reports, and billing tickets every day. **SupportSense** helps triage this volume by:
1. **Classifying Sentiment:** Categorizing incoming feedback as `Negative`, `Neutral`, or `Positive` using an interpretable scikit-learn pipeline (`TfidfVectorizer` + `LogisticRegression`).
2. **Transparent Priority Flagging:** Applying explicit, rule-based urgency heuristics to immediately flag critical tickets (outages, fraud, legal threats, cancellations) for human escalation without pretending to be a complex black box.
3. **Session Triage Dashboard:** Providing support agents with a responsive UI to analyze customer messages, view prediction confidence, inspect probability distributions, and review submitted items within the current session.

---

## Architecture & Design

```
+-------------------------------------------------------------------------+
|                        Browser Client (HTML / CSS / JS)                 |
|  - Real-time character count & client validation (<3 or >2000 chars)     |
|  - Demonstration chips, sentiment badges, probability tracks            |
|  - Non-persistent session triage feed with human escalation guidance    |
+------------------------------------+------------------------------------+
                                     | JSON REST API
                                     v
+------------------------------------+------------------------------------+
|                       Flask Application (app.py)                        |
|  - Payload validation (Content-Type, string type, boundary checks)      |
|  - Non-persistent Flask session memory for submitted feedback           |
+------------------+----------------------------------+-------------------+
                   |                                  |
                   v                                  v
+----------------------------------+ +------------------------------------+
|   Sentiment Classifier Pipeline   | |   Transparent Priority Engine      |
|   (model/classifier.py)          | |   (model/priority.py)              |
|                                  | |                                    |
|  - TfidfVectorizer (ngrams 1-2,  | |  - Tier 1: Severe outages, fraud,  |
|    sublinear TF, strip accents)  | |    legal threats (Unconditional)   |
|  - LogisticRegression            | |  - Tier 2: Escalations & deadlines |
|    (class_weight='balanced')     | |  - Tier 3: Churn & dissatisfaction |
|  - Top N-gram feature extraction | |  - Full human-auditable reasons    |
+----------------------------------+ +------------------------------------+
```

---

## Model Training & Real Test Evaluation

The classifier is trained using an **80/20 stratified train/test split** on a curated, domain-specific demonstration dataset (`data/support_feedback.csv`).

> [!IMPORTANT]
> **No Fabricated Metrics:** The metrics below were evaluated **strictly on the 20% held-out test set** (30 unseen customer tickets), never on training data.

### Actual Held-Out Test Set Metrics (`data/metrics.json`):
* **Held-out Test Accuracy:** **73.33%**
* **Macro Precision:** **73.33%**
* **Macro Recall:** **73.33%**
* **Macro F1-Score:** **73.33%**
* **Weighted F1-Score:** **73.33%**

### Per-Class Held-Out Breakdown:
| Class | Precision | Recall | F1-Score | Unseen Test Support |
| :--- | :---: | :---: | :---: | :---: |
| **Negative** | 80.0% | 80.0% | 80.0% | 10 |
| **Neutral**  | 80.0% | 80.0% | 80.0% | 10 |
| **Positive** | 60.0% | 60.0% | 60.0% | 10 |

To retrain the model and regenerate metrics at any time:
```powershell
python train.py
```

---

## Transparent Priority Triage Rule

In high-stakes support environments, unexplainable black-box urgency models often miss time-critical incidents or misguide operators. **SupportSense uses an explicit, transparent rule-based heuristic** for urgency triage:

* **Tier 1 (Critical - Unconditional):** Mentions of system outages (`outage`, `server down`, `production broken`), data loss (`corrupted`, `breach`), or legal/payment disputes (`lawyer`, `lawsuit`, `fraud`, `chargeback`, `unauthorized`) **always trigger Urgent status**.
* **Tier 2 (Escalation Demands):** Phrases like `urgent`, `urgently`, `asap`, `immediate`, `deadline`, or `escalate` trigger **Urgent** when combined with negative or neutral sentiment.
* **Tier 3 (Customer Churn Risk):** Threats to `cancel subscription`, `cancel renewal`, `demanding refund`, or `leaving for a competitor` combined with negative sentiment trigger **Urgent**.
* **Audit Trail:** Every evaluation outputs the exact list of triggered keywords and rules, accompanied by an actionable routing recommendation (e.g., *"Route to Tier-2 priority queue for immediate review within 15 minutes"*).

---

## Dataset & Known Limitations

1. **Compact Demonstration Dataset:** The dataset contains 150 balanced customer support messages across negative, neutral, and positive classes. While representative of common SaaS and customer support workflows, a production system would benefit from thousands of domain-specific historical tickets.
2. **Domain Specificity:** The vocabulary is centered on software tools, billing, features, outages, and onboarding. Off-topic, colloquial, or poetic text may lead to uncertain probability spreads.
3. **Sarcasm and Indirect Phrasing:** Linear models relying on n-gram TF-IDF can miss complex rhetorical irony (e.g., *"Oh brilliant, another server reboot right during our launch!"*).
4. **Heuristic Urgency:** The urgency flag is an auditable rule engine, **not** a supervised urgency prediction model.

---

## Getting Started on Windows

### Prerequisites
* Windows 10 or 11
* Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed with `Add Python to PATH` checked.

### 1. Open Terminal
Open **PowerShell** or **Command Prompt** and navigate to the project directory:
```powershell
cd c:\Users\Dell\Desktop\SupportSense
```

### 2. (Optional) Create & Activate a Virtual Environment
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If PowerShell restricts script execution, run: `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`)*

### 3. Install Dependencies
```powershell
pip install -r requirements.txt
```

### 4. Train the Model (Optional - Pre-trained Artifact Included)
```powershell
python train.py
```

### 5. Launch the Web Application
```powershell
python app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000/
```

---

## Running Automated Tests

SupportSense comes with an automated test suite covering model inference, input validation boundary conditions, and priority rules.

Run all tests via pytest:
```powershell
python -m pytest -v
```

### Expected Output:
```
============================= test session starts =============================
collected 22 items

tests/test_api.py::test_index_route PASSED                               [  4%]
tests/test_api.py::test_health_endpoint PASSED                           [  9%]
tests/test_api.py::test_metrics_endpoint PASSED                          [ 13%]
tests/test_api.py::test_analyze_valid_input PASSED                       [ 18%]
tests/test_api.py::test_analyze_empty_input_rejected PASSED              [ 22%]
tests/test_api.py::test_analyze_too_short_input PASSED                   [ 27%]
tests/test_api.py::test_analyze_excessive_length_rejected PASSED         [ 31%]
tests/test_api.py::test_analyze_non_string_type_rejected PASSED          [ 36%]
tests/test_api.py::test_analyze_missing_field_rejected PASSED            [ 40%]
tests/test_api.py::test_analyze_non_json_payload_rejected PASSED         [ 45%]
tests/test_api.py::test_clear_session_route PASSED                       [ 50%]
tests/test_model.py::test_model_is_loaded PASSED                         [ 54%]
tests/test_model.py::test_prediction_output_format PASSED                [ 59%]
tests/test_model.py::test_probability_distribution PASSED                [ 63%]
tests/test_model.py::test_classify_endpoint_helper PASSED                [ 68%]
tests/test_model.py::test_unfitted_model_raises_runtime_error PASSED     [ 72%]
tests/test_priority.py::test_critical_outage_keyword_triggers_urgent PASSED [ 77%]
tests/test_priority.py::test_legal_and_fraud_keywords_trigger_urgent PASSED [ 81%]
tests/test_priority.py::test_escalation_keyword_with_neutral_triggers_urgent PASSED [ 86%]
tests/test_priority.py::test_normal_support_inquiry_triggers_normal PASSED [ 90%]
tests/test_priority.py::test_positive_praise_triggers_normal PASSED      [ 95%]
tests/test_priority.py::test_rule_transparency_metadata PASSED           [100%]

============================= 22 passed in 7.03s ==============================
```

---

## API Endpoints

### 1. `POST /api/analyze`
Submits text for sentiment classification and urgency evaluation.

**Request Body:**
```json
{
  "text": "Our payment was charged twice and the site is broken! Urgent assistance needed ASAP."
}
```

**Response (HTTP 200):**
```json
{
  "status": "success",
  "result": {
    "id": "fb-20261029153012",
    "timestamp": "2026-10-29 15:30:12",
    "text": "Our payment was charged twice and the site is broken! Urgent assistance needed ASAP.",
    "sentiment": "negative",
    "confidence": 0.8124,
    "probabilities": {
      "negative": 0.8124,
      "neutral": 0.1142,
      "positive": 0.0734
    },
    "top_features": ["broken", "urgent", "charged", "payment"],
    "priority": "Urgent",
    "is_urgent": true,
    "reasons": [
      "Contains explicit escalation/time-sensitive keyword(s): 'urgent', 'asap'",
      "High-confidence negative sentiment (81.2%) detected by classifier"
    ],
    "matched_keywords": ["asap", "urgent"],
    "rule_type": "Transparent Heuristic Rules (Not an ML Urgency Model)",
    "suggested_action": "Escalate to Human Agent: Route to Tier-2 priority queue for immediate review within 15 minutes."
  },
  "session_total": 1
}
```

### 2. `GET /api/metrics`
Returns current held-out test evaluation scores and limitations.

### 3. `POST /api/clear-session`
Clears non-persistent session feedback history.

### 4. `GET /health`
Returns service and model health status (`{"status": "healthy", "model_loaded": true}`).

---

## Project Structure

```
SupportSense/
├── app.py                     # Flask web app and API routing
├── train.py                   # Model training, test evaluation, and metrics generation
├── requirements.txt           # Python package dependencies
├── .gitignore                 # Version control exclusions
├── LICENSE                    # MIT License and third-party attributions
├── README.md                  # Comprehensive documentation and Windows guide
├── data/
│   ├── support_feedback.csv   # Labeled demonstration dataset (150 tickets)
│   └── metrics.json           # Evaluation metrics on 20% held-out test set
├── model/
│   ├── __init__.py
│   ├── classifier.py          # TF-IDF + Logistic Regression pipeline wrapper
│   ├── priority.py            # Transparent urgency heuristic rule engine
│   └── sentiment_model.joblib # Trained serialized scikit-learn pipeline
├── static/
│   ├── css/
│   │   └── style.css          # Modern, responsive UI stylesheet
│   └── js/
│       └── app.js             # Client-side validation, fetch API, live triage rendering
├── templates/
│   └── index.html             # Dashboard UI template
└── tests/
    ├── __init__.py
    ├── test_api.py            # Route and validation unit tests
    ├── test_model.py          # Pipeline and classification tests
    └── test_priority.py       # Heuristic urgency rule tests
```

---

## Third-Party Attribution & License

SupportSense is licensed under the [MIT License](LICENSE).

### Third-Party Materials:
* **Flask** - BSD-3-Clause License - Pallets Projects
* **scikit-learn** - BSD-3-Clause License - scikit-learn developers
* **pandas** - BSD-3-Clause License - PyData Development Team
* **NumPy** - BSD-3-Clause License - NumPy Developers
* **Joblib** - BSD-3-Clause License - Joblib developers
* **pytest** - MIT License - Holger Krekel and contributors
* **Inter Font** - SIL Open Font License 1.1 - The Inter Project Authors
