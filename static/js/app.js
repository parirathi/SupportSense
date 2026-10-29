/**
 * SupportSense - Client-side Interactive Logic
 */

document.addEventListener("DOMContentLoaded", () => {
  const form = document.getElementById("analyze-form");
  const textarea = document.getElementById("feedback-input");
  const charCounter = document.getElementById("char-counter");
  const clearBtn = document.getElementById("clear-btn");
  const submitBtn = document.getElementById("submit-btn");
  const btnSpinner = submitBtn.querySelector(".btn-spinner");
  const btnText = submitBtn.querySelector(".btn-text");
  const validationError = document.getElementById("validation-error");

  const resultContainer = document.getElementById("result-container");
  const resultSentimentBadge = document.getElementById("result-sentiment-badge");
  const resultConfidenceText = document.getElementById("result-confidence-text");

  const probBarNeg = document.getElementById("prob-bar-negative");
  const probValNeg = document.getElementById("prob-val-negative");
  const probBarNeu = document.getElementById("prob-bar-neutral");
  const probValNeu = document.getElementById("prob-val-neutral");
  const probBarPos = document.getElementById("prob-bar-positive");
  const probValPos = document.getElementById("prob-val-positive");

  const topFeaturesWrapper = document.getElementById("top-features-wrapper");
  const topFeaturesList = document.getElementById("top-features-list");

  const priorityBanner = document.getElementById("priority-banner");
  const priorityIcon = document.getElementById("priority-icon");
  const priorityHeading = document.getElementById("priority-heading");
  const priorityPill = document.getElementById("priority-pill");
  const priorityReasonsList = document.getElementById("priority-reasons-list");
  const priorityActionText = document.getElementById("priority-action-text");

  const emptyHistory = document.getElementById("empty-history");
  const historyContainer = document.getElementById("history-items-container");
  const clearHistoryBtn = document.getElementById("clear-history-btn");

  const MAX_CHARS = 2000;
  const MIN_CHARS = 3;

  // 1. Character Counter & Input Listener
  function updateCharCount() {
    const len = textarea.value.length;
    charCounter.textContent = `${len} / ${MAX_CHARS}`;

    if (len > 1800 && len <= MAX_CHARS) {
      charCounter.className = "char-counter limit-near";
    } else if (len > MAX_CHARS) {
      charCounter.className = "char-counter limit-exceeded";
    } else {
      charCounter.className = "char-counter";
    }

    if (validationError.style.display !== "none") {
      hideValidationError();
    }
  }

  textarea.addEventListener("input", updateCharCount);

  // 2. Demonstration Chips Autofill
  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      const sample = chip.getAttribute("data-sample");
      if (sample) {
        textarea.value = sample;
        updateCharCount();
        textarea.focus();
      }
    });
  });

  // 3. Clear Input Form
  clearBtn.addEventListener("click", () => {
    textarea.value = "";
    updateCharCount();
    hideValidationError();
    textarea.focus();
  });

  function showValidationError(message) {
    validationError.textContent = message;
    validationError.style.display = "block";
    textarea.classList.add("has-error");
  }

  function hideValidationError() {
    validationError.textContent = "";
    validationError.style.display = "none";
    textarea.classList.remove("has-error");
  }

  function setLoading(isLoading) {
    submitBtn.disabled = isLoading;
    if (isLoading) {
      btnSpinner.style.display = "inline-block";
      btnText.textContent = "Analyzing Message...";
    } else {
      btnSpinner.style.display = "none";
      btnText.textContent = "Classify & Evaluate Priority";
    }
  }

  // 4. Form Submission & Classification
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    hideValidationError();

    const rawText = textarea.value;
    const trimmed = rawText.trim();

    if (!trimmed) {
      showValidationError("Please enter a customer feedback message before analyzing.");
      textarea.focus();
      return;
    }

    if (trimmed.length < MIN_CHARS) {
      showValidationError(`Message is too short. Please provide at least ${MIN_CHARS} characters.`);
      textarea.focus();
      return;
    }

    if (rawText.length > MAX_CHARS) {
      showValidationError(`Message exceeds the maximum limit of ${MAX_CHARS} characters.`);
      textarea.focus();
      return;
    }

    setLoading(true);

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: trimmed }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "An error occurred during sentiment classification.");
      }

      displayResult(data.result);
      prependHistoryItem(data.result);
    } catch (err) {
      showValidationError(err.message || "Network error. Please try again.");
    } finally {
      setLoading(false);
    }
  });

  // 5. Render Analysis Results
  function displayResult(result) {
    const { sentiment, confidence, probabilities, top_features, priority, is_urgent, reasons, suggested_action } = result;

    resultContainer.style.display = "block";

    // Sentiment badge
    resultSentimentBadge.className = `sentiment-badge badge-${sentiment.toLowerCase()}`;
    resultSentimentBadge.textContent = sentiment.charAt(0).toUpperCase() + sentiment.slice(1);
    resultConfidenceText.textContent = `${(confidence * 100).toFixed(1)}%`;

    // Probability bars
    const negPct = ((probabilities.negative || 0) * 100).toFixed(1);
    const neuPct = ((probabilities.neutral || 0) * 100).toFixed(1);
    const posPct = ((probabilities.positive || 0) * 100).toFixed(1);

    probBarNeg.style.width = `${negPct}%`;
    probValNeg.textContent = `${negPct}%`;

    probBarNeu.style.width = `${neuPct}%`;
    probValNeu.textContent = `${neuPct}%`;

    probBarPos.style.width = `${posPct}%`;
    probValPos.textContent = `${posPct}%`;

    // Top TF-IDF Terms
    topFeaturesList.innerHTML = "";
    if (top_features && top_features.length > 0) {
      topFeaturesWrapper.style.display = "block";
      top_features.forEach((term) => {
        const tag = document.createElement("span");
        tag.className = "feature-tag";
        tag.textContent = term;
        topFeaturesList.appendChild(tag);
      });
    } else {
      topFeaturesWrapper.style.display = "none";
    }

    // Transparent Priority Rule Banner
    if (is_urgent) {
      priorityBanner.className = "priority-banner priority-urgent";
      priorityIcon.textContent = "🚨";
      priorityHeading.textContent = "Priority Triage: Urgent";
      priorityPill.className = "priority-pill pill-urgent";
      priorityPill.textContent = "Urgent Escalation";
    } else {
      priorityBanner.className = "priority-banner priority-normal";
      priorityIcon.textContent = "🛡️";
      priorityHeading.textContent = "Priority Triage: Normal";
      priorityPill.className = "priority-pill pill-normal";
      priorityPill.textContent = "Standard Queue";
    }

    // Render Reasons
    priorityReasonsList.innerHTML = "";
    if (reasons && reasons.length > 0) {
      reasons.forEach((reason) => {
        const li = document.createElement("li");
        li.textContent = reason;
        priorityReasonsList.appendChild(li);
      });
    }

    priorityActionText.textContent = suggested_action || "Standard triage review.";

    // Smooth scroll to result
    resultContainer.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // 6. Prepend item to Session History list
  function prependHistoryItem(item) {
    if (emptyHistory) {
      emptyHistory.style.display = "none";
    }

    const itemDiv = document.createElement("div");
    itemDiv.id = item.id;
    itemDiv.className = `history-item ${item.is_urgent ? "item-urgent" : ""}`;

    const sentCapital = item.sentiment.charAt(0).toUpperCase() + item.sentiment.slice(1);
    const confPct = (item.confidence * 100).toFixed(1);

    itemDiv.innerHTML = `
      <div class="item-meta">
        <span class="item-time">${item.timestamp}</span>
        <div class="item-badges">
          <span class="sentiment-badge badge-${item.sentiment.toLowerCase()}">${sentCapital} (${confPct}%)</span>
          <span class="priority-pill ${item.is_urgent ? "pill-urgent" : "pill-normal"}">${item.priority}</span>
        </div>
      </div>
      <p class="item-text"></p>
      ${
        item.reasons && item.reasons.length > 0
          ? `<div class="item-reasons">
               <span class="reasons-label">Triage Audit:</span>
               <span class="reasons-content">${escapeHtml(item.reasons.join(" • "))}</span>
             </div>`
          : ""
      }
    `;

    // Safely assign user input text to prevent XSS
    itemDiv.querySelector(".item-text").textContent = item.text;

    historyContainer.insertBefore(itemDiv, historyContainer.firstChild);
  }

  function escapeHtml(str) {
    const p = document.createElement("p");
    p.appendChild(document.createTextNode(str));
    return p.innerHTML;
  }

  // 7. Clear Session History action
  clearHistoryBtn.addEventListener("click", async () => {
    if (!confirm("Clear all feedback submitted in this session? This action cannot be undone.")) {
      return;
    }

    try {
      const response = await fetch("/api/clear-session", { method: "POST" });
      if (response.ok) {
        historyContainer.innerHTML = "";
        if (emptyHistory) {
          emptyHistory.style.display = "block";
        }
      }
    } catch (err) {
      console.error("Error clearing session:", err);
    }
  });
});
