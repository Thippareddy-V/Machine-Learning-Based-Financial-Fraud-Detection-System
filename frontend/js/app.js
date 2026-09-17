/* ============================================================
   FraudShield AI — app.js
   Handles: API communication, validation, state rendering,
   risk gauge animation, reset, and error handling.
   ============================================================ */

(function () {
  "use strict";

  const API_URL = "/predict";
  const GAUGE_CIRCUMFERENCE = 2 * Math.PI * 86; // r = 86

  /* ----------------------------------------------------------
     Element references
     ---------------------------------------------------------- */
  const form = document.getElementById("transactionForm");
  const submitBtn = document.getElementById("submitBtn");

  const flaggedToggle = document.getElementById("flaggedToggle");
  const isFlaggedFraudInput = document.getElementById("isFlaggedFraud");

  const emptyState = document.getElementById("emptyState");
  const loadingState = document.getElementById("loadingState");
  const errorState = document.getElementById("errorState");
  const resultState = document.getElementById("resultState");
  const resultSubtitle = document.getElementById("resultSubtitle");
  const retryBtn = document.getElementById("retryBtn");
  const resetBtn = document.getElementById("resetBtn");

  const riskScoreCard = document.getElementById("riskScoreCard");
  const riskBadge = document.getElementById("riskBadge");
  const riskMessage = document.getElementById("riskMessage");
  const gaugeValue = document.getElementById("gaugeValue");
  const gaugeNumber = document.getElementById("gaugeNumber");

  const statPrediction = document.getElementById("statPrediction");
  const statProbability = document.getElementById("statProbability");
  const statThreshold = document.getElementById("statThreshold");
  const summaryList = document.getElementById("summaryList");

  const apiStatusPill = document.getElementById("apiStatusPill");
  const apiStatusLabel = document.getElementById("apiStatusLabel");

  let lastTransaction = null;
  let isSubmitting = false;

  /* ----------------------------------------------------------
     Flagged toggle
     ---------------------------------------------------------- */
  flaggedToggle.addEventListener("click", function () {
    const isOn = flaggedToggle.getAttribute("aria-checked") === "true";
    flaggedToggle.setAttribute("aria-checked", String(!isOn));
    isFlaggedFraudInput.value = isOn ? "0" : "1";
  });

  /* ----------------------------------------------------------
     Validation
     ---------------------------------------------------------- */
  function clearErrors() {
    form.querySelectorAll(".field__error").forEach((el) => (el.textContent = ""));
    form.querySelectorAll(".is-invalid").forEach((el) => el.classList.remove("is-invalid"));
  }

  function setError(fieldId, message) {
    const errorEl = form.querySelector('[data-error-for="' + fieldId + '"]');
    const inputEl = document.getElementById(fieldId);
    if (errorEl) errorEl.textContent = message;
    if (inputEl) inputEl.classList.add("is-invalid");
  }

  function validate(values) {
    clearErrors();
    let isValid = true;

    if (!values.type) {
      setError("type", "Transaction type is required.");
      isValid = false;
    }

    const numericFields = [
      ["amount", "Amount cannot be negative."],
      ["oldbalanceOrg", "Balance cannot be negative."],
      ["newbalanceOrig", "Balance cannot be negative."],
      ["oldbalanceDest", "Balance cannot be negative."],
      ["newbalanceDest", "Balance cannot be negative."],
      ["step", "Step cannot be negative."],
    ];

    numericFields.forEach(([field, message]) => {
      const raw = values[field];
      if (raw === "" || raw === null || Number.isNaN(Number(raw))) {
        setError(field, field === "step" ? "Step is required." : "This field is required.");
        isValid = false;
      } else if (Number(raw) < 0) {
        setError(field, message);
        isValid = false;
      }
    });

    return isValid;
  }

  /* ----------------------------------------------------------
     Formatting helpers
     ---------------------------------------------------------- */
  function formatCurrency(value) {
    const num = Number(value) || 0;
    return num.toLocaleString("en-US", {
      style: "currency",
      currency: "USD",
      maximumFractionDigits: 2,
    });
  }

  function formatPercent(fraction) {
    return (fraction * 100).toFixed(2) + "%";
  }

  function readableType(type) {
    const map = {
      PAYMENT: "Payment",
      TRANSFER: "Transfer",
      CASH_OUT: "Cash out",
      DEBIT: "Debit",
      CASH_IN: "Cash in",
    };
    return map[type] || type;
  }

  /* ----------------------------------------------------------
     State switching
     ---------------------------------------------------------- */
  function showState(name) {
    emptyState.hidden = name !== "empty";
    loadingState.hidden = name !== "loading";
    errorState.hidden = name !== "error";
    resultState.hidden = name !== "result";

    const subtitles = {
      empty: "Run the model to see a live risk assessment.",
      loading: "Scoring in progress…",
      error: "The last analysis could not be completed.",
      result: "Here's what the model found for this transaction.",
    };
    resultSubtitle.textContent = subtitles[name] || subtitles.empty;
  }

  function setSubmitting(submitting) {
    isSubmitting = submitting;
    submitBtn.disabled = submitting;
    submitBtn.classList.toggle("btn--loading", submitting);
  }

  /* ----------------------------------------------------------
     Gauge + scale animation
     ---------------------------------------------------------- */
  function clampFraction(value) {
    return Math.min(Math.max(Number(value) || 0, 0), 1);
  }

  function getRiskLevel(score, prediction) {
    if (prediction === "Fraud" || score >= 75) {
      return {
        key: "high",
        label: "High Risk / Fraud",
        message: "High fraud probability detected. This transaction exceeds the model's decision threshold.",
      };
    }

    if (score >= 40) {
      return {
        key: "medium",
        label: "Medium Risk",
        message: "Moderate model fraud probability detected. Review this transaction before approval.",
      };
    }

    return {
      key: "low",
      label: "Low Risk / Legitimate",
      message: "Low fraud probability detected. This transaction is below the model's decision threshold.",
    };
  }

  function formatScore(score) {
    return score % 1 === 0 ? String(score.toFixed(0)) : score.toFixed(2);
  }

  function animateGauge(fraction, riskKey) {
    gaugeValue.style.strokeDasharray = String(GAUGE_CIRCUMFERENCE);
    gaugeValue.style.strokeDashoffset = String(GAUGE_CIRCUMFERENCE);
    gaugeValue.classList.remove("is-high", "is-medium", "is-low");
    gaugeValue.classList.add("is-" + riskKey);

    // force reflow so the transition restarts on repeated submissions
    void gaugeValue.getBoundingClientRect();

    const offset = GAUGE_CIRCUMFERENCE * (1 - fraction);
    requestAnimationFrame(() => {
      gaugeValue.style.strokeDashoffset = String(offset);
    });

    animateNumber(gaugeNumber, fraction * 100);
  }

  function animateNumber(el, target) {
    const duration = 900;
    const start = performance.now();

    function tick(now) {
      const elapsed = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - elapsed, 3);
      el.textContent = formatScore(target * eased);
      if (elapsed < 1) requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  /* ----------------------------------------------------------
     Result rendering
     ---------------------------------------------------------- */
  function renderResult(response, submittedValues) {
    const probability = clampFraction(response.fraud_probability);
    const threshold = clampFraction(response.threshold);
    const prediction = response.prediction === "Fraud" ? "Fraud" : "Legitimate";
    const score = probability * 100;
    const risk = getRiskLevel(score, prediction);

    riskScoreCard.classList.remove("risk-score-card--high", "risk-score-card--medium", "risk-score-card--low");
    riskScoreCard.classList.add("risk-score-card--" + risk.key);
    riskBadge.classList.remove("risk-badge--high", "risk-badge--medium", "risk-badge--low", "risk-badge--neutral");
    riskBadge.classList.add("risk-badge--" + risk.key);
    riskBadge.textContent = risk.label;
    riskMessage.textContent = risk.message;

    animateGauge(probability, risk.key);

    statPrediction.textContent = prediction.toUpperCase();
    statProbability.textContent = formatPercent(probability);
    statThreshold.textContent = formatPercent(threshold);

    summaryList.innerHTML = "";
    const summaryRows = [
      ["Transaction type", readableType(submittedValues.type)],
      ["Amount", formatCurrency(submittedValues.amount)],
      ["Origin balance", formatCurrency(submittedValues.oldbalanceOrg)],
      ["Destination balance", formatCurrency(submittedValues.oldbalanceDest)],
    ];
    summaryRows.forEach(([label, value]) => {
      const row = document.createElement("div");
      row.innerHTML = "<dt>" + label + "</dt><dd>" + value + "</dd>";
      summaryList.appendChild(row);
    });

    showState("result");
  }

  /* ----------------------------------------------------------
     Form submission
     ---------------------------------------------------------- */
  form.addEventListener("submit", async function (event) {
    event.preventDefault();
    if (isSubmitting) return;

    const formData = new FormData(form);
    const rawValues = Object.fromEntries(formData.entries());

    if (!validate(rawValues)) return;

    const transaction = {
      type: rawValues.type,
      amount: Number(rawValues.amount),
      oldbalanceOrg: Number(rawValues.oldbalanceOrg),
      newbalanceOrig: Number(rawValues.newbalanceOrig),
      oldbalanceDest: Number(rawValues.oldbalanceDest),
      newbalanceDest: Number(rawValues.newbalanceDest),
      isFlaggedFraud: Number(isFlaggedFraudInput.value) || 0,
      step: Number(rawValues.step),
      nameOrig: rawValues.nameOrig ? rawValues.nameOrig.trim() || "unknown" : "unknown",
      nameDest: rawValues.nameDest ? rawValues.nameDest.trim() || "unknown" : "unknown",
    };

    lastTransaction = transaction;

    setSubmitting(true);
    showState("loading");

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(transaction),
      });

      if (!response.ok) {
        throw new Error("Request failed with status " + response.status);
      }

      const data = await response.json();
      renderResult(data, transaction);
      setApiStatus(true);
    } catch (err) {
      showState("error");
      setApiStatus(false);
    } finally {
      setSubmitting(false);
    }
  });

  /* ----------------------------------------------------------
     Retry / reset
     ---------------------------------------------------------- */
  retryBtn.addEventListener("click", async function () {
    if (!lastTransaction) {
      showState("empty");
      return;
    }
    showState("loading");
    setSubmitting(true);
    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(lastTransaction),
      });
      if (!response.ok) throw new Error("Request failed with status " + response.status);
      const data = await response.json();
      renderResult(data, lastTransaction);
      setApiStatus(true);
    } catch (err) {
      showState("error");
      setApiStatus(false);
    } finally {
      setSubmitting(false);
    }
  });

  resetBtn.addEventListener("click", function () {
    form.reset();
    clearErrors();
    flaggedToggle.setAttribute("aria-checked", "false");
    isFlaggedFraudInput.value = "0";
    document.getElementById("step").value = "1";
    lastTransaction = null;
    showState("empty");
    form.querySelector("select#type").focus();
  });

  /* ----------------------------------------------------------
     API connectivity indicator (best-effort, non-blocking)
     ---------------------------------------------------------- */
  function setApiStatus(online) {
    apiStatusPill.classList.remove("status-pill--online", "status-pill--offline");
    apiStatusPill.classList.add(online ? "status-pill--online" : "status-pill--offline");
    apiStatusLabel.textContent = online ? "System online" : "Service unreachable";
  }

  showState("empty");
})();
