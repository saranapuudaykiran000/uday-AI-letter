const els = {
  roleTitle: document.getElementById("role-title"),
  companyName: document.getElementById("company-name"),
  resumeText: document.getElementById("resume-text"),
  jdText: document.getElementById("jd-text"),
  resumeFile: document.getElementById("resume-file"),
  jdFile: document.getElementById("jd-file"),
  provider: document.getElementById("provider"),
  tone: document.getElementById("tone"),
  length: document.getElementById("length"),
  temperature: document.getElementById("temperature"),
  tempValue: document.getElementById("temp-value"),
  modelOverride: document.getElementById("model-override"),
  apiKeyOverride: document.getElementById("api-key-override"),
  loadSampleBtn: document.getElementById("load-sample-btn"),
  analyzeBtn: document.getElementById("analyze-btn"),
  generateBtn: document.getElementById("generate-btn"),
  refineBtn: document.getElementById("refine-btn"),
  refineFeedback: document.getElementById("refine-feedback"),
  copyBtn: document.getElementById("copy-btn"),
  exportTxtBtn: document.getElementById("export-txt-btn"),
  exportMdBtn: document.getElementById("export-md-btn"),
  printBtn: document.getElementById("print-btn"),
  atsScore: document.getElementById("ats-score"),
  atsProgressFill: document.getElementById("ats-progress-fill"),
  matchedCount: document.getElementById("matched-count"),
  missingCount: document.getElementById("missing-count"),
  matchedPills: document.getElementById("matched-pills"),
  missingPills: document.getElementById("missing-pills"),
  generationMeta: document.getElementById("generation-meta"),
  noticeBanner: document.getElementById("notice-banner"),
  coverLetterOutput: document.getElementById("cover-letter-output"),
  wordCountBar: document.getElementById("word-count-bar"),
  printSheet: document.getElementById("print-sheet"),
  inspectorSystem: document.getElementById("inspector-system"),
  inspectorUser: document.getElementById("inspector-user"),
  providerStatusPill: document.getElementById("provider-status-pill"),
};

function showNotice(message, isError = false) {
  if (!message) {
    els.noticeBanner.classList.add("hidden");
    els.noticeBanner.textContent = "";
    els.noticeBanner.classList.remove("error");
    return;
  }
  els.noticeBanner.textContent = message;
  els.noticeBanner.classList.remove("hidden");
  els.noticeBanner.classList.toggle("error", isError);
}

function updateDraftStats() {
  const text = els.coverLetterOutput.value || "";
  const trimmed = text.trim();
  const words = trimmed ? trimmed.split(/\s+/).length : 0;
  els.wordCountBar.textContent = `${words} words · ${text.length} characters`;
  els.printSheet.textContent = text;

  const hasDraft = Boolean(trimmed);
  els.copyBtn.disabled = !hasDraft;
  els.exportTxtBtn.disabled = !hasDraft;
  els.exportMdBtn.disabled = !hasDraft;
  els.printBtn.disabled = !hasDraft;
  els.refineBtn.disabled = !hasDraft;
}

function renderPills(container, items, className, emptyText) {
  container.innerHTML = "";
  if (!items || items.length === 0) {
    const span = document.createElement("span");
    span.className = "empty-note";
    span.textContent = emptyText;
    container.appendChild(span);
    return;
  }
  for (const item of items) {
    const pill = document.createElement("span");
    pill.className = `pill ${className}`;
    pill.textContent = item;
    container.appendChild(pill);
  }
}

function renderATS(ats) {
  if (!ats) return;
  const score = Number(ats.match_score || 0);
  els.atsScore.textContent = `${score}%`;
  els.atsProgressFill.style.width = `${Math.max(0, Math.min(100, score))}%`;
  const matched = ats.matched_keywords || [];
  const missing = ats.missing_keywords || [];
  els.matchedCount.textContent = String(matched.length);
  els.missingCount.textContent = String(missing.length);

  renderPills(els.matchedPills, matched, "pill-match", "No matched keywords detected yet.");
  renderPills(els.missingPills, missing, "pill-miss", "No missing keywords — 100% of extracted keywords matched!");
}

function renderPromptInspector(systemPrompt, userPrompt) {
  if (systemPrompt) els.inspectorSystem.textContent = systemPrompt;
  if (userPrompt) els.inspectorUser.textContent = userPrompt;
}

function gatherPayload() {
  return {
    resume_text: els.resumeText.value,
    job_description: els.jdText.value,
    role_title: els.roleTitle.value,
    company_name: els.companyName.value,
    provider: els.provider.value,
    tone: els.tone.value,
    length: els.length.value,
    temperature: parseFloat(els.temperature.value),
    model: els.modelOverride.value.trim(),
    api_key: els.apiKeyOverride.value.trim(),
    allow_fallback: true,
  };
}

async function postJson(url, body) {
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const data = await res.json();
  if (!res.ok) {
    throw new Error(data.error || `Request failed with status ${res.status}`);
  }
  return data;
}

function setBusy(busy, label = "") {
  els.analyzeBtn.disabled = busy;
  els.generateBtn.disabled = busy;
  if (busy) {
    els.refineBtn.disabled = true;
    els.generationMeta.textContent = label || "Working…";
  } else {
    updateDraftStats();
  }
}

async function handleAnalyze() {
  showNotice("");
  setBusy(true, "Running ATS keyword analysis…");
  try {
    const data = await postJson("/api/analyze", {
      resume_text: els.resumeText.value,
      job_description: els.jdText.value,
    });
    renderATS(data.ats);
    els.generationMeta.textContent = "ATS analysis complete";
  } catch (err) {
    showNotice(err.message, true);
    els.generationMeta.textContent = "Analysis failed";
  } finally {
    setBusy(false);
  }
}

async function handleGenerate() {
  showNotice("");
  setBusy(true, "Running 3-stage NLP + LLM pipeline…");
  try {
    const data = await postJson("/api/generate", gatherPayload());
    renderATS(data.ats);
    els.coverLetterOutput.value = data.cover_letter || "";
    updateDraftStats();
    renderPromptInspector(data.system_prompt, data.user_prompt);
    els.generationMeta.textContent = `Provider: ${data.provider} (${data.model})`;
    showNotice(data.notice || "", false);
  } catch (err) {
    showNotice(err.message, true);
    els.generationMeta.textContent = "Generation failed";
  } finally {
    setBusy(false);
  }
}

async function handleRefine() {
  const feedback = els.refineFeedback.value.trim();
  if (!feedback) {
    showNotice("Enter refinement feedback before clicking Refine Draft.", true);
    return;
  }
  showNotice("");
  setBusy(true, "Refining draft from feedback…");
  try {
    const payload = {
      ...gatherPayload(),
      current_draft: els.coverLetterOutput.value,
      feedback,
    };
    const data = await postJson("/api/refine", payload);
    renderATS(data.ats);
    els.coverLetterOutput.value = data.cover_letter || "";
    updateDraftStats();
    renderPromptInspector(data.system_prompt, data.user_prompt);
    els.generationMeta.textContent = `Refined via ${data.provider} (${data.model})`;
    showNotice(data.notice || "Draft refined using your feedback.", false);
  } catch (err) {
    showNotice(err.message, true);
    els.generationMeta.textContent = "Refinement failed";
  } finally {
    setBusy(false);
  }
}

function attachFileReader(inputEl, targetTextarea) {
  inputEl.addEventListener("change", async (e) => {
    const file = e.target.files && e.target.files[0];
    if (!file) return;
    targetTextarea.value = await file.text();
    inputEl.value = "";
  });
}

function downloadFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

async function init() {
  els.temperature.addEventListener("input", () => {
    els.tempValue.textContent = els.temperature.value;
  });

  els.coverLetterOutput.addEventListener("input", updateDraftStats);
  attachFileReader(els.resumeFile, els.resumeText);
  attachFileReader(els.jdFile, els.jdText);

  els.analyzeBtn.addEventListener("click", handleAnalyze);
  els.generateBtn.addEventListener("click", handleGenerate);
  els.refineBtn.addEventListener("click", handleRefine);

  els.refineFeedback.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      handleRefine();
    }
  });

  document.querySelectorAll(".chip").forEach((chip) => {
    chip.addEventListener("click", () => {
      els.refineFeedback.value = chip.dataset.feedback || "";
      els.refineFeedback.focus();
    });
  });

  els.loadSampleBtn.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/samples");
      const data = await res.json();
      els.resumeText.value = data.resume_text || "";
      els.jdText.value = data.job_description || "";
      showNotice("Loaded sample resume and job description.", false);
    } catch (err) {
      showNotice(`Could not load samples: ${err.message}`, true);
    }
  });

  els.copyBtn.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(els.coverLetterOutput.value);
      showNotice("Cover letter copied to clipboard.", false);
    } catch {
      showNotice("Clipboard access blocked by browser; copy directly from the editor.", true);
    }
  });

  els.exportTxtBtn.addEventListener("click", () => {
    downloadFile("cover-letter.txt", els.coverLetterOutput.value, "text/plain;charset=utf-8");
  });

  els.exportMdBtn.addEventListener("click", () => {
    const content = `# Cover Letter\n\n${els.coverLetterOutput.value}\n`;
    downloadFile("cover-letter.md", content, "text/markdown;charset=utf-8");
  });

  els.printBtn.addEventListener("click", () => {
    els.printSheet.textContent = els.coverLetterOutput.value;
    window.print();
  });

  try {
    const res = await fetch("/api/health");
    const data = await res.json();
    const providers = data.providers || {};
    const ready = Object.entries(providers)
      .filter(([, info]) => info.configured)
      .map(([name]) => name);
    els.providerStatusPill.textContent = `Ready: ${ready.join(", ")}`;
  } catch {
    els.providerStatusPill.textContent = "Server status unknown";
  }

  updateDraftStats();
}

init();
