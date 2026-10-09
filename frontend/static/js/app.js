"use strict";

const state = { model: null, scans: [], selectedFile: null, previewUrl: null, busy: false };
const MAX_UPLOAD_MB = Number(document.body.dataset.maxUploadMb || 12);
const $ = (id) => document.getElementById(id);

const elements = {
  uploadForm: $("upload-form"),
  fileInput: $("scan-file"),
  dropZone: $("drop-zone"),
  selectedFile: $("selected-file"),
  selectedPreview: $("selected-preview"),
  selectedName: $("selected-name"),
  selectedMeta: $("selected-meta"),
  uploadButton: $("upload-button"),
  uploadFeedback: $("upload-feedback"),
  tableBody: $("scan-table-body"),
  historyEmpty: $("history-empty"),
  tableWrap: $("history-table-wrap"),
  historyCount: $("history-count"),
  summaryTotal: $("metric-total"),
  summaryCompleted: $("metric-completed"),
  metricModelState: $("metric-model-state"),
  metricModelCaption: $("metric-model-caption"),
  topModelPill: $("top-model-pill"),
  topModelLabel: $("top-model-label"),
  modelCardTitle: $("model-card-title"),
  modelStatusCopy: $("model-status-copy"),
  modelPathDisplay: $("model-path-display"),
  search: $("history-search"),
  filter: $("status-filter"),
  dialog: $("scan-dialog"),
  dialogBody: $("dialog-body"),
  dialogTitle: $("dialog-title"),
  toastRegion: $("toast-region"),
};

function formatBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function formatDate(value, includeTime = false) {
  if (!value) return "—";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "—";
  const options = includeTime
    ? { day: "numeric", month: "short", year: "numeric", hour: "2-digit", minute: "2-digit" }
    : { day: "numeric", month: "short", year: "numeric" };
  return new Intl.DateTimeFormat(undefined, options).format(date);
}

function setDate() {
  const now = new Date();
  $("local-date").textContent = new Intl.DateTimeFormat(undefined, { weekday: "short", day: "numeric", month: "short", year: "numeric" }).format(now);
}

function toast(message, isError = false) {
  const item = document.createElement("div");
  item.className = `toast${isError ? " is-error" : ""}`;
  item.textContent = message;
  elements.toastRegion.append(item);
  window.setTimeout(() => item.remove(), 4600);
}

async function requestJson(url, options = {}) {
  const response = await fetch(url, { cache: "no-store", ...options });
  let payload;
  try { payload = await response.json(); }
  catch { payload = {}; }
  if (!response.ok) {
    const error = new Error(payload?.error?.message || `Request failed (${response.status}).`);
    error.status = response.status;
    error.code = payload?.error?.code || "REQUEST_FAILED";
    error.details = payload?.error?.details || null;
    throw error;
  }
  return payload;
}

function renderModelStatus(model) {
  state.model = model;
  const pill = elements.topModelPill;
  pill.classList.toggle("is-ready", Boolean(model.ready));
  pill.classList.toggle("is-error", model.state === "invalid_artifact" || model.state === "dependencies_missing");
  const setupIssue = model.state === "invalid_artifact" || model.state === "dependencies_missing";
  elements.topModelLabel.textContent = model.ready ? "Model ready" : setupIssue ? "Model needs attention" : "Model not configured";
  elements.metricModelState.textContent = model.ready ? "Ready" : setupIssue ? "Setup issue" : "Not configured";
  elements.metricModelCaption.textContent = model.ready ? "CPU inference available" : setupIssue ? "Review checkpoint setup" : "No predictions will be generated";
  elements.modelCardTitle.textContent = model.ready ? "Checkpoint loaded" : model.state === "dependencies_missing" ? "ML dependencies missing" : model.state === "invalid_artifact" ? "Checkpoint needs attention" : "No checkpoint installed";
  elements.modelStatusCopy.textContent = model.message || "Model status unavailable.";
  elements.modelPathDisplay.textContent = model.artifact || "models/brain_tumor_resnet50.pt";
}

function statusLabel(status) {
  return ({
    awaiting_model: "Awaiting model",
    completed: "Analyzed",
    inference_failed: "Inference failed",
  })[status] || "Unknown";
}

function makeActionButton(label, action, scanId, iconName, danger = false) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `table-action${danger ? " is-danger" : ""}`;
  button.dataset.action = action;
  button.dataset.id = scanId;
  button.setAttribute("aria-label", `${label} scan`);
  button.title = label;
  const paths = {
    view: '<rect x="3" y="4" width="18" height="16" rx="3"></rect><circle cx="12" cy="12" r="3"></circle><path d="M5 18 9 14l3 3 3-4 4 5"></path>',
    analyze: '<path d="m8 5 11 7-11 7V5Z"></path>',
    retry: '<path d="M20 7v5h-5M4 17v-5h5"></path><path d="M6.2 9A6.5 6.5 0 0 1 17 6.8L20 12M4 12l3 5.2A6.5 6.5 0 0 0 18 15"></path>',
    delete: '<path d="M4 7h16M10 11v6m4-6v6M6 7l1 13h10l1-13M9 7V4h6v3"></path>',
  };
  button.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true">${paths[iconName]}</svg>`;
  return button;
}

function renderScans() {
  const query = (elements.search.value || "").trim().toLowerCase();
  const statusFilter = elements.filter.value;
  const scans = state.scans.filter((scan) => {
    const matchesName = !query || scan.filename.toLowerCase().includes(query);
    const matchesStatus = statusFilter === "all" || scan.status === statusFilter;
    return matchesName && matchesStatus;
  });

  elements.tableBody.replaceChildren();
  for (const scan of scans) {
    const row = document.createElement("tr");

    const imageCell = document.createElement("td");
    const fileInfo = document.createElement("div");
    fileInfo.className = "scan-file-cell";
    const thumb = document.createElement("span");
    thumb.className = "scan-thumb";
    const image = document.createElement("img");
    image.src = `${scan.image_url}?v=${encodeURIComponent(scan.created_at || scan.id)}`;
    image.alt = "";
    image.loading = "lazy";
    thumb.append(image);
    const nameWrap = document.createElement("span");
    nameWrap.className = "scan-name-wrap";
    const name = document.createElement("strong");
    name.className = "scan-file-name";
    name.textContent = scan.filename;
    name.title = scan.filename;
    const meta = document.createElement("span");
    meta.className = "scan-file-meta";
    meta.textContent = formatBytes(scan.size_bytes);
    nameWrap.append(name, meta);
    fileInfo.append(thumb, nameWrap);
    imageCell.append(fileInfo);

    const dateCell = document.createElement("td");
    dateCell.textContent = formatDate(scan.created_at);
    const dimensionsCell = document.createElement("td");
    dimensionsCell.textContent = `${scan.width} × ${scan.height}`;
    const statusCell = document.createElement("td");
    const status = document.createElement("span");
    status.className = `status-chip status-${scan.status}`;
    status.textContent = statusLabel(scan.status);
    statusCell.append(status);

    const outputCell = document.createElement("td");
    if (scan.prediction_class) {
      const prediction = document.createElement("span");
      prediction.className = "prediction-label";
      prediction.textContent = scan.prediction_class;
      const confidence = document.createElement("span");
      confidence.className = "prediction-sub";
      confidence.textContent = `Model score ${(Number(scan.confidence) * 100).toFixed(1)}% · not calibrated`;
      outputCell.append(prediction, confidence);
    } else {
      outputCell.textContent = scan.status === "inference_failed" ? "Retry after checking setup" : "No output recorded";
    }

    const actionCell = document.createElement("td");
    actionCell.className = "action-cell";
    actionCell.append(makeActionButton("View", "view", scan.id, "view"));
    if (scan.status !== "completed") {
      actionCell.append(makeActionButton(scan.status === "inference_failed" ? "Retry analysis" : "Analyze", "analyze", scan.id, scan.status === "inference_failed" ? "retry" : "analyze"));
    }
    actionCell.append(makeActionButton("Delete", "delete", scan.id, "delete", true));

    row.append(imageCell, dateCell, dimensionsCell, statusCell, outputCell, actionCell);
    elements.tableBody.append(row);
  }

  const isEmpty = scans.length === 0;
  elements.historyEmpty.hidden = !isEmpty;
  elements.tableWrap.hidden = isEmpty;
  if (isEmpty && state.scans.length > 0) {
    $("empty-title").textContent = "No matching scans";
    $("empty-copy").textContent = "Try a different filename or status filter.";
  } else if (isEmpty) {
    $("empty-title").textContent = "No scans yet";
    $("empty-copy").textContent = "Your saved images will appear here. Start with a single 2D MRI image.";
  }
  const countText = `${scans.length} ${scans.length === 1 ? "scan" : "scans"}`;
  elements.historyCount.textContent = countText;
  $("nav-scan-count").textContent = state.scans.length > 99 ? "99+" : String(state.scans.length);
}

async function loadWorkspace() {
  try {
    const [model, scanResponse, summary] = await Promise.all([
      requestJson("/api/model/status"),
      requestJson("/api/scans?limit=100"),
      requestJson("/api/summary"),
    ]);
    renderModelStatus(model);
    state.scans = scanResponse.scans;
    elements.summaryTotal.textContent = String(summary.total);
    elements.summaryCompleted.textContent = String(summary.completed);
    renderScans();
  } catch (error) {
    elements.summaryTotal.textContent = "—";
    elements.summaryCompleted.textContent = "—";
    toast(`Could not load workspace data: ${error.message}`, true);
  }
}

function setSelectedFile(file) {
  if (!file) return;
  const allowedTypes = ["image/jpeg", "image/png", "image/webp"];
  const extensionAllowed = /\.(jpe?g|png|webp)$/i.test(file.name);
  if (!allowedTypes.includes(file.type) || !extensionAllowed) {
    toast("Choose a JPEG, PNG or WebP image.", true);
    elements.fileInput.value = "";
    return;
  }
  const maxBytes = MAX_UPLOAD_MB * 1024 * 1024;
  if (file.size > maxBytes) {
    toast(`The selected image is larger than the ${MAX_UPLOAD_MB} MB limit.`, true);
    elements.fileInput.value = "";
    return;
  }
  if (file.size === 0) {
    toast("The selected file is empty.", true);
    elements.fileInput.value = "";
    return;
  }
  state.selectedFile = file;
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = URL.createObjectURL(file);
  elements.selectedPreview.src = state.previewUrl;
  elements.selectedName.textContent = file.name;
  elements.selectedMeta.textContent = `${formatBytes(file.size)} · ${file.type.split("/")[1]?.toUpperCase() || "IMAGE"}`;
  elements.selectedFile.hidden = false;
  elements.uploadButton.disabled = false;
  elements.uploadFeedback.hidden = true;
}

function clearSelectedFile() {
  if (state.previewUrl) URL.revokeObjectURL(state.previewUrl);
  state.previewUrl = null;
  state.selectedFile = null;
  elements.fileInput.value = "";
  elements.selectedFile.hidden = true;
  elements.selectedPreview.removeAttribute("src");
  elements.uploadButton.disabled = true;
}

function setFeedback(title, detail, isError = false) {
  elements.uploadFeedback.replaceChildren();
  const strong = document.createElement("strong");
  strong.textContent = title;
  const text = document.createElement("span");
  text.textContent = detail;
  elements.uploadFeedback.append(strong, text);
  elements.uploadFeedback.classList.toggle("is-error", isError);
  elements.uploadFeedback.hidden = false;
}

async function handleUpload(event) {
  event.preventDefault();
  if (!state.selectedFile || state.busy) return;
  state.busy = true;
  elements.uploadButton.disabled = true;
  const label = elements.uploadButton.querySelector(".button-label");
  const originalLabel = label.textContent;
  label.textContent = "Saving image…";
  elements.uploadFeedback.hidden = true;
  try {
    const form = new FormData();
    form.append("file", state.selectedFile, state.selectedFile.name);
    const result = await requestJson("/api/scans", { method: "POST", body: form });
    let scan = result.scan;
    let modelNote;
    if (state.model?.ready) {
      label.textContent = "Running model…";
      try {
        const analysis = await requestJson(`/api/scans/${encodeURIComponent(scan.id)}/analyze`, { method: "POST" });
        scan = analysis.scan;
        modelNote = `Research output: ${scan.prediction_class} · ${(Number(scan.confidence) * 100).toFixed(1)}% model score. This is not a diagnosis.`;
        setFeedback("Review saved and analyzed.", modelNote);
      } catch (error) {
        modelNote = error.message;
        setFeedback("Image saved; no prediction was produced.", `${error.message} Check Model & method before retrying.`, true);
      }
    } else {
      modelNote = "No classification was made because no compatible model checkpoint is installed.";
      setFeedback("Image saved locally.", modelNote, false);
    }
    clearSelectedFile();
    await loadWorkspace();
    toast(scan.status === "completed" ? "Scan saved and analysis recorded." : "Scan saved. No prediction was generated.");
    document.getElementById("history").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    setFeedback("Upload could not be completed.", error.message, true);
    toast(error.message, true);
  } finally {
    state.busy = false;
    label.textContent = originalLabel;
    elements.uploadButton.disabled = !state.selectedFile;
  }
}

function appendDetailRow(container, label, value) {
  const row = document.createElement("div");
  row.className = "dialog-meta-row";
  const key = document.createElement("span");
  key.textContent = label;
  const val = document.createElement("strong");
  val.textContent = value;
  row.append(key, val);
  container.append(row);
}

function openScanDetails(scan) {
  elements.dialogTitle.textContent = scan.filename;
  elements.dialogBody.replaceChildren();
  const imageWrap = document.createElement("div");
  imageWrap.className = "dialog-image-wrap";
  const image = document.createElement("img");
  image.src = `${scan.image_url}?v=${encodeURIComponent(scan.created_at || scan.id)}`;
  image.alt = `Saved image preview: ${scan.filename}`;
  imageWrap.append(image);

  const meta = document.createElement("div");
  meta.className = "dialog-meta";
  const heading = document.createElement("h3");
  heading.textContent = "Review details";
  meta.append(heading);
  appendDetailRow(meta, "Added", formatDate(scan.created_at, true));
  appendDetailRow(meta, "Image size", `${scan.width} × ${scan.height} px`);
  appendDetailRow(meta, "File size", formatBytes(scan.size_bytes));
  appendDetailRow(meta, "Status", statusLabel(scan.status));
  if (scan.prediction_class) {
    const result = document.createElement("div");
    result.className = "dialog-result";
    const title = document.createElement("strong");
    title.textContent = `${scan.prediction_class} · ${(Number(scan.confidence) * 100).toFixed(1)}% model score`;
    const disclaimer = document.createElement("p");
    disclaimer.textContent = "Research-only output. Scores are not calibrated probabilities and must not guide clinical decisions.";
    result.append(title, disclaimer);
    if (scan.probabilities) {
      for (const [label, score] of Object.entries(scan.probabilities)) {
        const line = document.createElement("p");
        line.textContent = `${label}: ${(Number(score) * 100).toFixed(1)}%`;
        result.append(line);
      }
    }
    meta.append(result);
  } else {
    const result = document.createElement("div");
    result.className = "dialog-result";
    result.textContent = scan.status === "inference_failed"
      ? "Inference failed. Check the model artifact and server log before retrying."
      : "No classification has been made. A compatible trained checkpoint is required.";
    meta.append(result);
  }
  elements.dialogBody.append(imageWrap, meta);
  elements.dialog.showModal();
}

async function handleTableAction(event) {
  const button = event.target.closest("button[data-action]");
  if (!button) return;
  const scan = state.scans.find((entry) => entry.id === button.dataset.id);
  if (!scan) return;
  if (button.dataset.action === "view") {
    openScanDetails(scan);
    return;
  }
  if (button.dataset.action === "delete") {
    const confirmed = window.confirm(`Delete “${scan.filename}” and its locally stored image? This cannot be undone.`);
    if (!confirmed) return;
    button.disabled = true;
    try {
      await requestJson(`/api/scans/${encodeURIComponent(scan.id)}`, { method: "DELETE" });
      state.scans = state.scans.filter((entry) => entry.id !== scan.id);
      renderScans();
      const summary = await requestJson("/api/summary");
      elements.summaryTotal.textContent = String(summary.total);
      elements.summaryCompleted.textContent = String(summary.completed);
      toast("Scan and saved image deleted.");
    } catch (error) {
      button.disabled = false;
      toast(error.message, true);
    }
    return;
  }
  if (button.dataset.action === "analyze" || button.dataset.action === "retry") {
    if (!state.model?.ready) {
      toast("No compatible model checkpoint is installed. No prediction was made.", true);
      document.getElementById("model").scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }
    button.disabled = true;
    try {
      await requestJson(`/api/scans/${encodeURIComponent(scan.id)}/analyze`, { method: "POST" });
      await loadWorkspace();
      toast("Model output recorded. Treat it as research-only.");
    } catch (error) {
      toast(error.message, true);
      await loadWorkspace();
    }
  }
}

function initNavigation() {
  const navButtons = [...document.querySelectorAll("[data-section]")];
  const sectionNames = { overview: "Overview", "new-analysis": "New review", history: "Scan history", model: "Model & method" };
  for (const button of navButtons) {
    button.addEventListener("click", () => {
      const section = document.getElementById(button.dataset.section);
      if (!section) return;
      section.scrollIntoView({ behavior: "smooth", block: "start" });
      document.querySelectorAll(".side-nav .nav-link").forEach((nav) => {
        const active = nav.dataset.section === button.dataset.section;
        nav.classList.toggle("is-active", active || (button.dataset.section === "new-analysis" && nav.dataset.section === "overview"));
        if (active) nav.setAttribute("aria-current", "page");
        else nav.removeAttribute("aria-current");
      });
      $("page-crumb").textContent = sectionNames[button.dataset.section] || "Overview";
      closeMobileNav();
    });
  }

  const observer = new IntersectionObserver((entries) => {
    const visible = entries.filter((entry) => entry.isIntersecting).sort((a, b) => b.intersectionRatio - a.intersectionRatio)[0];
    if (!visible) return;
    const id = visible.target.id;
    const navTarget = id === "new-analysis" ? "overview" : id;
    const label = id === "new-analysis" ? "New review" : sectionNames[id];
    if (!label) return;
    $("page-crumb").textContent = label;
    document.querySelectorAll(".side-nav .nav-link").forEach((nav) => {
      const active = nav.dataset.section === navTarget;
      nav.classList.toggle("is-active", active);
      if (active) nav.setAttribute("aria-current", "page");
      else nav.removeAttribute("aria-current");
    });
  }, { rootMargin: "-18% 0px -68% 0px", threshold: [0, .1, .25] });
  ["overview", "new-analysis", "history", "model"].forEach((id) => observer.observe(document.getElementById(id)));
}

function closeMobileNav() {
  $("sidebar").classList.remove("is-open");
  $("mobile-scrim").hidden = true;
  $("mobile-menu").setAttribute("aria-expanded", "false");
}

function initMobileNav() {
  $("mobile-menu").addEventListener("click", () => {
    const isOpen = $("sidebar").classList.toggle("is-open");
    $("mobile-scrim").hidden = !isOpen;
    $("mobile-menu").setAttribute("aria-expanded", String(isOpen));
  });
  $("mobile-scrim").addEventListener("click", closeMobileNav);
}

function initUpload() {
  elements.fileInput.addEventListener("change", () => setSelectedFile(elements.fileInput.files?.[0]));
  elements.uploadForm.addEventListener("submit", handleUpload);
  $("remove-file").addEventListener("click", clearSelectedFile);
  elements.dropZone.addEventListener("dragover", (event) => { event.preventDefault(); elements.dropZone.classList.add("is-dragover"); });
  elements.dropZone.addEventListener("dragleave", () => elements.dropZone.classList.remove("is-dragover"));
  elements.dropZone.addEventListener("drop", (event) => {
    event.preventDefault();
    elements.dropZone.classList.remove("is-dragover");
    const file = event.dataTransfer?.files?.[0];
    if (file) setSelectedFile(file);
  });
}

function initHistory() {
  elements.tableBody.addEventListener("click", handleTableAction);
  elements.search.addEventListener("input", renderScans);
  elements.filter.addEventListener("change", renderScans);
  $("refresh-scans").addEventListener("click", loadWorkspace);
  $("manage-storage").addEventListener("click", () => document.getElementById("history").scrollIntoView({ behavior: "smooth", block: "start" }));
}

function initDialog() {
  $("close-dialog").addEventListener("click", () => elements.dialog.close());
  elements.dialog.addEventListener("click", (event) => {
    if (event.target === elements.dialog) elements.dialog.close();
  });
}

function initModelActions() {
  $("copy-setup").addEventListener("click", async () => {
    const command = "pip install -r requirements-ml.txt\npython scripts/train.py --data-dir data/processed --output models/brain_tumor_resnet50.pt";
    const feedback = $("copy-feedback");
    try {
      await navigator.clipboard.writeText(command);
      feedback.textContent = "Copied.";
    } catch {
      feedback.textContent = command;
    }
    window.setTimeout(() => { feedback.textContent = ""; }, 5000);
  });
}

setDate();
initNavigation();
initMobileNav();
initUpload();
initHistory();
initDialog();
initModelActions();
loadWorkspace();
