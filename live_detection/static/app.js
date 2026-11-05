"use strict";
import {LiveCapture} from "./live-capture.mjs";
const $ = id => document.getElementById(id);
let status, csrf, mode = "upload", jobId = null, pollTimer = null;
let capture = null, history = [], reportData = null, objectUrl = null;

function showError(message) { $("error").textContent = message; $("error").hidden = false; }
function clearError() { $("error").hidden = true; }
async function api(path, options = {}) {
  const headers = {...(options.headers || {})};
  if ((options.method || "GET") !== "GET") headers["X-CSRF-Token"] = csrf;
  const response = await fetch(path, {...options, headers});
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request could not be completed.");
  return data;
}
function busy(active) {
  document.querySelectorAll(".tab").forEach(x => { x.disabled = active; });
  $("analyze").disabled = active || !status?.model.ready || !$("file").files.length;
  $("stream-start").disabled = active || !status?.model.ready || !status?.streaming_enabled;
  $("camera-start").disabled = active || !status;
  $("screen-start").disabled = active || !status;
  $("file").disabled = active;
  $("stop").hidden = !active;
}
function resetResults() {
  history = []; reportData = null; $("download").disabled = true;
  $("verdict").textContent = "Collecting video"; $("verdict").className = "";
  $("verdict-detail").textContent = "Waiting for one complete window with a clear, continuous face.";
  $("timeline-body").replaceChildren(); $("timeline").hidden = true; $("timeline-empty").hidden = false;
  $("score").textContent = $("coverage").textContent = $("latency").textContent = "—";
  $("window-count").textContent = "0 windows"; $("timing-note").textContent = "";
}
function render(windows, summary = null) {
  history = windows;
  const latest = windows.at(-1);
  if (!latest) return;
  const label = summary?.label || latest.label;
  $("verdict").textContent = label.toLowerCase().replace(/^./, c => c.toUpperCase());
  $("verdict").className = label.includes("FAKE") || label.includes("FLAGGED") ? "fake" : label === "LIKELY REAL" ? "real" : "uncertain";
  $("result-scope").textContent = summary ? "SUMMARY OF SAMPLED WINDOWS" : "LATEST COMPLETED LIVE WINDOW";
  const reasons = Object.keys(latest.reasons).map(x => x.replaceAll("_", " "));
  $("verdict-detail").textContent = reasons.length ? "Latest window: " + reasons.join(", ") + ". No prediction was issued for this window." : status.model.compatibility_mode ? "Provisional result from the last sampled face in each window. The supplied model’s class mapping is unverified; its score does not prove authenticity." : "Window scores reflect the detector’s assessment of visible facial patterns. Review the timeline for changes across the video.";
  $("score").textContent = latest.p_fake === null ? "—" : (latest.p_fake * 100).toFixed(1) + "%";
  $("coverage").textContent = windows.filter(x => x.p_fake !== null).length + " / " + windows.length;
  $("latency").textContent = (latest.processing_ms / 1000).toFixed(2) + " s";
  $("window-count").textContent = windows.length + " windows";
  $("timeline").hidden = false; $("timeline-empty").hidden = true;
  const fragment = document.createDocumentFragment();
  windows.slice(-200).reverse().forEach(row => {
    const tr = document.createElement("tr");
    const cells = [row.start_seconds.toFixed(1) + "–" + row.end_seconds.toFixed(1), row.label,
      row.p_fake === null ? "—" : (row.p_fake * 100).toFixed(1) + "%",
      Object.keys(row.reasons).map(x => x.replaceAll("_", " ")).join(", ") || row.usable_faces + " face frames"];
    cells.forEach(value => { const td = document.createElement("td"); td.textContent = value; tr.append(td); });
    fragment.append(tr);
  });
  $("timeline-body").replaceChildren(fragment); $("download").disabled = false;
}
async function followJob(id) {
  jobId = id;
  const poll = async () => {
    try {
      const data = await api("/api/jobs/" + id);
      reportData = data; render(data.timeline, data.summary);
      $("session-state").textContent = data.state + " · " + data.processed_seconds.toFixed(1) + " s sampled";
      $("timing-note").textContent = data.message + ". Dropped stream samples: " + data.dropped_samples + ".";
      if (data.kind === "stream" && data.processed_seconds > 0) {
        $("stream-preview").src = "/api/jobs/" + id + "/preview.jpg?t=" + Date.now();
        $("stream-preview").hidden = false; $("preview-empty").hidden = true;
      }
      if (["complete", "stopped", "error"].includes(data.state)) {
        busy(false); pollTimer = null;
        if (data.state === "error") showError(data.message);
        if (!data.timeline.length) { $("verdict").textContent = "No result"; $("verdict-detail").textContent = data.message; }
        return;
      }
      pollTimer = setTimeout(poll, 1000);
    } catch (error) { showError(error.message); busy(false); pollTimer = null; }
  };
  await poll();
}
document.querySelectorAll(".tab").forEach(tab => tab.addEventListener("click", () => {
  mode = tab.dataset.mode;
  document.querySelectorAll(".tab").forEach(x => { x.classList.toggle("active", x === tab); x.setAttribute("aria-selected", String(x === tab)); });
  ["upload", "camera", "screen", "stream"].forEach(x => { $("panel-" + x).hidden = x !== mode; });
  $("video").pause(); $("video").hidden = true; $("stream-preview").hidden = true; $("preview-empty").hidden = false;
  if (mode === "upload" && objectUrl) { $("video").src = objectUrl; $("video").hidden = false; $("preview-empty").hidden = true; }
  clearError();
}));
$("file").addEventListener("change", () => {
  const file = $("file").files[0];
  if (!file) return;
  if (objectUrl) URL.revokeObjectURL(objectUrl);
  objectUrl = URL.createObjectURL(file);
  $("filename").textContent = file.name;
  $("video").srcObject = null; $("video").src = objectUrl; $("video").muted = false;
  $("video").hidden = false; $("preview-empty").hidden = true;
  $("analyze").disabled = !status?.model.ready;
});
$("analyze").addEventListener("click", async () => {
  clearError(); const file = $("file").files[0];
  if (!file) return showError("Choose a video first.");
  if (file.size > status.limits.upload_mb * 1024 * 1024) return showError("This video exceeds the upload limit.");
  busy(true); resetResults();
  try { const body = new FormData(); body.append("file", file); const data = await api("/api/uploads", {method:"POST", body}); await followJob(data.id); }
  catch (error) { showError(error.message); busy(false); }
});
$("stream-start").addEventListener("click", async () => {
  clearError(); busy(true); resetResults();
  try { const data = await api("/api/streams", {method:"POST", headers:{"Content-Type":"application/json"}, body:JSON.stringify({url:$("stream-url").value.trim()})}); await followJob(data.id); }
  catch (error) { showError(error.message); busy(false); }
});
function acquireVideo(kind) {
  if (kind === "screen") {
    if (!navigator.mediaDevices?.getDisplayMedia) throw new Error("Screen sharing needs a supported desktop browser on localhost or HTTPS.");
    return navigator.mediaDevices.getDisplayMedia({video: true, audio: false});
  }
  if (!navigator.mediaDevices?.getUserMedia) throw new Error("Webcam needs localhost or HTTPS and a supported browser.");
  return navigator.mediaDevices.getUserMedia({video: {width: {ideal: 640}, height: {ideal: 480}}, audio: false});
}
function captureError(error) {
  const message = error.name === "NotAllowedError" ? "Capture was cancelled or permission was denied. Start again and allow the source you want to check."
    : error.name === "NotReadableError" ? "The video source is unavailable. Check camera or screen-recording permission in your operating system."
    : error.message || "Video capture was not available.";
  showError(message);
}
function startCapture(kind) {
  clearError();
  if (!capture) return showError("Wait for the model status before starting.");
  jobId = null;
  resetResults();
  $("video").hidden = false;
  $("stream-preview").hidden = true;
  $("preview-empty").hidden = true;
  // Call immediately from the click handler: screen capture requires activation.
  void capture.start(kind);
}
$("camera-start").addEventListener("click", () => startCapture("camera"));
$("screen-start").addEventListener("click", () => startCapture("screen"));
$("stop").addEventListener("click", async () => {
  if (capture?.state === "starting" || capture?.state === "running") return capture.stop();
  if (jobId) { try { await api("/api/jobs/" + jobId + "/stop",{method:"POST"}); $("session-state").textContent = "Stopping…"; } catch (error) { showError(error.message); } }
});
$("download").addEventListener("click", () => {
  if (!["camera", "screen"].includes(reportData?.kind) && jobId) { window.location.href = "/api/jobs/" + jobId + "/report"; return; }
  const blob = new Blob([JSON.stringify(reportData,null,2)],{type:"application/json"});
  const url = URL.createObjectURL(blob), anchor = document.createElement("a"); anchor.href = url; anchor.download = (reportData?.kind || "live") + "-analysis.json"; anchor.click(); setTimeout(() => URL.revokeObjectURL(url),1000);
});
window.addEventListener("pagehide", () => {
  capture?.stop();
  if (pollTimer && jobId) fetch("/api/jobs/" + jobId + "/stop",{method:"POST",headers:{"X-CSRF-Token":csrf},keepalive:true}).catch(() => {});
});
(async () => {
  try {
    status = await api("/api/status"); csrf = status.csrf;
    capture = new LiveCapture({
      video: $("video"), model: status.model, acquire: acquireVideo,
      maxSeconds: status.limits.stream_seconds,
      sendWindow: (payload, signal) => api("/api/live/window", {
        method: "POST", headers: {"Content-Type": "application/json"},
        body: JSON.stringify(payload), signal
      }),
      onState: ({active, state, message}) => {
        busy(active);
        $("session-state").textContent = message;
        if (state === "running" && !status.model.ready) {
          $("verdict").textContent = "Preview only";
          $("verdict-detail").textContent = "Install the trained checkpoint and restart the server to enable predictions.";
        }
      },
      onWindow: (data, kind) => {
        clearError();
        history = [...history, data].slice(-200);
        render(history);
        reportData = {kind, timeline: history, model: status.model};
        $("timing-note").textContent = "Latest completed window: " + data.start_seconds.toFixed(1) + "–" + data.end_seconds.toFixed(1) + " s. History retains the most recent 200 windows.";
      },
      onError: captureError
    });
    $("model-state").textContent = status.model.ready ? (status.model.compatibility_mode ? "Supplied checkpoint loaded" : "Trained model loaded") : "Model setup required";
    $("model-dot").classList.toggle("ready",status.model.ready);
    $("model-details").textContent = status.model.ready ? status.model.sequence_length + " frames · " + status.model.sample_fps + " fps · " + status.model.device : "Predictions are disabled until weights are installed";
    $("setup-banner").hidden = status.model.ready; $("setup-message").textContent = status.model.message || "";
    if (status.model.ready && !status.model.calibrated) { $("setup-banner").hidden = false; $("setup-banner").querySelector("strong").textContent = "Thresholds are provisional"; $("setup-message").textContent = "This checkpoint has not been calibrated on validation data."; }
    if (status.model.ready && status.model.compatibility_mode) { $("setup-banner").hidden = false; $("setup-banner").querySelector("strong").textContent = "Legacy model · provisional predictions"; $("setup-message").textContent = status.model.warning; }
    $("upload-limits").textContent = "Up to " + status.limits.upload_mb + " MB · analyzes the first " + status.limits.video_seconds + " seconds";
    $("stream-help").textContent = status.streaming_enabled ? "Session limit: " + status.limits.stream_seconds + " seconds." : "Enable a trusted stream host in the setup guide before connecting.";
    busy(false);
  } catch (error) { showError(error.message); $("model-state").textContent = "Server unavailable"; }
})();
