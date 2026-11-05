// Browser capture lifecycle shared by webcam and screen video.
// Dependencies are injectable so cancellation and backpressure can be tested
// without granting access to a real camera or screen.
export class LiveCapture {
  constructor({video, model, acquire, sendWindow, onState, onWindow, onError,
    maxSeconds = 600, captureFrame, now = () => performance.now(),
    schedule = (fn, ms) => setInterval(fn, ms), cancel = id => clearInterval(id)}) {
    Object.assign(this, {video, model, acquire, sendWindow, onState, onWindow,
      onError, maxSeconds, now, schedule, cancel});
    this.captureFrame = captureFrame || (() => this.jpeg());
    this.epoch = 0;
    this.state = "idle";
    this.stream = null;
    this.timer = null;
    this.controller = null;
    this.frames = [];
    this.timestamps = [];
    this.inFlight = false;
  }

  notify(state, message) {
    this.state = state;
    this.onState({state, message, kind: this.kind,
      active: state === "starting" || state === "running"});
  }

  async start(kind) {
    if (this.state === "starting" || this.state === "running") return;
    const epoch = ++this.epoch;
    this.kind = kind;
    this.notify("starting", kind === "screen" ? "Choose the video tab or window…" : "Waiting for camera permission…");
    try {
      // Called before any await so screen sharing retains click activation.
      const stream = await this.acquire(kind);
      if (epoch !== this.epoch) {
        stream.getTracks().forEach(track => track.stop());
        return;
      }
      this.stream = stream;
      const track = stream.getVideoTracks()[0];
      if (!track || track.readyState === "ended") throw new Error("No active video track was shared.");
      track.addEventListener("ended", () => {
        if (epoch === this.epoch) this.stop("Sharing ended");
      }, {once: true});
      this.video.removeAttribute("src");
      this.video.srcObject = stream;
      this.video.muted = true;
      this.video.controls = false;
      await this.video.play();
      if (epoch !== this.epoch) return;
      this.started = this.now();
      this.notify("running", this.model.ready ? "Collecting live frames" : "Preview only · install the checkpoint to enable predictions");
      this.timer = this.schedule(() => this.tick(epoch), 1000 / this.model.sample_fps);
    } catch (error) {
      if (epoch !== this.epoch) return;
      this.stop("Capture not started");
      this.onError(error);
    }
  }

  stop(message = "Capture stopped") {
    ++this.epoch;
    if (this.timer !== null) this.cancel(this.timer);
    this.timer = null;
    this.controller?.abort();
    this.controller = null;
    this.stream?.getTracks().forEach(track => track.stop());
    this.stream = null;
    this.video.srcObject = null;
    this.video.controls = true;
    this.frames = [];
    this.timestamps = [];
    this.inFlight = false;
    this.notify("idle", message);
  }

  jpeg() {
    this.canvas ||= document.createElement("canvas");
    const {video, canvas} = this;
    const limit = this.kind === "screen" ? 960 : 640;
    const scale = Math.min(1, limit / video.videoWidth, limit / video.videoHeight);
    canvas.width = Math.max(1, Math.round(video.videoWidth * scale));
    canvas.height = Math.max(1, Math.round(video.videoHeight * scale));
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    let jpeg = canvas.toDataURL("image/jpeg", 0.7).split(",")[1];
    if (jpeg.length > 240000) jpeg = canvas.toDataURL("image/jpeg", 0.4).split(",")[1];
    if (jpeg.length > 240000) throw new Error("Shared image is too detailed. Share only the video window.");
    return jpeg;
  }

  tick(epoch) {
    if (epoch !== this.epoch || !this.stream) return;
    const elapsed = (this.now() - this.started) / 1000;
    if (elapsed >= this.maxSeconds) return this.stop("Session time limit reached");
    if (!this.model.ready || this.video.readyState < 2 || !this.video.videoWidth) return;
    try {
      this.frames.push(this.captureFrame());
      this.timestamps.push(elapsed);
      const n = this.model.sequence_length;
      if (this.frames.length > n) {
        this.frames.shift();
        this.timestamps.shift();
      }
      this.notify("running", this.inFlight ? "Analyzing · collecting latest frames" : `Capturing ${this.frames.length} / ${n}`);
      if (this.inFlight || this.frames.length !== n) return;
      const payload = {frames: this.frames, timestamps: this.timestamps};
      this.frames = [];
      this.timestamps = [];
      this.inFlight = true;
      this.controller = new AbortController();
      Promise.resolve(this.sendWindow(payload, this.controller.signal))
        .then(data => {
          if (epoch === this.epoch) this.onWindow(data, this.kind);
        })
        .catch(error => {
          if (epoch === this.epoch && error.name !== "AbortError") this.onError(error);
        })
        .finally(() => {
          if (epoch === this.epoch) {
            this.inFlight = false;
            this.controller = null;
          }
        });
    } catch (error) {
      this.stop("Capture interrupted");
      this.onError(error);
    }
  }
}
