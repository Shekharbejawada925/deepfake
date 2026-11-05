# Verification for the live-video integration

Checked on 2026-11-05 with Python 3.12, PyTorch 2.8.0+cpu,
TorchVision 0.23.0+cpu, OpenCV 4.11.0 and Node.js.

## Automated regression checks

```bash
python -m pytest -q
node --test tests/live-capture.test.mjs
```

- **52 Python tests passed**: model shape and temporal behavior, original-graph
  equivalence, preprocessing, missing/ambiguous face abstention, upload jobs,
  private reports, CSRF, complete/incomplete windows, safe checkpoint installation,
  stream cancellation and actual local HTTP video decoding through FFmpeg.
- **10 JavaScript tests passed**: webcam/screen lifecycle, increasing timestamps,
  model-sized windows, one outstanding request, bounded buffering, cancellation
  while awaiting permission, stale results after stop/restart, browser track-ended
  handling, denied permission, preview without a model and session time limits.
- JavaScript syntax checks passed.
- The live-window API was also exercised with the actual supplied checkpoint
  loaded: ten blank JPEG frames returned HTTP 200, `INCONCLUSIVE`, `p_fake: null`,
  and `no_face: 10`. `/live`, the setup page and JavaScript modules served correctly.

API tests use explicit model/extractor test doubles where deterministic scores
are needed. Model tests use random weights to check mathematical properties.
Neither proves classification accuracy.

## Supplied checkpoint

The actual supplied checkpoint was installed locally and loaded successfully
using its SHA-256 profile and strict model-state checks. A forward pass on a
synthetic image crop returned a finite score; the measured CPU inference call
was 110.5 ms on this execution host. This isolated timing excludes capture,
face detection, queuing, transfer and browser rendering, and is not a sustained
real-time benchmark. Synthetic crops are not real/fake validation data.

## Optional browser smoke check

`tests/browser_smoke.py` exercises synthetic screen sharing, Chromium's fake
webcam, JSON export, stop controls and desktop/mobile overflow against the real
application. Install Playwright and its Chromium browser separately to run it:

```bash
python -m pip install playwright==1.51.0
python -m playwright install chromium
python tests/browser_smoke.py
```

This check **could not run in the execution environment**: Chromium startup was
blocked by an operating-system socket permission restriction. It is not counted
among the 62 passing tests. No visual/browser pass is claimed for this release.

## Remaining validation

Real/fake accuracy, the output class order and decision thresholds remain
unverified against labeled held-out data. The supplied model uses only the last
sampled face in each window. Corrected temporal modeling requires retraining.

Physical camera behavior, browser screen-picker permission, OS permission prompts,
DRM playback, RTSP authentication and production deployment need testing on the
intended device and feed. The automated transport check covers local HTTP media;
it does not certify every codec, HLS server or RTSP camera.
