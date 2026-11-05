# Execution and deployment notes

The supported validation target for this package is one local research workspace
using one Waitress process. The server binds to 127.0.0.1 by default. Do not run
multiple workers against the same runtime directory: the in-memory registry,
model and startup cleanup are deliberately process-local.

Browser webcam access requires localhost or HTTPS. A phone accessing a plain
HTTP LAN address will usually not receive camera permission. HTTPS, browser
permissions and source-specific credentials must be configured in the user's
environment. An ordinary platform watch-page URL is not a valid media input.

For public/multi-user deployment, implement real authentication, persistent job
queues, shared storage and resource quotas; isolate untrusted video decoding and
restrict worker network egress. CSRF and unguessable session-scoped jobs do not
constitute user authentication. The removed JavaScript password screen did not
provide backend protection.

The stream host allowlist is an explicit operator configuration. OpenCV/FFmpeg
handles redirects, HLS playlists and media subrequests below the Python layer.
Only trusted media endpoints should be enabled; deploy an egress proxy/firewall
when stronger network isolation is required. Do not expose this app publicly
with a broad allowlist. RTSP credentials may appear in native decoder diagnostic
logs; keep those logs private. JSON reports omit the source stream URL.

Open/read timeouts are supplied through the FFmpeg capture backend. The code
does not fall back to an unbounded remote capture if this backend is unavailable.
Stream decoding runs separately from inference, with bounded sampled-frame
storage and dropped-frame counts. The session can be stopped; a blocking native
read may take up to its configured timeout to exit. Automatic reconnection is
not enabled; an ended or timed-out feed must be restarted.

Raw uploads are deleted after processing. Runtime previews and JSON reports are
removed on an explicit DELETE, on later API cleanup after expiry, or at server
restart. Camera and screen frames are processed in memory and are not saved as raw images.
The browser retains at most 200 live browser window results, which can be exported.

GPU acceleration can be selected with `DEVICE=cuda`; install the corresponding
PyTorch/torchvision builds for the user's CUDA environment. CPU is the portable
default. Apple MPS may be selected explicitly if supported by the local install,
but that device path was not tested here. The Dockerfile is a CPU convenience
configuration; only local Python execution was exercised during this build.
