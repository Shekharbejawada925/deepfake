# API guide

All endpoints are on the same origin as the app. First `GET /api/status` to obtain
the session cookie and `csrf` value. Send that value as `X-CSRF-Token` on every
POST/DELETE. Keep the cookie. Jobs are visible only within their creating browser
session. No account authentication is supplied.

| Method | Route | Request / response |
| --- | --- | --- |
| GET | `/api/health` | Server liveness and `model_ready` |
| GET | `/api/status` | Model contract summary, limits, stream enablement, CSRF token |
| POST | `/api/uploads` | Multipart `file`; 202 with job `id` |
| POST | `/api/streams` | JSON `{"url":"rtsp://approved-host/live"}`; 202 with job `id` |
| GET | `/api/jobs/<id>` | State, timeline, coverage, summary and processing times |
| POST | `/api/jobs/<id>/stop` | Cancel capture/processing; poll until terminal state |
| DELETE | `/api/jobs/<id>` | Delete a completed, stopped or failed job and its files |
| GET | `/api/jobs/<id>/preview.jpg` | Latest sampled frame; 404 until available |
| GET | `/api/jobs/<id>/report` | Download a JSON report of the current state |
| POST | `/api/live/window` (or `/api/camera/window`) | JSON `frames` (base64 JPEG array) and `timestamps` (seconds); one window result |

Camera requests must contain exactly the checkpoint's sequence length, no data-URL
prefix, JPEG images up to 960 pixels per side, and finite nonnegative timestamps.
Payloads above 8 MiB are rejected when content length is present. The global
request cap still applies to all bodies. UI capture preserves aspect ratio, with at most 640 pixels per side for webcam
and 960 for screen capture. Inconclusive windows return 200 with `p_fake: null` and quality reasons.
There is no fake prediction fallback.

Common status codes: 400 invalid input, 403 expired/missing CSRF, 404 private or
expired job, 413 request too large, 429 capacity limit, 503 unavailable model.

Job states: `queued`, `running`, `complete`, `stopped`, `error`. Histories and
previews are ephemeral, bounded to 100 retained jobs, and expire on later access
after the configured retention period. Restarting the single server process
removes prior runtime job folders.


## Compatibility status and score scope

`/api/status` includes `compatibility_mode`, `warning`, `architecture`,
`label_mapping_verified`, `color_space` and `temporal_behavior`. The bundled
profile reports `last_frame_only`, BGR and an unverified class mapping.
Every window includes `score_scope` and `compatibility_mode`. Preserve these
fields when showing or exporting results; `p_fake` is not measured accuracy.

Camera requests still contain the full reporting window (ten frames for the
bundled model). Partial windows at the end of an uploaded video can score only
in legacy mode. The complete JSON download includes model status and warning.

The old `/live` page URL still works. The scaffold `/api/live/predict` route is
replaced by `/api/live/window`, which requires model-sized windows, timestamps
and CSRF. Use `/api/status` for the current model contract; fixed eight-frame
requests from the old scaffold are not compatible.
