# Audit of the uploaded code

The source audit below describes the corrected training path. The supplied
checkpoint now confirms the bias-free LSTM and supports an explicit original-
graph compatibility path. Eight unlabeled clips were analyzed. Their scores show
strong within-video variation on several inputs, but without true labels we
cannot determine accuracy or identify which predictions are false positives.
The personal-video diagnostics remain in the saved project ZIP; they are not published here. See `VALIDATION.md` for this release’s verification.

| Finding | Original behavior | Rebuild |
| --- | --- | --- |
| LSTM sequence dimension | Tensor is `[batch,time,2048]`, but LSTM defaults to `batch_first=False`. For a single-video batch, it sees one timestep and treats frames as independent batch items. | Explicit `batch_first=True`; regressions verify early-frame influence and batch independence. Retraining may be required. |
| Positional LSTM argument | `nn.LSTM(latent_dim,hidden_dim,lstm_layers,bidirectional)` passes the fourth argument as `bias`, not `bidirectional`. | Explicit named arguments, one layer, bias enabled, unidirectional. |
| Color inconsistency | OpenCV BGR frames enter PIL/ImageNet normalization and face detection without RGB conversion; display extraction converts RGB separately. | One RGB preprocessing implementation shared across training and all inference modes. |
| Label assumptions | Output index 1 is unconditionally called REAL without checkpoint metadata. | Explicit class order in a validated checkpoint contract. |
| Missing/bad faces | Exceptions fall through to classifying the entire frame. | No-face, multi-face, tiny/blurred/poorly lit faces and track jumps abstain. |
| Misleading certainty | A softmax score is assigned to a variable named `accuracy`; there is no evaluation procedure. | UI says model score; validation calibration and held-out evaluation are separate tools. |
| Frame mismatch | Inference uses the first requested frames; displayed frames are sampled differently over the clip. | Results and preview use the same chronological windows. |
| Short video crash | `total_frames // scroll_value` can be zero; modulo by zero then fails. | Timestamp-based iteration; incomplete windows remain unscored. |
| Missing streaming backend | Form has `video_url`, but `/upload` always requires a file. | Separate upload, camera-window and direct-stream APIs. |
| Per-request model loading | Builds ResNeXt, potentially downloads ImageNet weights and reloads the checkpoint for every upload. | Load a trained checkpoint once at startup; inference mode and shared model lock; chunked CNN encoding. |
| Shared output filenames | `frame_0.jpg`, `face_0.jpg`, uploaded names and `/content/1.png` may collide or fail. | Private UUID job directories and atomically updated previews; source files removed after processing. |
| Invalid localization claim | Dot product of CNN feature maps and a classifier trained after an LSTM is not a valid temporal attribution method. | Removed the unsupported heatmap. Bounding boxes mark the detected face, not manipulation localization. |
| Frontend mismatch | `main1.js` calls `/predict`, absent from Flask. Several scripts reference missing plugin assets. | One HTML/CSS/JS application using the implemented API; no CDN dependencies. |
| Login and static assets | Credentials are checked only in browser JavaScript; the new uploaded `index.html` is a Bootstrap icon catalogue, not the project homepage. CSS references absent images/fonts. | Removed the misleading login and unrelated template assets. Local workspace with session-separated jobs, no account system. |

The original source and paper remain in the saved project ZIP. Personal videos,
original frontend templates, and research-paper binaries are not redistributed
in this source repository.

Reference for LSTM signature and shapes:
https://docs.pytorch.org/docs/2.8/generated/torch.nn.LSTM.html


## Findings after checkpoint and sample-video inspection

- All 322 checkpoint entries are finite tensors and match the legacy architecture.
- The original graph and the optimized final-frame path produced identical logits
  for the same extracted WhatsApp face features. This does not verify class labels.
- BGR/PIL preprocessing is preserved for legacy inference; corrected training uses
  RGB/OpenCV preprocessing. These paths are recorded rather than silently mixed.
- Fixed-grid sampling corrects cumulative undersampling: 25-fps input now supplies
  30 frames over three seconds at 10 sampled fps, instead of approximately 25.
- Legacy reporting uses ten frames with partial uploads allowed. Corrected LSTM
  models retain full-sequence checks. All modes still reject unusable face evidence.
- Several clips show both near-zero and near-one frame scores. A confident score
  on one face cannot establish that the full source is genuine or manipulated.
