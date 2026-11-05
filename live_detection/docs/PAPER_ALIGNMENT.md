# Research paper alignment and explicit departures

Source: the uploaded **Real-Time Deep Fake Detection for Video Streaming** paper.
This is an implementation of its described approach, not a verified numerical
reproduction. The paper does not contain the checkpoint, training code, dataset
manifest, random seed, calibrated thresholds, or reproducible latency results.

| Topic | Paper | Implementation / limitation |
| --- | --- | --- |
| Spatial model | Abstract says ResNet50; methodology and supplied code say ResNeXt50_32x4d | Follows methodology and supplied code: ResNeXt50_32x4d. |
| Features | 2048 per frame | Adaptive average pooling to 2048. |
| Temporal model | One LSTM layer, 2048 hidden units | Correct batch-first LSTM; LeakyReLU and 0.4 dropout before the two-class linear layer. |
| Face size | 112 × 112 | Same; RGB and ImageNet normalization explicit. |
| Face extraction | Face-only crops; original code uses face_recognition | Portable Haar default; optional HOG profile. This is a deliberate implementation difference. Training and inference must use the same detector. |
| Temporal sampling | Methodology mentions first 150 frames at 30 fps; conclusion mentions 10 fps | Explicit checkpoint sample rate; paper-oriented prepare profile 60 frames at 30 fps, practical webcam profile 20 frames at 10 fps. Rates are not silently mixed. |
| Sequence lengths | 10, 20, 40, 60, 80, 100 | Supported as training configurations. Corrected inference uses the trained length. Legacy compatibility uses ten-frame reporting windows because its graph scores only the last frame. |
| Dataset | 6,000 total, 3,000 real and 3,000 fake; FaceForensics++, DFDC, Celeb-DF | Eight unlabeled user videos were supplied and analyzed. The paper dataset and its labeled splits are unavailable. Manifest checks identity/source group separation. |
| Split | 70% train, 30% test | Reserve independent validation/calibration within the development portion. No test-set threshold tuning. |
| Accuracy, precision, recall, F1 | 94.3%, 93.7%, 94.9%, 94.3% respectively | Paper-reported only. No new measured accuracy is claimed. Evaluation tools produce measured results when data and weights exist. |
| Live streaming | Intended for live and pre-recorded input | Actual browser-frame and direct-stream processing paths added. Live source and hardware validation remains necessary. |
| Real-time performance | General claim, no reproducible latency measurements | Reports processing latency and sampling windows. No guaranteed FPS or subsecond decision time. |
| Browser plug-in | Listed as future scope | Not implemented. This is a server/browser workspace, not a platform extension. |
| Other manipulations | Full-body detection listed as future scope | Facial video only; no audio or full-body detection. |

## Compatibility mode and the corrected research model

The locally installed supplied checkpoint runs the **original inference graph**, which has
no temporal dependence for one video. It uses BGR face crops, a bias-free LSTM,
and no LeakyReLU activation. The corrected RGB, batch-first model described in
the table is the separate training target in `tools.train`. Its LSTM and output
layer must be trained with labeled data. Haar face extraction, strict quality
checks, and fixed-grid 10-fps sampling are documented implementation differences
from the original upload pipeline. Claims of exact paper reproduction or repaired
classifier accuracy would be unsupported.

## Relevant primary documentation

- PyTorch LSTM: https://docs.pytorch.org/docs/2.8/generated/torch.nn.LSTM.html
- TorchVision ResNeXt50: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnext50_32x4d.html
- OpenCV capture flags: https://docs.opencv.org/4.13.0/d4/d15/group__videoio__flags__base.html
- PyTorch checkpoint loading: https://docs.pytorch.org/docs/2.8/notes/serialization.html
