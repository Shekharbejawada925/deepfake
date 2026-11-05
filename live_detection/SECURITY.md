# Security and deployment boundaries

This experimental app binds to 127.0.0.1 and is **not** ready for public exposure.
Do not expose the Flask development server to the internet.
- Camera permission is granted by the browser; captures are sent to localhost and not written to disk by this code.
- Inference frames contain potentially sensitive biometric data: avoid uploading them to third-party services without consent.
- The API limits total request size, frame bytes, count and dimensions but lacks authentication, quotas, CSRF and per-user concurrency limits.
- Use a production ASGI/WSGI server, HTTPS, authentication, rate limiting, and load tests for deployment.
- Never commit model weights, access tokens, private datasets or .env files.
- A score is not evidence that a person is lying or a recording is genuine. Validate on unseen sources.
- For this model, class index 0 is **assumed fake**, based on the original Flask source; independently check training labels.
- The original legacy LSTM does not correctly model a length-8 sequence at batch size 1. Retrain the corrected model before describing the system as temporal detection.
