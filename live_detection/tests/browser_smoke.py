"""Optional UI smoke test. Requires Playwright, Chromium and installed weights.

Run from live_detection: python tests/browser_smoke.py
Screen capture uses a synthetic canvas stream; webcam uses Chromium's fake
device. These checks do not replace real browser/OS permission testing.
"""
import json
import os
from pathlib import Path
import sys
import threading

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepfake.web import create_app
from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server


def main():
    app = create_app()
    assert app.extensions["detector"].ready, app.extensions["detector"].error
    server = make_server("127.0.0.1", 0, app, threaded=True)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        with sync_playwright() as p:
            options = {"headless": True, "args": ["--no-sandbox",
                       "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"]}
            if os.getenv("BROWSER_EXECUTABLE"):
                options["executable_path"] = os.environ["BROWSER_EXECUTABLE"]
            browser = p.chromium.launch(**options)
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            page.add_init_script("""(() => {
              Object.defineProperty(navigator.mediaDevices, 'getDisplayMedia', {
                value: async () => {
                  const canvas = document.createElement('canvas');
                  canvas.width = 1280; canvas.height = 720;
                  const ctx = canvas.getContext('2d');
                  ctx.fillStyle = '#334455'; ctx.fillRect(0, 0, 1280, 720);
                  const stream = canvas.captureStream(15);
                  window.sharedTrack = stream.getVideoTracks()[0];
                  const timer = setInterval(() => {
                    ctx.fillRect(0, 0, 1280, 720);
                    if (window.sharedTrack.readyState === 'ended') clearInterval(timer);
                  }, 60);
                  return stream;
                }
              });
            })();""")
            page.goto(f"http://127.0.0.1:{server.server_port}/live")
            page.wait_for_function("document.querySelector('#model-state').textContent.includes('loaded')")
            page.click("#tab-screen")
            with page.expect_response(lambda r: r.url.endswith("/api/live/window"), timeout=15000) as captured:
                page.click("#screen-start")
            response = captured.value
            assert response.status == 200
            assert response.json()["p_fake"] is None  # generated media contains no face
            page.wait_for_function("document.querySelectorAll('#timeline-body tr').length > 0")
            page.click("#stop")
            assert page.evaluate("window.sharedTrack.readyState") == "ended"
            with page.expect_download() as exported:
                page.click("#download")
            report = json.loads(Path(exported.value.path()).read_text())
            assert report["kind"] == "screen" and report["timeline"]
            assert "frames" not in report
            page.click("#tab-camera")
            with page.expect_response(lambda r: r.url.endswith("/api/live/window"), timeout=15000) as webcam:
                page.click("#camera-start")
            assert webcam.value.status == 200
            page.click("#stop")
            for size in ({"width": 1440, "height": 1000}, {"width": 390, "height": 844}):
                page.set_viewport_size(size)
                assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
            assert not errors, errors
            browser.close()
            print(json.dumps({"screen_window": "passed", "webcam_window": "passed",
                  "stop_releases_track": "passed", "json_export": "passed",
                  "desktop_and_mobile_layout": "passed", "console_errors": errors}))
    finally:
        server.shutdown(); server.server_close(); worker.join(timeout=2)


if __name__ == "__main__":
    main()
