import test from "node:test";
import assert from "node:assert/strict";
import {LiveCapture} from "../static/live-capture.mjs";

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => {resolve = a; reject = b;});
  return {promise, resolve, reject};
};
const flush = () => new Promise(resolve => setImmediate(resolve));
function fixture(options = {}) {
  const ended = new Map(), reports = [], errors = [], states = [], requests = [];
  let stopped = 0, clock = 0, tick;
  const track = {readyState: "live", stop() {stopped++;},
    addEventListener(name, fn) {ended.set(name, fn);}};
  const stream = {getTracks: () => [track], getVideoTracks: () => [track]};
  const video = {videoWidth: 640, readyState: 4, srcObject: null,
    removeAttribute() {}, play: async () => {}};
  const capture = new LiveCapture({video,
    model: {ready: true, sequence_length: 3, sample_fps: 10},
    acquire: async () => stream,
    sendWindow: (data, signal) => {requests.push({data, signal}); return Promise.resolve({label: "INCONCLUSIVE"});},
    captureFrame: () => "jpeg", now: () => clock,
    schedule: fn => {tick = fn; return 1;}, cancel: () => {tick = null;},
    onState: data => states.push(data), onWindow: data => reports.push(data),
    onError: error => errors.push(error), ...options});
  return {capture, stream, video, reports, errors, states, requests, ended,
    stopped: () => stopped,
    advance(count = 1) {for (let i = 0; i < count; i++) {clock += 100; tick?.();}}};
}

test("sends a model-sized window with increasing capture timestamps", async () => {
  const f = fixture(); await f.capture.start("screen");
  f.advance(2); assert.equal(f.requests.length, 0);
  f.advance(); await flush();
  assert.deepEqual(f.requests[0].data.timestamps, [0.1, 0.2, 0.3]);
  assert.equal(f.requests[0].data.frames.length, 3);
  assert.equal(f.reports.length, 1); f.capture.stop();
});

test("pending permission can be cancelled and a late stream is stopped", async () => {
  const pending = deferred(); const f = fixture({acquire: () => pending.promise});
  const started = f.capture.start("camera"); f.capture.stop();
  pending.resolve(f.stream); await started;
  assert.equal(f.stopped(), 1); assert.equal(f.video.srcObject, null);
  assert.equal(f.capture.state, "idle"); assert.equal(f.errors.length, 0);
});

test("stop aborts pending inference and ignores its late result", async () => {
  const pending = deferred(); let signal;
  const f = fixture({sendWindow: (_, s) => {signal = s; return pending.promise;}});
  await f.capture.start("screen"); f.advance(3); f.capture.stop();
  assert.equal(signal.aborted, true);
  pending.resolve({label: "LIKELY REAL"}); await flush();
  assert.equal(f.reports.length, 0); assert.equal(f.stopped(), 1);
});

test("a stale result cannot change a newly started capture", async () => {
  const pending = deferred();
  const f = fixture({sendWindow: () => pending.promise});
  await f.capture.start("camera"); f.advance(3); f.capture.stop();
  await f.capture.start("screen");
  pending.reject(new Error("old failure")); await flush();
  assert.equal(f.capture.kind, "screen"); assert.equal(f.capture.state, "running");
  assert.equal(f.errors.length, 0); f.capture.stop();
});

test("slow inference retains only the latest complete window", async () => {
  const pending = deferred(); let calls = 0;
  const f = fixture({sendWindow: () => {calls++; return pending.promise;}});
  await f.capture.start("screen"); f.advance(30);
  assert.equal(calls, 1); assert.equal(f.capture.frames.length, 3);
  assert.deepEqual(f.capture.timestamps, [2.8, 2.9, 3]);
  pending.resolve({}); await flush(); f.advance();
  assert.equal(calls, 2); f.capture.stop();
});

test("browser Stop sharing event releases resources", async () => {
  const f = fixture(); await f.capture.start("screen");
  f.ended.get("ended")(); f.advance(10);
  assert.equal(f.capture.state, "idle"); assert.equal(f.video.srcObject, null);
  assert.equal(f.requests.length, 0);
});

test("missing model permits preview without sending frames", async () => {
  const f = fixture({model: {ready: false, sample_fps: 10}});
  await f.capture.start("camera"); f.advance(40);
  assert.equal(f.video.srcObject, f.stream);
  assert.equal(f.requests.length, 0); assert.equal(f.capture.frames.length, 0);
  f.capture.stop();
});

test("repeated start while permission is pending opens only one prompt", async () => {
  const pending = deferred(); let prompts = 0;
  const f = fixture({acquire: () => {prompts++; return pending.promise;}});
  const started = f.capture.start("screen"); await f.capture.start("screen");
  pending.resolve(f.stream); await started;
  assert.equal(prompts, 1); f.capture.stop();
});

test("permission failure returns to idle and permits a retry", async () => {
  let count = 0; const f = fixture({acquire: async () => {
    if (!count++) throw new Error("Permission denied"); return f.stream;
  }});
  await f.capture.start("camera"); assert.equal(f.capture.state, "idle");
  assert.equal(f.errors.length, 1);
  await f.capture.start("camera"); assert.equal(f.capture.state, "running");
  f.capture.stop();
});

test("browser capture respects the configured session time limit", async () => {
  const f = fixture({maxSeconds: 1}); await f.capture.start("screen");
  f.advance(11); assert.equal(f.capture.state, "idle");
  assert.equal(f.states.at(-1).message, "Session time limit reached");
});
