// Record the Gatekeeper live demo page via CDP screencast (headless shell).
// Timeline (~29.5s, matches S3 narration 28.75s + pad):
//   0-9.5s  preset allow  ($49 -> 5.7% ALLOW)
//   9.5-19.5s preset review ($2,000 3am new -> 45.0% REVIEW, queue shows)
//   19.5-29.5s preset block ($12,000 burst -> 78.8% BLOCK)
// Frames are saved as JPEGs with timestamps for later assembly.
import { spawn } from 'node:child_process';
import { writeFileSync, mkdirSync } from 'node:fs';
import { setTimeout as sleep } from 'node:timers/promises';

const CHROME = process.env.HOME + '/.cache/ms-playwright/chromium_headless_shell-1248/chrome-headless-shell-linux64/chrome-headless-shell';
const PORT = 9223;
const PAGE_URL = 'http://127.0.0.1:8765/gatekeeper-live-demo/index.html';
const OUT = new URL('./frames/', import.meta.url).pathname;
mkdirSync(OUT, { recursive: true });

const sleepMs = (ms) => sleep(ms);
async function waitFor(pred, timeout = 20000, step = 250) {
  const t0 = Date.now();
  while (Date.now() - t0 < timeout) { try { if (await pred()) return true; } catch {} await sleepMs(step); }
  throw new Error('timeout waiting');
}

const chrome = spawn(CHROME, [
  '--headless', '--disable-gpu', '--no-sandbox',
  '--window-size=1920,1080', '--hide-scrollbars',
  `--remote-debugging-port=${PORT}`, '--force-device-scale-factor=1',
  'about:blank',
], { stdio: 'ignore' });

try {
  await waitFor(async () => { const r = await fetch(`http://127.0.0.1:${PORT}/json/version`); return r.ok; });
  const created = await fetch(`http://127.0.0.1:${PORT}/json/new?${encodeURIComponent(PAGE_URL)}`, { method: 'PUT' });
  const target = await created.json();
  const ws = new WebSocket(target.webSocketDebuggerUrl);
  await new Promise((res, rej) => { ws.onopen = res; ws.onerror = rej; });

  let id = 0; const pending = new Map();
  const frames = []; let t0 = null;
  const events = [];
  ws.onmessage = (ev) => {
    const m = JSON.parse(ev.data);
    if (m.id && pending.has(m.id)) { const { res, rej } = pending.get(m.id); pending.delete(m.id); m.error ? rej(new Error(JSON.stringify(m.error))) : res(m.result); }
    else if (m.method === 'Page.screencastFrame') {
      const now = Date.now();
      if (t0 === null) t0 = now;
      frames.push({ t: (now - t0) / 1000, data: m.params.data, sessionId: m.params.sessionId });
      ws.send(JSON.stringify({ id: ++id, method: 'Page.screencastFrameAck', params: { sessionId: m.params.sessionId } }));
    } else if (m.method) { events.push(m.method); }
  };
  const send = (method, params = {}) => new Promise((res, rej) => { const i = ++id; pending.set(i, { res, rej }); ws.send(JSON.stringify({ id: i, method, params })); });
  const evalJs = async (expr) => (await send('Runtime.evaluate', { expression: expr, returnByValue: true })).result.value;

  await send('Page.enable'); await send('Runtime.enable');
  // wait for load + JS render
  await waitFor(async () => await evalJs(`document.readyState === 'complete' && !!document.querySelector('[data-preset="allow"]')`), 20000);
  await sleepMs(800);
  // zoom out so the whole demo panel (presets, verdict, factors, queue) fits
  // in a 1080px viewport, then scroll instantly to the demo section top
  console.log('scroll:', await evalJs(`(() => { document.documentElement.style.scrollBehavior='auto'; document.body.style.zoom='0.8'; const el = document.getElementById('live-demo'); const y = el.getBoundingClientRect().top + window.scrollY - 10; window.scrollTo(0, y); return Math.round(window.scrollY); })()`));
  await sleepMs(400);
  console.log('scrollAfter:', await evalJs(`Math.round(window.scrollY)`));
  console.log('queueRect:', await evalJs(`(() => { const q = document.getElementById('queue'); const r = q.getBoundingClientRect(); return Math.round(r.top) + '..' + Math.round(r.bottom); })()`));
  console.log('initial:', await evalJs(`document.getElementById('scoreBig').textContent + ' ' + document.getElementById('decisionTitle').textContent.trim()`));
  await sleepMs(500);

  await send('Page.startScreencast', { format: 'jpeg', quality: 80, maxWidth: 1920, maxHeight: 1080, everyNthFrame: 1 });
  t0 = Date.now();
  const click = async (name) => {
    console.log(name, await evalJs(`(() => { document.querySelector('[data-preset="${name}"]').click(); return document.getElementById('scoreBig').textContent + ' ' + document.getElementById('decisionTitle').textContent.trim() + ' queue=' + document.getElementById('queue').classList.contains('show'); })()`));
  };
  await click('allow');
  await sleepMs(9500);
  await click('review');
  await sleepMs(10000);
  await click('block');
  await sleepMs(10000);
  await send('Page.stopScreencast');
  await sleepMs(500);

  // save frames (dedupe identical consecutive bytes is left to assembly; save all)
  frames.forEach((f, i) => writeFileSync(`${OUT}frame-${String(i).padStart(4, '0')}-${f.t.toFixed(2)}.jpg`, Buffer.from(f.data, 'base64')));
  const meta = { count: frames.length, duration_s: frames.length ? frames[frames.length - 1].t : 0, times: frames.map(f => +f.t.toFixed(2)) };
  writeFileSync(new URL('./frames-meta.json', import.meta.url).pathname, JSON.stringify(meta, null, 2));
  console.log('FRAMES', meta.count, 'duration', meta.duration_s.toFixed(2), 's');
  ws.close();
} finally {
  chrome.kill('SIGKILL');
}
console.log('done');
