const logStream = document.getElementById('logStream');
const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const modelSelect = document.getElementById('modelSelect');
const statusLabel = document.getElementById('statusLabel');
const statusDot = document.getElementById('statusDot');
const cameraFeed = document.getElementById('cameraFeed');
const debugBtn = document.getElementById('debugBtn');
const mouseBtn = document.getElementById('mouseBtn');
const quitBtn = document.getElementById('quitBtn');

const state = {
  running: false,
  model: modelSelect.value,
  pid: null,
  status: 'idle',
  controlPort: 8766,
  debugMode: true,
  mouseEnabled: false,
};

function timeLabel(iso) {
  const date = new Date(iso);
  return date.toLocaleTimeString([], { hour12: false });
}

function appendLog({ stream = 'info', message, time }) {
  const distanceFromBottom = logStream.scrollHeight - logStream.scrollTop - logStream.clientHeight;
  const shouldStickToBottom = distanceFromBottom < 32;

  const row = document.createElement('div');
  row.className = `log-line log-${stream}`;
  row.innerHTML = `<span class="log-time">[${timeLabel(time)}]</span> ${escapeHtml(message)}`;
  logStream.appendChild(row);

  if (shouldStickToBottom) {
    logStream.scrollTop = logStream.scrollHeight;
  }
}

function escapeHtml(text) {
  return String(text)
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function renderStatus() {
  statusLabel.textContent = state.running ? 'Active' : state.status === 'error' ? 'Error' : 'Idle';
  statusDot.classList.toggle('active', state.running);
  startBtn.disabled = state.running;
  stopBtn.disabled = !state.running;
  debugBtn.textContent = `Debug: ${state.debugMode ? 'On' : 'Off'}`;
  debugBtn.classList.toggle('active', state.debugMode);
  mouseBtn.textContent = `Mouse: ${state.mouseEnabled ? 'On' : 'Off'}`;
  mouseBtn.classList.toggle('active', state.mouseEnabled);
}

let frameLoopActive = false;
let lastFrameErrorLogAt = 0;
let previewCanvasSized = false;
let consecutiveFrameFailures = 0;

function throttledFrameError(message) {
  const now = Date.now();
  if (now - lastFrameErrorLogAt < 1500) {
    return;
  }
  lastFrameErrorLogAt = now;
  appendLog({ stream: 'stderr', message, time: new Date().toISOString() });
}

function startCameraFeedLoop() {
  if (frameLoopActive) return;
  frameLoopActive = true;

  const canvas = cameraFeed;
  const ctx = canvas.getContext('2d');
  if (!ctx) {
    frameLoopActive = false;
    return;
  }

  async function fetchAndRenderFrame() {
    if (!frameLoopActive || !state.running) return;

    try {
      const response = await fetch(`http://127.0.0.1:8765/frame?ts=${Date.now()}`);
      if (response.status === 404 || response.status === 503) {
        consecutiveFrameFailures += 1;
        setTimeout(fetchAndRenderFrame, 120);
        return;
      }
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const blob = await response.blob();
      const img = new Image();
      img.onload = () => {
        if (!previewCanvasSized) {
          canvas.width = img.width;
          canvas.height = img.height;
          previewCanvasSized = true;
        }
        ctx.drawImage(img, 0, 0);
        URL.revokeObjectURL(img.src);
      };
      img.onerror = () => {
        throttledFrameError('Failed to load frame image');
      };
      img.src = URL.createObjectURL(blob);
      consecutiveFrameFailures = 0;
    } catch (error) {
      consecutiveFrameFailures += 1;
      if (consecutiveFrameFailures % 20 === 0) {
        throttledFrameError(`Frame fetch failed: ${error.message}`);
      }
    }

    setTimeout(fetchAndRenderFrame, 50); // ~20 FPS
  }

  fetchAndRenderFrame();
}

function stopCameraFeedLoop() {
  frameLoopActive = false;
  previewCanvasSized = false;
}

function renderCameraFeed() {
  if (state.running) {
    startCameraFeedLoop();
  } else {
    stopCameraFeedLoop();
  }
}

async function fetchControlState() {
  const response = await fetch(`http://127.0.0.1:${state.controlPort}/state`);
  return response.json();
}

async function postControlState(payload) {
  const response = await fetch(`http://127.0.0.1:${state.controlPort}/control`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return response.json();
}

async function refreshStatus() {
  const backendState = await window.airpointer.getStatus();
  state.running = backendState.running;
  state.model = backendState.model;
  state.pid = backendState.pid;
  state.status = backendState.status;
  state.controlPort = backendState.controlPort || state.controlPort;
  modelSelect.value = backendState.model;
  renderStatus();
  renderCameraFeed();
  try {
    const controlState = await fetchControlState();
    state.debugMode = controlState.debug_mode;
    state.mouseEnabled = controlState.mouse_enabled;
    renderStatus();
  } catch (error) {
    appendLog({ stream: 'stderr', message: `Control API unavailable: ${error.message}`, time: new Date().toISOString() });
  }
}

startBtn.addEventListener('click', async () => {
  const model = modelSelect.value;
  const result = await window.airpointer.startBackend(model);
  if (!result.ok) {
    appendLog({ stream: 'stderr', message: result.error || 'Failed to start backend', time: new Date().toISOString() });
  }
  await refreshStatus();
});

stopBtn.addEventListener('click', async () => {
  await window.airpointer.stopBackend();
  await refreshStatus();
});

modelSelect.addEventListener('change', () => {
  state.model = modelSelect.value;
  renderStatus();
});

debugBtn.addEventListener('click', async () => {
  if (!state.running) return;
  state.debugMode = !state.debugMode;
  renderStatus();
  await postControlState({ debug_mode: state.debugMode });
});

mouseBtn.addEventListener('click', async () => {
  if (!state.running) return;
  state.mouseEnabled = !state.mouseEnabled;
  renderStatus();
  await postControlState({ mouse_enabled: state.mouseEnabled });
});

quitBtn.addEventListener('click', async () => {
  await window.airpointer.stopBackend();
  await window.airpointer.quitApp();
});

window.airpointer.onEvent((event) => {
  if (event.type === 'status') {
    state.running = event.state.running;
    state.model = event.state.model;
    state.pid = event.state.pid;
    state.status = event.state.status;
    state.controlPort = event.state.controlPort || state.controlPort;
    modelSelect.value = event.state.model;
    renderStatus();
    renderCameraFeed();
    return;
  }

  if (event.type === 'log') {
    appendLog(event);
  }
});

appendLog({ stream: 'info', message: 'AirPointer dashboard ready.', time: new Date().toISOString() });
refreshStatus();
renderCameraFeed();
