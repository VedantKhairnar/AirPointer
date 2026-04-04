const { app, BrowserWindow, ipcMain } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

let mainWindow = null;
let backendProcess = null;
let stdoutBuffer = '';
let stderrBuffer = '';

const projectRoot = path.resolve(__dirname, '..');
const pythonCandidates = [
  process.env.AIRPOINTER_PYTHON,
  path.join(projectRoot, '..', '.venv', 'bin', 'python'),
  path.join(projectRoot, '..', '.venv', 'Scripts', 'python.exe'),
].filter(Boolean);

const previewPort = 8765;
const controlPort = previewPort + 1;

const backendState = {
  running: false,
  model: 'mediapipe',
  pid: null,
  status: 'idle',
  startedAt: null,
  controlPort,
};

function sendEvent(event) {
  if (mainWindow && !mainWindow.isDestroyed()) {
    mainWindow.webContents.send('backend:event', event);
  }
}

function sendStatus() {
  sendEvent({ type: 'status', state: { ...backendState } });
}

function emitLog(text, stream = 'stdout') {
  const lines = String(text).split(/\r?\n/).filter(Boolean);
  for (const line of lines) {
    sendEvent({ type: 'log', stream, message: line, time: new Date().toISOString() });
  }
}

function resolvePython() {
  for (const candidate of pythonCandidates) {
    if (candidate && fs.existsSync(candidate)) {
      return candidate;
    }
  }
  return 'python3';
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1480,
    height: 920,
    minWidth: 1240,
    minHeight: 820,
    backgroundColor: '#050816',
    title: 'AirPointer',
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false,
    },
  });

  mainWindow.loadFile(path.join(__dirname, 'index.html'));
  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

function startBackend(model) {
  if (backendProcess) {
    return { ok: false, error: 'Backend already running' };
  }

  const python = resolvePython();
  const script = path.join(projectRoot, 'main_advanced.py');
  backendState.running = true;
  backendState.model = model;
  backendState.status = 'starting';
  backendState.startedAt = Date.now();
  backendState.pid = null;
  sendStatus();

  backendProcess = spawn(python, [script, '--model', model], {
    cwd: projectRoot,
    env: {
      ...process.env,
      PYTHONUNBUFFERED: '1',
      AIRPOINTER_EMBEDDED_UI: '1',
      AIRPOINTER_PREVIEW_PORT: String(previewPort),
      AIRPOINTER_CONTROL_PORT: String(controlPort),
    },
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  backendState.pid = backendProcess.pid;
  backendState.status = 'running';
  sendStatus();
  emitLog(`Launching AirPointer backend with model=${model} using ${python}`, 'stdout');

  backendProcess.stdout.on('data', (chunk) => {
    stdoutBuffer += chunk.toString();
    const parts = stdoutBuffer.split(/\r?\n/);
    stdoutBuffer = parts.pop() || '';
    for (const line of parts) {
      if (line.trim()) {
        emitLog(line, 'stdout');
      }
    }
  });

  backendProcess.stderr.on('data', (chunk) => {
    stderrBuffer += chunk.toString();
    const parts = stderrBuffer.split(/\r?\n/);
    stderrBuffer = parts.pop() || '';
    for (const line of parts) {
      if (line.trim()) {
        emitLog(line, 'stderr');
      }
    }
  });

  backendProcess.on('exit', (code, signal) => {
    if (stdoutBuffer.trim()) {
      emitLog(stdoutBuffer.trim(), 'stdout');
    }
    if (stderrBuffer.trim()) {
      emitLog(stderrBuffer.trim(), 'stderr');
    }
    stdoutBuffer = '';
    stderrBuffer = '';

    emitLog(`Backend exited with code=${code} signal=${signal || 'none'}`, 'stderr');
    backendProcess = null;
    backendState.running = false;
    backendState.pid = null;
    backendState.status = code === 0 ? 'stopped' : 'error';
    backendState.controlPort = controlPort;
    sendStatus();
  });

  backendProcess.on('error', (error) => {
    emitLog(`Failed to start backend: ${error.message}`, 'stderr');
    backendProcess = null;
    backendState.running = false;
    backendState.pid = null;
    backendState.status = 'error';
    backendState.controlPort = controlPort;
    sendStatus();
  });

  return { ok: true, state: { ...backendState } };
}

function stopBackend() {
  if (!backendProcess) {
    backendState.running = false;
    backendState.pid = null;
    backendState.status = 'stopped';
    backendState.controlPort = controlPort;
    sendStatus();
    return { ok: true, message: 'Backend was not running' };
  }

  emitLog('Stopping AirPointer backend...', 'stdout');
  backendState.status = 'stopping';
  backendState.controlPort = controlPort;
  sendStatus();
  backendProcess.kill('SIGINT');

  setTimeout(() => {
    if (backendProcess) {
      backendProcess.kill('SIGKILL');
    }
  }, 2500);

  return { ok: true };
}

ipcMain.handle('backend:start', (_event, payload) => startBackend(payload?.model || 'custom1'));
ipcMain.handle('backend:stop', () => stopBackend());
ipcMain.handle('backend:status', () => ({ ...backendState }));
ipcMain.handle('app:quit', () => {
  app.quit();
  return { ok: true };
});

app.whenReady().then(() => {
  createWindow();
  sendStatus();
  setTimeout(() => {
    if (!backendProcess) {
      startBackend(backendState.model);
    }
  }, 500);
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  if (backendProcess) {
    backendProcess.kill('SIGINT');
  }
});
