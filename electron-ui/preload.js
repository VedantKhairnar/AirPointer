const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('airpointer', {
  startBackend: (model) => ipcRenderer.invoke('backend:start', { model }),
  stopBackend: () => ipcRenderer.invoke('backend:stop'),
  getStatus: () => ipcRenderer.invoke('backend:status'),
  quitApp: () => ipcRenderer.invoke('app:quit'),
  onEvent: (callback) => {
    const listener = (_event, payload) => callback(payload);
    ipcRenderer.on('backend:event', listener);
    return () => ipcRenderer.removeListener('backend:event', listener);
  },
});
