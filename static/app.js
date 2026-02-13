const startBtn = document.getElementById('startBtn');
const stopBtn = document.getElementById('stopBtn');
const videoSelect = document.getElementById('videoSelect');
const modeSelect = document.getElementById('modeSelect');
const streamImg = document.getElementById('stream');
const statusEl = document.getElementById('status');
const downloadBtn = document.getElementById('downloadBtn');
const manageBtn = document.getElementById('manageBtn');
const platesPanel = document.getElementById('platesPanel');
const platesStatus = document.getElementById('platesStatus');
const plateInput = document.getElementById('plateInput');
const addPlateBtn = document.getElementById('addPlateBtn');
const platesTableBody = document.getElementById('platesTableBody');
let sessionId = null;
let statusTimer = null;

function setStatus(text) {
  statusEl.textContent = text;
}

function setPlatesStatus(text) {
  if (platesStatus) {
    platesStatus.textContent = text;
  }
}

function setControls(running) {
  startBtn.disabled = running;
  stopBtn.disabled = !running;
}

function resetDownload() {
  downloadBtn.classList.add('disabled');
  downloadBtn.setAttribute('aria-disabled', 'true');
  downloadBtn.href = '#';
}

function enableDownload(url) {
  downloadBtn.classList.remove('disabled');
  downloadBtn.setAttribute('aria-disabled', 'false');
  downloadBtn.href = url;
}

function renderPlates(plates) {
  platesTableBody.innerHTML = '';
  plates.forEach(plate => {
    const row = document.createElement('tr');
    row.dataset.id = plate.id;

    const plateCell = document.createElement('td');
    const input = document.createElement('input');
    input.type = 'text';
    input.value = plate.text || '';
    input.className = 'plate-edit-input';
    plateCell.appendChild(input);

    const actionsCell = document.createElement('td');
    const actions = document.createElement('div');
    actions.className = 'plates-actions';

    const updateBtn = document.createElement('button');
    updateBtn.type = 'button';
    updateBtn.textContent = 'Update';
    updateBtn.className = 'update-btn';

    const deleteBtn = document.createElement('button');
    deleteBtn.type = 'button';
    deleteBtn.textContent = 'Delete';
    deleteBtn.className = 'delete-btn secondary';

    actions.appendChild(updateBtn);
    actions.appendChild(deleteBtn);
    actionsCell.appendChild(actions);

    row.appendChild(plateCell);
    row.appendChild(actionsCell);
    platesTableBody.appendChild(row);
  });
}

function loadPlates() {
  setPlatesStatus('Loading...');
  fetch('/plates')
    .then(res => res.json())
    .then(data => {
      renderPlates(data.plates || []);
      setPlatesStatus('Ready');
    })
    .catch(() => setPlatesStatus('Failed to load plates.'));
}

function addPlate() {
  const text = plateInput.value.trim();
  if (!text) {
    setPlatesStatus('Enter a plate.');
    return;
  }

  fetch('/plates', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  })
    .then(res => res.json())
    .then(() => {
      plateInput.value = '';
      loadPlates();
      setPlatesStatus('Added.');
    })
    .catch(() => setPlatesStatus('Failed to add plate.'));
}

function updatePlate(plateId, text) {
  const trimmed = text.trim();
  if (!trimmed) {
    setPlatesStatus('Plate text required.');
    return;
  }

  fetch(`/plates/${plateId}`,
    {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: trimmed })
    })
    .then(res => res.json())
    .then(() => {
      loadPlates();
      setPlatesStatus('Updated.');
    })
    .catch(() => setPlatesStatus('Failed to update plate.'));
}

function deletePlate(plateId) {
  fetch(`/plates/${plateId}`, { method: 'DELETE' })
    .then(res => res.json())
    .then(() => {
      loadPlates();
      setPlatesStatus('Deleted.');
    })
    .catch(() => setPlatesStatus('Failed to delete plate.'));
}

function startStatusPolling() {
  if (statusTimer) {
    clearInterval(statusTimer);
  }

  statusTimer = setInterval(() => {
    if (!sessionId) {
      return;
    }
    fetch(`/status/${sessionId}`)
      .then(res => res.json())
      .then(data => {
        if (data.completed && data.download_url) {
          enableDownload(data.download_url);
          setStatus('Completed');
          clearInterval(statusTimer);
          statusTimer = null;
        } else if (data.finished) {
          setStatus('Stopped');
          clearInterval(statusTimer);
          statusTimer = null;
        }
      })
      .catch(() => {
        // Keep polling; transient errors should not break the UI.
      });
  }, 1000);
}

startBtn.addEventListener('click', () => {
  const video = videoSelect.value;
  const mode = modeSelect.value;
  if (!video || !mode) {
    setStatus('Select a video and mode.');
    return;
  }

  // Prevent double-start while request is in flight.
  startBtn.disabled = true;
  stopBtn.disabled = true;
  resetDownload();
  setStatus('Starting...');

  fetch('/start', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ video, mode })
  })
    .then(res => res.json())
    .then(data => {
      sessionId = data.session_id;
      const url = `${data.stream_url}&t=${Date.now()}`;
      streamImg.src = url;
      setControls(true);
      setStatus(`Streaming ${video} (${mode})`);
      startStatusPolling();
    })
    .catch(() => {
      setControls(false);
      setStatus('Failed to start stream.');
    });
});

stopBtn.addEventListener('click', () => {
  if (!sessionId) {
    return;
  }

  // Prevent double-stop while request is in flight.
  stopBtn.disabled = true;
  fetch('/stop', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: sessionId })
  })
    .then(res => res.json())
    .then(data => {
      streamImg.src = '';
      setControls(false);
      setStatus('Stopped');
      resetDownload();
      if (data.download_url) {
        const wantDownload = window.confirm('Download the partially processed video?');
        if (wantDownload) {
          enableDownload(data.download_url);
          window.location.href = data.download_url;
        }
      }
      if (statusTimer) {
        clearInterval(statusTimer);
        statusTimer = null;
      }
      sessionId = null;
    })
    .catch(() => {
      setControls(true);
      setStatus('Failed to stop stream.');
    });
});

if (manageBtn && platesPanel) {
  manageBtn.addEventListener('click', () => {
    platesPanel.classList.toggle('hidden');
    if (!platesPanel.classList.contains('hidden')) {
      loadPlates();
    }
  });
}

if (addPlateBtn) {
  addPlateBtn.addEventListener('click', addPlate);
}

if (platesTableBody) {
  platesTableBody.addEventListener('click', event => {
    const target = event.target;
    if (!(target instanceof HTMLElement)) {
      return;
    }

    const row = target.closest('tr');
    if (!row) {
      return;
    }

    const plateId = row.dataset.id;
    const input = row.querySelector('input');
    const text = input ? input.value : '';

    if (target.classList.contains('update-btn')) {
      updatePlate(plateId, text);
    }

    if (target.classList.contains('delete-btn')) {
      const ok = window.confirm('Delete this plate?');
      if (ok) {
        deletePlate(plateId);
      }
    }
  });
}
