// NeuroScribe — Main Client Application

let currentTab = 'canvas-tab';
let canvasMode = 'mnist'; // 'mnist' | 'emnist' | 'auto'
let scannerPipeline = 'segmentation'; // 'segmentation' | 'crnn'
let autoPredictTimeout = null;
let chartInstance = null;

// Canvas state
const canvas = document.getElementById('paint-canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
let isDrawing = false;
let lastX = 0;
let lastY = 0;
let strokeHistory = [];
let hasDrawn = false;

// ----------------------------------------------------
// Initialization
// ----------------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
  initCanvas();
  initChart();
  loadSampleGallery();
  loadDatasetSamples();
  setupDragAndDrop();
  
  // Set brush size listener
  const brushSlider = document.getElementById('brush-size');
  if (brushSlider) {
    brushSlider.addEventListener('input', (e) => {
      ctx.lineWidth = parseInt(e.target.value, 10);
    });
  }
});

// ----------------------------------------------------
// Tab Switching
// ----------------------------------------------------
function switchTab(tabId) {
  currentTab = tabId;
  document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  
  const targetContent = document.getElementById(tabId);
  const targetBtn = document.getElementById(`tab-btn-${tabId}`);
  if (targetContent) targetContent.classList.remove('hidden');
  if (targetBtn) targetBtn.classList.add('active');
  
  if (tabId === 'canvas-tab') {
    // Resize chart if needed
    if (chartInstance) chartInstance.resize();
  }
}

// ----------------------------------------------------
// Canvas Core & Drawing
// ----------------------------------------------------
function initCanvas() {
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = parseInt(document.getElementById('brush-size').value, 10);
  
  saveCanvasState();
  
  // Mouse / Touch / Pointer events
  canvas.addEventListener('pointerdown', startDrawing);
  canvas.addEventListener('pointermove', draw);
  canvas.addEventListener('pointerup', stopDrawing);
  canvas.addEventListener('pointerout', stopDrawing);
}

function saveCanvasState() {
  strokeHistory.push(ctx.getImageData(0, 0, canvas.width, canvas.height));
  if (strokeHistory.length > 25) {
    strokeHistory.shift();
  }
}

function getPointerPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY
  };
}

function startDrawing(e) {
  isDrawing = true;
  const pos = getPointerPos(e);
  lastX = pos.x;
  lastY = pos.y;
  
  ctx.beginPath();
  ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
  ctx.fillStyle = '#ffffff';
  ctx.fill();
  
  const hint = document.getElementById('canvas-empty-hint');
  if (hint) hint.classList.add('hidden');
  hasDrawn = true;
}

function draw(e) {
  if (!isDrawing) return;
  const pos = getPointerPos(e);
  
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();
  
  lastX = pos.x;
  lastY = pos.y;
  
  triggerDebouncedPredict();
}

function stopDrawing() {
  if (!isDrawing) return;
  isDrawing = false;
  saveCanvasState();
  triggerDebouncedPredict(150);
}

function clearCanvas() {
  ctx.fillStyle = '#000000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  strokeHistory = [];
  saveCanvasState();
  hasDrawn = false;
  
  const hint = document.getElementById('canvas-empty-hint');
  if (hint) hint.classList.remove('hidden');
  
  resetCanvasResults();
}

function undoCanvas() {
  if (strokeHistory.length > 1) {
    strokeHistory.pop(); // Remove current
    const prevState = strokeHistory[strokeHistory.length - 1];
    ctx.putImageData(prevState, 0, 0);
    triggerDebouncedPredict(100);
  } else if (strokeHistory.length === 1) {
    clearCanvas();
  }
}

function resetCanvasResults() {
  document.getElementById('top-pred-char').innerText = '?';
  document.getElementById('top-pred-class-desc').innerText = 'Draw or select a preset';
  document.getElementById('top-pred-conf').innerText = '0.0%';
  document.getElementById('inference-time').innerText = '0 ms';
  document.getElementById('preprocessed-preview').src = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABwAAAAcCAQAAACoKsFlAAAAD0lEQVR42mNkGAWjYBSMAgACmgABsO2e3gAAAABJRU5ErkJggg==';
  
  if (chartInstance) {
    chartInstance.data.labels = ['-', '-', '-', '-', '-'];
    chartInstance.data.datasets[0].data = [0, 0, 0, 0, 0];
    chartInstance.update();
  }
}

function setCanvasMode(mode) {
  canvasMode = mode;
  ['mnist', 'emnist', 'auto'].forEach(m => {
    const btn = document.getElementById(`mode-${m}`);
    if (m === mode) {
      btn.className = "px-3 py-1.5 rounded-lg font-medium text-xs sm:text-sm transition-all bg-indigo-600 text-white shadow";
    } else {
      btn.className = "px-3 py-1.5 rounded-lg font-medium text-xs sm:text-sm text-slate-400 hover:text-slate-200 transition-all";
    }
  });
  
  if (hasDrawn) {
    predictCanvas();
  }
}

function triggerDebouncedPredict(delay = 320) {
  const autoPredict = document.getElementById('auto-predict-toggle').checked;
  if (!autoPredict) return;
  
  clearTimeout(autoPredictTimeout);
  autoPredictTimeout = setTimeout(() => {
    predictCanvas();
  }, delay);
}

// ----------------------------------------------------
// Canvas Prediction API Call
// ----------------------------------------------------
async function predictCanvas() {
  if (!hasDrawn) return;
  
  const dataURL = canvas.toDataURL('image/png');
  const t0 = performance.now();
  
  try {
    const res = await fetch('/api/predict_canvas', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: dataURL,
        mode: canvasMode
      })
    });
    
    const data = await res.json();
    const elapsed = Math.round(performance.now() - t0);
    
    if (data.status === 'ok') {
      const top = data.top_prediction;
      document.getElementById('top-pred-char').innerText = top.char;
      document.getElementById('top-pred-class-desc').innerText = top.description;
      document.getElementById('top-pred-conf').innerText = `${(top.probability * 100).toFixed(1)}%`;
      document.getElementById('inference-time').innerText = `${elapsed} ms`;
      
      if (data.preprocessed_preview) {
        document.getElementById('preprocessed-preview').src = data.preprocessed_preview;
      }
      
      updateChart(data.top_5);
    }
  } catch (err) {
    console.error('Error predicting canvas:', err);
  }
}

// ----------------------------------------------------
// Chart.js Probability Visualizer
// ----------------------------------------------------
function initChart() {
  const chartCanvas = document.getElementById('confidence-chart');
  if (!chartCanvas) return;
  
  chartInstance = new Chart(chartCanvas.getContext('2d'), {
    type: 'bar',
    data: {
      labels: ['-', '-', '-', '-', '-'],
      datasets: [{
        label: 'Confidence (%)',
        data: [0, 0, 0, 0, 0],
        backgroundColor: [
          'rgba(99, 102, 241, 0.9)', // Indigo primary
          'rgba(129, 140, 248, 0.5)',
          'rgba(129, 140, 248, 0.4)',
          'rgba(129, 140, 248, 0.3)',
          'rgba(129, 140, 248, 0.2)'
        ],
        borderColor: [
          '#6366f1',
          '#818cf8',
          '#818cf8',
          '#818cf8',
          '#818cf8'
        ],
        borderWidth: 1,
        borderRadius: 6
      }]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          beginAtZero: true,
          max: 100,
          grid: { color: 'rgba(51, 65, 85, 0.3)' },
          ticks: { color: '#94a3b8', font: { family: 'JetBrains Mono', size: 10 } }
        },
        y: {
          grid: { display: false },
          ticks: { color: '#f1f5f9', font: { family: 'JetBrains Mono', weight: 'bold', size: 13 } }
        }
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => `Confidence: ${ctx.parsed.x.toFixed(2)}%`
          }
        }
      }
    }
  });
}

function updateChart(top5) {
  if (!chartInstance || !top5) return;
  chartInstance.data.labels = top5.map(item => `'${item.char}'`);
  chartInstance.data.datasets[0].data = top5.map(item => (item.probability * 100).toFixed(1));
  chartInstance.update();
}

// ----------------------------------------------------
// Canvas Presets Loader
// ----------------------------------------------------
async function loadPreset(char) {
  try {
    const res = await fetch(`/api/get_preset?char=${char}`);
    const data = await res.json();
    if (data.status === 'ok' && data.image) {
      const img = new Image();
      img.onload = () => {
        clearCanvas();
        // Scale and center onto paint-canvas
        const size = 260;
        const x = (canvas.width - size) / 2;
        const y = (canvas.height - size) / 2;
        ctx.drawImage(img, x, y, size, size);
        saveCanvasState();
        hasDrawn = true;
        const hint = document.getElementById('canvas-empty-hint');
        if (hint) hint.classList.add('hidden');
        predictCanvas();
      };
      img.src = data.image;
    }
  } catch (e) {
    console.error('Error loading preset:', e);
  }
}

// ----------------------------------------------------
// TAB 2: Document & Word Scanner
// ----------------------------------------------------
function setScannerPipeline(pipe) {
  scannerPipeline = pipe;
  const btnSeg = document.getElementById('pipeline-segmentation');
  const btnCrnn = document.getElementById('pipeline-crnn');
  
  if (pipe === 'segmentation') {
    btnSeg.className = "px-3.5 py-1.5 rounded-lg font-medium text-xs sm:text-sm transition-all bg-indigo-600 text-white shadow";
    btnCrnn.className = "px-3.5 py-1.5 rounded-lg font-medium text-xs sm:text-sm text-slate-400 hover:text-slate-200 transition-all";
  } else {
    btnCrnn.className = "px-3.5 py-1.5 rounded-lg font-medium text-xs sm:text-sm transition-all bg-indigo-600 text-white shadow";
    btnSeg.className = "px-3.5 py-1.5 rounded-lg font-medium text-xs sm:text-sm text-slate-400 hover:text-slate-200 transition-all";
  }
  
  // Re-run scan if image is loaded
  const origImg = document.getElementById('scanner-img-original');
  if (origImg && origImg.src && !origImg.classList.contains('hidden')) {
    scanUploadedImage(origImg.src);
  }
}

function setupDragAndDrop() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-upload-input');
  
  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });
  
  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });
  
  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });
  
  fileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });
}

function handleFile(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    scanUploadedImage(e.target.result);
  };
  reader.readAsDataURL(file);
}

async function scanUploadedImage(base64DataUrl) {
  const dropEmpty = document.getElementById('drop-zone-empty');
  const imgAnnotated = document.getElementById('scanner-img-annotated');
  const imgBinary = document.getElementById('scanner-img-binary');
  const imgOriginal = document.getElementById('scanner-img-original');
  
  dropEmpty.classList.add('hidden');
  imgOriginal.src = base64DataUrl;
  
  const textDisplay = document.getElementById('recognized-text-display');
  textDisplay.innerHTML = '<span class="text-indigo-400 text-lg flex items-center gap-2"><i class="fa-solid fa-spinner fa-spin"></i> Processing and recognizing characters...</span>';
  
  const t0 = performance.now();
  
  try {
    const endpoint = scannerPipeline === 'crnn' ? '/api/predict_word_crnn' : '/api/predict_image';
    const res = await fetch(endpoint, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64DataUrl })
    });
    
    const data = await res.json();
    const elapsed = Math.round(performance.now() - t0);
    
    if (data.status === 'ok') {
      imgAnnotated.src = data.annotated_image;
      imgBinary.src = data.binary_image;
      
      toggleScannerView('annotated');
      
      textDisplay.innerText = data.recognized_text || "(No characters recognized)";
      document.getElementById('overall-confidence-badge').innerText = `${(data.average_confidence * 100).toFixed(1)}%`;
      document.getElementById('scanner-char-count').innerText = `${data.character_count} character(s) recognized`;
      document.getElementById('scanner-time').innerText = `Inference: ${elapsed} ms`;
      
      renderCharacterBreakdown(data.characters || []);
    } else {
      textDisplay.innerText = `Error: ${data.message || 'Recognition failed'}`;
    }
  } catch (err) {
    console.error('Scan error:', err);
    textDisplay.innerText = 'Network or processing error occurred';
  }
}

function toggleScannerView(view) {
  const imgAnnotated = document.getElementById('scanner-img-annotated');
  const imgBinary = document.getElementById('scanner-img-binary');
  const imgOriginal = document.getElementById('scanner-img-original');
  
  ['annotated', 'binary', 'original'].forEach(v => {
    const btn = document.getElementById(`btn-view-${v}`);
    if (v === view) {
      btn.className = "px-2.5 py-1 rounded-md bg-indigo-600 text-white font-medium";
    } else {
      btn.className = "px-2.5 py-1 rounded-md bg-slate-800 text-slate-400 hover:text-white";
    }
  });
  
  imgAnnotated.classList.add('hidden');
  imgBinary.classList.add('hidden');
  imgOriginal.classList.add('hidden');
  
  if (view === 'annotated') imgAnnotated.classList.remove('hidden');
  if (view === 'binary') imgBinary.classList.remove('hidden');
  if (view === 'original') imgOriginal.classList.remove('hidden');
}

function renderCharacterBreakdown(chars) {
  const container = document.getElementById('segmented-chars-list');
  if (!chars || chars.length === 0) {
    container.innerHTML = '<div class="text-center text-slate-500 text-xs py-8">No characters detected</div>';
    return;
  }
  
  container.innerHTML = chars.map((c, idx) => `
    <div class="flex items-center justify-between p-2 rounded-xl bg-slate-950/80 border border-slate-800/80 hover:border-indigo-500/50 transition-all">
      <div class="flex items-center space-x-3">
        <span class="text-xs font-mono font-bold text-slate-400 w-6">#${idx + 1}</span>
        <div class="w-8 h-8 rounded bg-black border border-slate-700 flex items-center justify-center p-0.5">
          <img src="${c.thumbnail}" class="w-full h-full object-contain pixelated" alt="char">
        </div>
        <div>
          <span class="font-mono font-bold text-white text-base">${c.predicted_char}</span>
          <span class="text-[11px] text-slate-400 ml-2">Line ${c.line_idx + 1}</span>
        </div>
      </div>
      <div class="flex items-center space-x-2">
        <span class="text-xs font-mono text-indigo-300 font-semibold">${(c.confidence * 100).toFixed(1)}%</span>
      </div>
    </div>
  `).join('');
}

function copyRecognizedText() {
  const text = document.getElementById('recognized-text-display').innerText;
  if (!text || text.includes('will appear here')) return;
  
  navigator.clipboard.writeText(text).then(() => {
    const copyBtn = document.getElementById('copy-btn-text');
    copyBtn.innerText = 'Copied!';
    setTimeout(() => copyBtn.innerText = 'Copy', 2000);
  });
}

// ----------------------------------------------------
// Sample Gallery Loader
// ----------------------------------------------------
async function loadSampleGallery() {
  const container = document.getElementById('sample-gallery-container');
  try {
    const res = await fetch('/api/sample_words');
    const data = await res.json();
    if (data.samples) {
      container.innerHTML = data.samples.map(sample => `
        <button onclick="scanSampleImage('${sample.filename}')" class="p-2 rounded-xl bg-slate-950/70 border border-slate-800 hover:border-indigo-500 hover:bg-slate-900 transition-all text-left flex flex-col items-center group">
          <div class="w-full h-12 bg-black rounded-lg flex items-center justify-center overflow-hidden p-1 border border-slate-800">
            <img src="${sample.url}" alt="${sample.label}" class="max-h-full max-w-full object-contain filter invert opacity-90 group-hover:opacity-100">
          </div>
          <span class="text-[11px] font-mono text-slate-400 group-hover:text-indigo-300 mt-1.5 font-semibold">${sample.label}</span>
        </button>
      `).join('');
    }
  } catch (e) {
    console.error('Error loading sample gallery:', e);
  }
}

async function scanSampleImage(filename) {
  switchTab('scanner-tab');
  try {
    const res = await fetch(`/static/samples/${filename}`);
    const blob = await res.blob();
    const reader = new FileReader();
    reader.onload = (e) => scanUploadedImage(e.target.result);
    reader.readAsDataURL(blob);
  } catch (e) {
    console.error('Error loading sample image:', e);
  }
}

// ----------------------------------------------------
// TAB 3: Dataset Explorer
// ----------------------------------------------------
async function loadDatasetSamples() {
  const select = document.getElementById('dataset-filter-select');
  const source = select ? select.value : 'mnist';
  const grid = document.getElementById('dataset-samples-grid');
  grid.innerHTML = '<div class="col-span-full text-center text-slate-400 py-10"><i class="fa-solid fa-spinner fa-spin text-indigo-400 text-xl"></i> Loading dataset samples...</div>';
  
  try {
    const res = await fetch(`/api/dataset_samples?source=${source}`);
    const data = await res.json();
    
    if (data.samples) {
      grid.innerHTML = data.samples.map(s => `
        <div onclick="testDatasetSample('${s.image}', '${s.label}')" class="cursor-pointer p-3 rounded-xl bg-slate-900/60 border border-slate-800/80 hover:border-indigo-500 hover:scale-105 transition-all flex flex-col items-center group">
          <div class="w-16 h-16 rounded-lg bg-black border border-slate-700/80 p-1 flex items-center justify-center">
            <img src="${s.image}" class="w-full h-full object-contain pixelated" alt="${s.label}">
          </div>
          <div class="mt-2 flex items-center space-x-1.5">
            <span class="text-[10px] text-slate-400">Truth:</span>
            <span class="text-xs font-mono font-bold text-white">${s.label}</span>
          </div>
        </div>
      `).join('');
    }
  } catch (e) {
    grid.innerHTML = '<div class="col-span-full text-center text-red-400 text-xs py-10">Failed to load dataset samples</div>';
  }
}

function testDatasetSample(dataUrl, label) {
  // Load into canvas tab and run prediction
  switchTab('canvas-tab');
  const img = new Image();
  img.onload = () => {
    clearCanvas();
    const size = 260;
    const x = (canvas.width - size) / 2;
    const y = (canvas.height - size) / 2;
    ctx.drawImage(img, x, y, size, size);
    saveCanvasState();
    hasDrawn = true;
    const hint = document.getElementById('canvas-empty-hint');
    if (hint) hint.classList.add('hidden');
    predictCanvas();
  };
  img.src = dataUrl;
}

