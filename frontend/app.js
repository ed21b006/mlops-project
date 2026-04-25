// Canvas drawing + API calls for digit prediction

const canvas = document.getElementById('draw-canvas');
const ctx = canvas.getContext('2d');
let isDrawing = false;
let lastX = 0, lastY = 0;
let lastPrediction = null;
let lastPixels = null;
let stats = { predictions: 0, feedback: 0, accuracy: null, latencies: [] };

// init canvas
ctx.fillStyle = '#000';
ctx.fillRect(0, 0, 280, 280);
ctx.strokeStyle = '#fff';
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.lineWidth = 16;

// mouse events
canvas.addEventListener('mousedown', e => { isDrawing = true; [lastX, lastY] = getPos(e); });
canvas.addEventListener('mousemove', e => { if (isDrawing) draw(e); });
canvas.addEventListener('mouseup', () => isDrawing = false);
canvas.addEventListener('mouseout', () => isDrawing = false);

// touch events
canvas.addEventListener('touchstart', e => { e.preventDefault(); isDrawing = true; [lastX, lastY] = getTouchPos(e); });
canvas.addEventListener('touchmove', e => { e.preventDefault(); if (isDrawing) drawTouch(e); });
canvas.addEventListener('touchend', () => isDrawing = false);

// brush size
const brushSlider = document.getElementById('brush-size');
brushSlider.addEventListener('input', e => {
    ctx.lineWidth = parseInt(e.target.value);
    document.getElementById('brush-value').textContent = e.target.value;
});

function getPos(e) {
    const rect = canvas.getBoundingClientRect();
    return [(e.clientX - rect.left) * (280 / rect.width),
            (e.clientY - rect.top) * (280 / rect.height)];
}

function getTouchPos(e) {
    const rect = canvas.getBoundingClientRect();
    const t = e.touches[0];
    return [(t.clientX - rect.left) * (280 / rect.width),
            (t.clientY - rect.top) * (280 / rect.height)];
}

function draw(e) {
    const [x, y] = getPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
}

function drawTouch(e) {
    const [x, y] = getTouchPos(e);
    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();
    [lastX, lastY] = [x, y];
}

function getPixelArray() {
    // downscale canvas to 28x28
    const tmp = document.createElement('canvas');
    tmp.width = 28; tmp.height = 28;
    const tmpCtx = tmp.getContext('2d');
    tmpCtx.drawImage(canvas, 0, 0, 28, 28);
    const data = tmpCtx.getImageData(0, 0, 28, 28).data;

    const pixels = [];
    for (let i = 0; i < data.length; i += 4) {
        pixels.push(parseFloat((data[i] / 255.0).toFixed(6)));
    }
    return pixels;
}

function isCanvasBlank() {
    const data = ctx.getImageData(0, 0, 280, 280).data;
    for (let i = 0; i < data.length; i += 4) {
        if (data[i] > 10) return false;
    }
    return true;
}

async function handlePredict() {
    if (isCanvasBlank()) { alert('Please draw a digit first!'); return; }

    const pixels = getPixelArray();
    lastPixels = pixels;
    document.getElementById('loading').classList.remove('hidden');

    try {
        const res = await fetch('/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ pixel_array: pixels }),
        });
        if (!res.ok) throw new Error(`Server error: ${res.status}`);

        const result = await res.json();
        lastPrediction = result;
        stats.predictions++;
        stats.latencies.push(result.inference_time_ms);
        showResult(result);
        updateStats();
    } catch (err) {
        alert('Prediction failed: ' + err.message);
    } finally {
        document.getElementById('loading').classList.add('hidden');
    }
}

function showResult(r) {
    document.getElementById('result-placeholder').classList.add('hidden');
    document.getElementById('result-content').classList.remove('hidden');
    document.getElementById('predicted-digit').textContent = r.predicted_digit;

    const badge = document.getElementById('conf-badge');
    badge.textContent = (r.confidence * 100).toFixed(1) + '%';
    badge.className = 'conf-badge' + (r.confidence < 0.5 ? ' very-low' : r.confidence < 0.8 ? ' low' : '');

    document.getElementById('latency-val').textContent = r.inference_time_ms.toFixed(1) + 'ms';

    // probability bars
    const container = document.getElementById('chart-bars');
    container.innerHTML = '';
    for (let d = 0; d <= 9; d++) {
        const pct = (r.probabilities[d] * 100).toFixed(1);
        const isTop = d === r.predicted_digit;
        container.innerHTML += `<div class="bar-row">
            <span class="bar-label">${d}</span>
            <div class="bar-track"><div class="bar-fill ${isTop ? 'top' : ''}" style="width:${Math.max(r.probabilities[d] * 100, 0.5)}%"></div></div>
            <span class="bar-pct">${pct}%</span>
        </div>`;
    }
}

function handleClear() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, 280, 280);
    document.getElementById('result-placeholder').classList.remove('hidden');
    document.getElementById('result-content').classList.add('hidden');
    document.getElementById('feedback-msg').classList.add('hidden');
    lastPrediction = null;
}

async function submitFeedback(label) {
    if (!lastPrediction) { alert('Make a prediction first'); return; }

    try {
        const res = await fetch('/submit_feedback', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                correct_label: label,
                predicted_label: lastPrediction.predicted_digit,
                pixel_array: lastPixels,
            }),
        });
        const data = await res.json();
        // Since API no longer returns global stats for security, we only record the prediction
        updateStats();

        const msg = document.getElementById('feedback-msg');
        msg.textContent = label === lastPrediction.predicted_digit
            ? '✓ Correct! Thanks.' : `Got it, correct digit was ${label}. Thanks!`;
        msg.classList.remove('hidden');
        setTimeout(() => msg.classList.add('hidden'), 3000);
    } catch (err) {
        alert('Feedback failed: ' + err.message);
    }
}

function updateStats() {
    document.getElementById('stat-predictions').textContent = stats.predictions;
    if (stats.latencies.length > 0) {
        const avg = stats.latencies.reduce((a, b) => a + b, 0) / stats.latencies.length;
        document.getElementById('stat-latency').textContent = avg.toFixed(1) + 'ms';
    }
}

// build feedback buttons (0-9)
const grid = document.getElementById('feedback-grid');
for (let d = 0; d <= 9; d++) {
    const btn = document.createElement('button');
    btn.className = 'fb-btn';
    btn.textContent = d;
    btn.onclick = () => submitFeedback(d);
    grid.appendChild(btn);
}

// health check
async function checkHealth() {
    try {
        const r = await fetch('/health', { signal: AbortSignal.timeout(5000) });
        document.getElementById('status-dot').className = r.ok ? 'dot online' : 'dot offline';
        document.getElementById('status-text').textContent = r.ok ? 'Online' : 'Error';
    } catch {
        document.getElementById('status-dot').className = 'dot offline';
        document.getElementById('status-text').textContent = 'Offline';
    }
}
checkHealth();
setInterval(checkHealth, 15000);

// keyboard shortcuts
document.addEventListener('keydown', e => {
    if (e.key === 'Enter') handlePredict();
    if (e.key === 'Escape') handleClear();
});
