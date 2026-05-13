/* ============================================================
   VPR System Frontend — app.js
   ============================================================ */

// ─── State ───────────────────────────────────────────────────
// ⚠️ Sửa URL này thành URL thật của Render service của bạn!
// VD: https://vpr-api-gateway.onrender.com
const RENDER_URL = 'https://vpr-ot-crossimage.onrender.com';

let currentServer = 'render';
let apiBase = RENDER_URL;

// ─── Init ────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    updateUrlBadge();
    checkHealth();
    setInterval(checkHealth, 30000); // poll health mỗi 30s
    setupDragDrop();
});

// ─── Server Switching ────────────────────────────────────────
function switchServer(type) {
    currentServer = type;
    document.getElementById('btn-render').classList.toggle('active', type === 'render');
    document.getElementById('btn-k8s').classList.toggle('active', type === 'k8s');
    document.getElementById('k8s-url-wrap').style.display = type === 'k8s' ? 'flex' : 'none';

    if (type === 'render') {
        apiBase = RENDER_URL;
        updateUrlBadge();
        checkHealth();
    } else {
        apiBase = document.getElementById('k8s-url').value.trim() || 'http://127.0.0.1:8080';
        updateUrlBadge();
        checkHealth();
    }
}

function applyK8sUrl() {
    apiBase = document.getElementById('k8s-url').value.trim() || 'http://127.0.0.1:8080';
    updateUrlBadge();
    checkHealth();
    showToast('✅ Đã áp dụng URL Kubernetes', 'success');
}

function updateUrlBadge() {
    document.getElementById('current-url-badge').textContent = apiBase;
}

function getApiKey() {
    return document.getElementById('api-key-input').value.trim();
}

// ─── Health Check ─────────────────────────────────────────────
async function checkHealth() {
    try {
        const resp = await fetch(`${apiBase}/api/v1/health`, { signal: AbortSignal.timeout(8000) });
        const data = await resp.json();

        setDot('triton-status', data.triton_ready === true ? 'ok' : 'error');
        setDot('gallery-status', data.gallery_ready === true ? 'ok' : 'error');
        const gsize = data.gallery_size ?? 0;
        document.getElementById('gallery-label').textContent = `Gallery (${gsize})`;
    } catch {
        setDot('triton-status', 'error');
        setDot('gallery-status', 'error');
    }
}

function setDot(id, state) {
    const el = document.getElementById(id);
    const dot = el.querySelector('.dot');
    dot.className = `dot dot-${state}`;
}

// ─── Tab Navigation ──────────────────────────────────────────
function showTab(tab) {
    document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById(`tab-${tab}`).classList.add('active');
    event.currentTarget.classList.add('active');
}

// ─── Image Preview ────────────────────────────────────────────
function previewImage(prefix) {
    const file = document.getElementById(`${prefix}-file`).files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = e => {
        const preview = document.getElementById(`${prefix}-preview`);
        preview.src = e.target.result;
        preview.style.display = 'block';
        const icons = document.querySelectorAll(`#${prefix}-dropzone .upload-icon, #${prefix}-dropzone .upload-text, #${prefix}-dropzone .upload-hint`);
        icons.forEach(el => el.style.opacity = '0');
    };
    reader.readAsDataURL(file);
}

// ─── Drag & Drop ──────────────────────────────────────────────
function setupDragDrop() {
    ['retrieve', 'index'].forEach(prefix => {
        const zone = document.getElementById(`${prefix}-dropzone`);
        zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
        zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
        zone.addEventListener('drop', e => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            const file = e.dataTransfer.files[0];
            if (!file || !file.type.startsWith('image/')) return;
            const input = document.getElementById(`${prefix}-file`);
            const dt = new DataTransfer();
            dt.items.add(file);
            input.files = dt.files;
            previewImage(prefix);
        });
    });
}

// ─── RETRIEVE ─────────────────────────────────────────────────
async function doRetrieve() {
    const fileInput = document.getElementById('retrieve-file');
    if (!fileInput.files[0]) { showToast('⚠️ Vui lòng chọn ảnh truy vấn', 'error'); return; }

    const topK = document.getElementById('top-k').value;
    showLoading('🔍 Đang tìm kiếm địa điểm...');

    const form = new FormData();
    form.append('image', fileInput.files[0]);
    form.append('top_k', topK);

    try {
        const resp = await fetch(`${apiBase}/api/v1/retrieve`, {
            method: 'POST',
            headers: { 'X-API-Key': getApiKey() },
            body: form,
            signal: AbortSignal.timeout(60000),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        renderResults(data);
        document.getElementById('retrieve-empty').style.display = 'none';
        document.getElementById('retrieve-results').style.display = 'block';
        showToast(`✅ Tìm thấy ${data.matches.length} kết quả`, 'success');

    } catch (e) {
        showToast(`❌ Lỗi: ${e.message}`, 'error');
    } finally {
        hideLoading();
    }
}

function renderResults(data) {
    document.getElementById('results-meta').textContent =
        `Mô tả: ${data.query_descriptor_dim}D | Gallery: ${data.gallery_size} ảnh | Trả về: ${data.top_k} kết quả`;

    const grid = document.getElementById('results-grid');
    grid.innerHTML = '';

    data.matches.forEach(m => {
        const imageUrl = m.metadata?.image_url || '';
        const placeName = m.metadata?.place_name || m.image_path || `Gallery #${m.gallery_id}`;
        const lat = m.metadata?.lat;
        const lon = m.metadata?.lon;
        const gpsText = (lat && lon) ? `📍 ${parseFloat(lat).toFixed(4)}, ${parseFloat(lon).toFixed(4)}` : '';
        const scorePercent = Math.round(m.score * 100);

        const card = document.createElement('div');
        card.className = 'result-card';
        card.innerHTML = `
            <div class="result-img ${imageUrl ? '' : 'no-img'}">
                ${imageUrl
                ? `<img src="${imageUrl}" alt="${placeName}" loading="lazy" onerror="this.parentElement.classList.add('no-img');this.remove();this.parentElement.innerHTML='🖼️'">`
                : '🖼️'
            }
            </div>
            <div class="result-body">
                <div class="result-rank"># Hạng ${m.rank}</div>
                <div class="result-name" title="${placeName}">${placeName}</div>
                <div class="result-score">Tương đồng: <strong>${scorePercent}%</strong> &nbsp;|&nbsp; Khoảng cách: ${m.distance.toFixed(3)}</div>
                <div class="score-bar-wrap"><div class="score-bar" style="width:${scorePercent}%"></div></div>
                ${gpsText ? `<div class="result-gps">${gpsText}</div>` : ''}
            </div>
        `;
        grid.appendChild(card);
    });
}

// ─── INDEX ────────────────────────────────────────────────────
async function doIndex() {
    const fileInput = document.getElementById('index-file');
    if (!fileInput.files[0]) { showToast('⚠️ Vui lòng chọn ảnh để index', 'error'); return; }

    showLoading('📸 Đang nhúng ảnh và lưu vào Gallery...');

    const form = new FormData();
    form.append('image', fileInput.files[0]);
    form.append('place_name', document.getElementById('place-name').value);
    const lat = document.getElementById('lat').value;
    const lon = document.getElementById('lon').value;
    if (lat) form.append('lat', lat);
    if (lon) form.append('lon', lon);

    try {
        const resp = await fetch(`${apiBase}/api/v1/index`, {
            method: 'POST',
            headers: { 'X-API-Key': getApiKey() },
            body: form,
            signal: AbortSignal.timeout(120000),
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `HTTP ${resp.status}`);
        }

        const data = await resp.json();
        const resultCard = document.getElementById('index-result-card');
        const resultMsg = document.getElementById('index-result-msg');
        const imgHtml = data.image_url ? `<img src="${data.image_url}" style="height:60px;border-radius:8px;margin-left:auto">` : '';
        resultMsg.innerHTML = `✅ ${data.message}<br><small>Gallery size: ${data.gallery_size} ảnh</small>${imgHtml}`;
        resultCard.style.display = 'block';
        showToast(`✅ Đã thêm vào gallery #${data.gallery_id}`, 'success');
        checkHealth(); // refresh gallery size

    } catch (e) {
        showToast(`❌ Lỗi: ${e.message}`, 'error');
    } finally {
        hideLoading();
    }
}

// ─── UI Helpers ───────────────────────────────────────────────
function showLoading(text = 'Đang xử lý...') {
    document.getElementById('loading-text').textContent = text;
    document.getElementById('loading-overlay').classList.add('show');
}
function hideLoading() {
    document.getElementById('loading-overlay').classList.remove('show');
}

let toastTimer;
function showToast(msg, type = '') {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.className = `toast ${type} show`;
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove('show'), 3500);
}
