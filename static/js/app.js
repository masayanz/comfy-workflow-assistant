const state = { models: [], family: 'all', selected: null, lastPayload: null };
const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const response = await fetch(path, { ...options, headers: { 'Content-Type': 'application/json', ...(options.headers || {}) } });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `通信に失敗しました (${response.status})`);
  return body;
}

function formatSize(size) {
  if (size == null) return 'ComfyUIから取得';
  if (size > 1024 ** 3) return `${(size / 1024 ** 3).toFixed(1)} GB`;
  return `${(size / 1024 ** 2).toFixed(0)} MB`;
}

function showMessage(text, kind = 'info') {
  const box = $('#message');
  box.textContent = text;
  box.className = `message ${kind}`;
}

function renderModels() {
  const query = $('#model-search').value.trim().toLowerCase();
  const items = state.models.filter((model) => model.type === 'checkpoint' &&
    (state.family === 'all' || model.family === state.family) && model.name.toLowerCase().includes(query));
  $('#model-count').textContent = `${items.length}モデル`;
  const list = $('#model-list');
  if (!items.length) {
    list.innerHTML = `<div class="empty-state">${state.models.length ? '条件に合うモデルがありません。' : 'Checkpointがありません。設定でComfyUIの場所を確認してください。'}</div>`;
    return;
  }
  list.innerHTML = items.map((model, index) => `<button class="model-card ${state.selected?.path === model.path ? 'selected' : ''}" data-index="${index}">
    <span class="model-icon">◈</span><span class="model-info"><strong title="${escapeHtml(model.name)}">${escapeHtml(model.name)}</strong><small>${formatSize(model.size)} <span class="family-pill ${model.family}">${familyName(model.family)}</span></small></span><span class="radio"></span></button>`).join('');
  list.querySelectorAll('.model-card').forEach((button) => button.addEventListener('click', () => {
    state.selected = items[Number(button.dataset.index)];
    renderModels(); renderSelected();
  }));
}

function familyName(family) { return ({ sdxl: 'SDXL', sd15: 'SD1.5', flux: 'Flux', unknown: '不明' })[family] || family; }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]); }

function renderSelected() {
  const holder = $('#selected-model');
  if (!state.selected) { holder.innerHTML = '<span class="model-placeholder">左からCheckpointを選択してください</span>'; $('#model-family-tag').textContent = '未選択'; return; }
  $('#model-family-tag').textContent = familyName(state.selected.family);
  if (state.selected.family === 'sd15') {
    $('#resolution').value = '512,512'; $('#steps').value = 25; $('#cfg').value = 7;
  } else if (state.selected.family === 'sdxl') {
    $('#resolution').value = '1024,1024'; $('#steps').value = 28; $('#cfg').value = 6;
  }
  $('#custom-size').classList.add('hidden');
  $('#node-flow').innerHTML = `${$('#lora').value ? '<span>Checkpoint Loader</span><i>↓</i><span>LoRA Loader</span><i>↓</i>' : '<span>Checkpoint Loader</span><i>↓</i>'}<span>Text Encode × 2</span><i>↓</i><span>Empty Latent Image</span><i>↓</i><span>KSampler</span><i>↓</i><span>VAE Decode</span><i>↓</i><span>Save Image</span>`;
  holder.innerHTML = `<span class="model-icon large">◈</span><div class="selected-model-info"><strong>${escapeHtml(state.selected.name)}</strong><small>${formatSize(state.selected.size)}</small></div><select class="family-select" id="family-select" aria-label="モデル系統">${[['sdxl','SDXL'],['sd15','SD1.5'],['flux','Flux'],['unknown','不明']].map(([value,label]) => `<option value="${value}" ${state.selected.family === value ? 'selected' : ''}>${label}</option>`).join('')}</select>`;
  $('#family-select').addEventListener('change', async (event) => {
    try {
      const updated = await api('/api/models/classify', { method: 'POST', body: JSON.stringify({ model: state.selected.comfy_name || state.selected.name, family: event.target.value }) });
      state.selected = updated; state.models = state.models.map((model) => model.path === updated.path ? updated : model); renderModels(); renderSelected();
      showMessage('モデル系統を保存しました。', 'success');
    } catch (error) { showMessage(error.message, 'error'); }
  });
}

async function refresh() {
  $('#scan-button').disabled = true;
  $('#scan-button').textContent = 'スキャン中…';
  try {
    const [status, models] = await Promise.all([api('/api/status'), api('/api/models')]);
    state.models = models;
    const lora = $('#lora');
    const selectedLora = lora.value;
    lora.innerHTML = '<option value="">使わない</option>' + models.filter((model) => model.type === 'lora').map((model) => `<option value="${escapeHtml(model.comfy_name || model.name)}">${escapeHtml(model.name)}</option>`).join('');
    if ([...lora.options].some((option) => option.value === selectedLora)) lora.value = selectedLora;
    $('#status-dot').classList.toggle('online', status.comfy.online);
    $('#status-label').textContent = status.comfy.online ? 'ComfyUI 接続中' : 'ComfyUI 未接続';
    $('#root-label').textContent = status.comfy_path ? status.comfy_path.split(/[\\/]/).slice(-2).join('/') : (status.model_source === 'comfy_api' ? 'ComfyUI API' : '環境未検出');
    if (status.settings) { $('#comfy-url').value = status.settings.comfy_url; $('#comfy-path').value = status.settings.comfy_path || ''; $('#web-port').value = status.settings.port || 7865; $('#open-browser').checked = status.settings.open_browser !== false; }
    if (state.selected) state.selected = state.models.find((model) => model.path === state.selected.path) || null;
    renderModels(); renderSelected();
  } catch (error) {
    showMessage(error.message, 'error');
  } finally {
    $('#scan-button').disabled = false;
    $('#scan-button').textContent = '↻ 再スキャン';
  }
}

function getPayload() {
  if (!state.selected) throw new Error('Checkpointを選択してください。');
  const prompt = $('#prompt').value.trim();
  if (!prompt) throw new Error('作りたいものを入力してください。');
  let [width, height] = $('#resolution').value.split(',').map(Number);
  if ($('#resolution').value === 'custom') { width = Number($('#width').value); height = Number($('#height').value); }
  const seed = $('#seed-mode').value === 'fixed' ? Number($('#seed').value) : Math.floor(Math.random() * 4294967296);
  const payload = { model: state.selected.comfy_name || state.selected.name, prompt, negative_prompt: $('#negative').value, width, height,
    steps: Number($('#steps').value), cfg: Number($('#cfg').value), sampler: $('#sampler').value, seed,
    lora: $('#lora').value || null, lora_weight: Number($('#lora-weight').value) };
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 64 || height < 64 || width > 4096 || height > 4096) throw new Error('解像度は64〜4096の整数で入力してください。');
  return payload;
}

async function run(payload = null) {
  try {
    const request = payload || getPayload();
    state.lastPayload = request;
    $('#run-button').disabled = true;
    $('#regenerate-button').disabled = true;
    showMessage('ComfyUIへWorkflowを送信しています…', 'info');
    const queued = await api('/api/workflow/run', { method: 'POST', body: JSON.stringify(request) });
    request.seed = queued.seed;
    state.lastPayload = request;
    showMessage(`Queue登録済み · Seed ${queued.seed}`, 'info');
    const deadline = Date.now() + 15 * 60 * 1000;
    let result;
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      result = await api(`/api/workflow/status/${encodeURIComponent(queued.prompt_id)}`);
      if (result.status === 'COMPLETED') break;
      if (result.status === 'ERROR') throw new Error(result.message || 'ComfyUIで画像生成に失敗しました。');
      showMessage(result.status === 'RUNNING' ? '画像を生成中です…' : 'Queue待機中です…', 'info');
    }
    if (!result || result.status !== 'COMPLETED') throw new Error('画像生成がタイムアウトしました。ComfyUIの状態を確認してください。');
    if (!result.images?.length) throw new Error('ComfyUIは処理を完了しましたが、画像が見つかりませんでした。SaveImageノードと出力を確認してください。');
    $('#result-images').innerHTML = result.images.map((image) => `<a href="${image.url}" target="_blank"><img src="${image.url}" alt="生成画像"></a>`).join('');
    $('#result-meta').textContent = `${state.selected?.name || request.model} · ${request.width} × ${request.height} · ${request.steps} steps · CFG ${request.cfg} · ${request.sampler} · Seed ${request.seed}`;
    $('#result').classList.remove('hidden');
    showMessage('画像生成が完了しました。', 'success');
  } catch (error) { showMessage(error.message, 'error'); }
  finally { $('#run-button').disabled = false; $('#regenerate-button').disabled = false; }
}

$('#build-button').addEventListener('click', async () => {
  try {
    const payload = getPayload();
    const result = await api('/api/workflow/build', { method: 'POST', body: JSON.stringify(payload) });
    $('#workflow-json').textContent = JSON.stringify(result.workflow, null, 2);
    $('#api-prompt-preview').classList.remove('hidden');
    $('#api-prompt-preview').open = true;
    if (result.ui_workflow) {
      $('#ui-workflow-json').textContent = JSON.stringify(result.ui_workflow, null, 2);
      $('#ui-workflow-preview').classList.remove('hidden');
    } else {
      $('#ui-workflow-preview').classList.add('hidden');
    }
    showMessage('WorkflowDefinitionからAPI PromptとComfyUI Workflowを生成しました。', 'success');
  } catch (error) { showMessage(error.message, 'error'); }
});
$('#copy-workflow').addEventListener('click', async () => {
  try { await navigator.clipboard.writeText($('#workflow-json').textContent); showMessage('Workflow JSONをコピーしました。', 'success'); }
  catch { showMessage('クリップボードへコピーできませんでした。JSON欄からコピーしてください。', 'error'); }
});

$('#filters').addEventListener('click', (event) => {
  const button = event.target.closest('[data-filter]'); if (!button) return;
  state.family = button.dataset.filter;
  $('#filters').querySelectorAll('.filter').forEach((item) => item.classList.toggle('active', item === button));
  renderModels();
});
$('#model-search').addEventListener('input', renderModels);
$('#scan-button').addEventListener('click', refresh);
$('#run-button').addEventListener('click', () => run());
$('#regenerate-button').addEventListener('click', () => run(state.lastPayload));
$('#resolution').addEventListener('change', () => $('#custom-size').classList.toggle('hidden', $('#resolution').value !== 'custom'));
$('#seed-mode').addEventListener('change', () => { $('#seed').disabled = $('#seed-mode').value !== 'fixed'; });
$('#lora').addEventListener('change', renderSelected);
$('#settings-open').addEventListener('click', () => $('#settings-dialog').showModal());
$('#settings-form').addEventListener('submit', async (event) => {
  if (event.submitter?.value !== 'save') return;
  event.preventDefault();
  try {
    await api('/api/settings', { method: 'PUT', body: JSON.stringify({ comfy_url: $('#comfy-url').value, comfy_path: $('#comfy-path').value, port: Number($('#web-port').value), open_browser: $('#open-browser').checked }) });
    $('#settings-dialog').close(); await refresh(); showMessage('設定を保存しました。', 'success');
  } catch (error) { showMessage(error.message, 'error'); }
});
async function saveWorkflow(endpoint, button, label) {
  button.disabled = true;
  try {
    const saved = await api(endpoint, { method: 'POST', body: JSON.stringify(getPayload()) });
    const link = document.createElement('a'); link.href = saved.download_url; link.download = saved.filename; link.click();
    showMessage(`${label}を保存しました: ${saved.filename}`, 'success');
  } catch (error) { showMessage(error.message, 'error'); }
  finally { button.disabled = false; }
}
$('#save-ui-button').addEventListener('click', (event) => saveWorkflow('/api/workflow/save-ui', event.currentTarget, 'ComfyUI Workflow'));
$('#save-api-button').addEventListener('click', (event) => saveWorkflow('/api/workflow/save', event.currentTarget, 'API Prompt'));
refresh();
