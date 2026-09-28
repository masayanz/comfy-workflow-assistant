const state = { models: [], profiles: new Map(), family: 'all', selected: null, activeProfile: null, profileRequest: 0, lastPayload: null, inputImage: null, isRunning: false };
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
  if (state.family === 'flux' && state.profiles.get('flux')?.enabled) {
    const list = $('#model-list');
    const selected = state.selected?.path === 'profile:flux';
    const label = 'Flux txt2img（分割モデル構成）';
    const visible = label.toLowerCase().includes(query);
    $('#model-count').textContent = visible ? 'Flux Profile' : '0モデル';
    list.innerHTML = visible ? `<button class="model-card ${selected ? 'selected' : ''}" data-flux-profile="true"><span class="model-icon">◈</span><span class="model-info"><strong>${label}</strong><small>UNET + Text Encoders + VAE <span class="family-pill flux">Flux</span></small></span><span class="radio"></span></button>` : '<div class="empty-state">検索条件に合うProfileがありません。</div>';
    list.querySelector('[data-flux-profile]')?.addEventListener('click', () => {
      state.selected = { name: label, family: 'flux', type: 'profile', path: 'profile:flux' };
      state.activeProfile = null;
      renderModels(); renderSelected(); loadModelProfile(state.selected);
    });
    return;
  }
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
    state.activeProfile = null;
    renderModels(); renderSelected(); loadModelProfile(state.selected);
  }));
}

function familyName(family) { return ({ sdxl: 'SDXL', sd15: 'SD1.5', flux: 'Flux', unknown: '不明' })[family] || family; }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]); }

function renderSelected() {
  const holder = $('#selected-model');
  if (!state.selected) {
    holder.innerHTML = '<span class="model-placeholder">左からCheckpointを選択してください</span>';
    $('#model-family-tag').textContent = '未選択';
    $('#model-components').classList.add('hidden');
    setGenerationEnabled(false);
    return;
  }
  $('#model-family-tag').textContent = familyName(state.selected.family);
  renderNodeFlow();
  setGenerationEnabled(false);
  if (state.selected.type === 'profile') {
    holder.innerHTML = `<span class="model-icon large">◈</span><div class="selected-model-info"><strong>${escapeHtml(state.selected.name)}</strong><small>ComfyUIにある個別モデルを下で指定してください</small></div>`;
    return;
  }
  const classificationOptions = [...state.profiles.values()].map((profile) => [profile.id, profile.name]);
  classificationOptions.push(['unknown', '不明']);
  holder.innerHTML = `<span class="model-icon large">◈</span><div class="selected-model-info"><strong>${escapeHtml(state.selected.name)}</strong><small>${formatSize(state.selected.size)}</small></div><select class="family-select" id="family-select" aria-label="モデル系統">${classificationOptions.map(([value,label]) => `<option value="${escapeHtml(value)}" ${state.selected.family === value ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('')}</select>`;
  $('#family-select').addEventListener('change', async (event) => {
    try {
      const updated = await api('/api/models/classify', { method: 'POST', body: JSON.stringify({ model: state.selected.comfy_name || state.selected.name, family: event.target.value }) });
      state.selected = updated; state.models = state.models.map((model) => model.path === updated.path ? updated : model); renderModels(); renderSelected();
      await loadModelProfile(updated);
      showMessage('モデル系統を保存しました。', 'success');
    } catch (error) { showMessage(error.message, 'error'); }
  });
}

function renderFamilyFilters() {
  const filters = $('#filters');
  const entries = [['all', 'すべて'], ...[...state.profiles.values()].map((profile) => [profile.id, familyName(profile.id)]), ['unknown', '不明']];
  filters.innerHTML = entries.map(([id, label]) => `<button class="filter ${state.family === id ? 'active' : ''}" data-filter="${escapeHtml(id)}">${escapeHtml(label)}</button>`).join('');
}

function setGenerationEnabled(enabled) {
  const imageReady = selectedGenerationType() !== 'img2img' || Boolean(state.inputImage?.upload_id);
  const buildReady = enabled && imageReady && !state.isRunning;
  const assetsReady = buildReady && selectedComponentsReady();
  $('#build-button').disabled = !buildReady;
  for (const selector of ['#save-ui-button', '#save-api-button', '#run-button']) $(selector).disabled = !assetsReady;
}

function selectedGenerationType() {
  return document.querySelector('input[name="generation_type"]:checked')?.value || 'txt2img';
}

function syncGenerationMode(profile) {
  const txt2img = $('input[name="generation_type"][value="txt2img"]');
  const img2img = $('input[name="generation_type"][value="img2img"]');
  txt2img.disabled = !profile.capabilities.txt2img;
  img2img.disabled = !profile.capabilities.img2img;
  if (img2img.disabled && img2img.checked) txt2img.checked = true;
  if (txt2img.disabled && !img2img.disabled) img2img.checked = true;
  updateGenerationModeUI();
}

function updateGenerationModeUI() {
  const imageMode = selectedGenerationType() === 'img2img';
  $('#generation-type-label').textContent = imageMode ? 'IMG2IMG' : 'TXT2IMG';
  $('#image-input-section').classList.toggle('hidden', !imageMode);
  $('#resolution-grid').classList.toggle('hidden', imageMode);
  renderNodeFlow();
  const profile = state.activeProfile;
  setGenerationEnabled(Boolean(profile?.enabled && profile.capabilities[selectedGenerationType()]));
}

function selectedComponentsReady() {
  const components = state.activeProfile?.ui?.model_components || [];
  return components.every((component) => !component.required || Boolean($(`#component-${component.key}`)?.value));
}

function renderNodeFlow() {
  if (state.selected?.family === 'flux') {
    $('#node-flow').innerHTML = '<span>UNET Loader + Dual CLIP Loader</span><i>↓</i><span>CLIP Text Encode</span><i>↓</i><span>Flux Guidance + Empty SD3 Latent</span><i>↓</i><span>KSampler</span><i>↓</i><span>VAE Decode</span><i>↓</i><span>Save Image</span>';
    return;
  }
  if (selectedGenerationType() === 'img2img') {
    $('#node-flow').innerHTML = `${$('#lora').value ? '<span>Checkpoint Loader → LoRA Loader</span>' : '<span>Checkpoint Loader</span>'}<i>↓</i><span>CLIP Encode × 2</span><i>↓</i><span>Load Image → VAE Encode</span><i>↓</i><span>KSampler（変化の強さ）</span><i>↓</i><span>VAE Decode</span><i>↓</i><span>Save Image</span>`;
    return;
  }
  $('#node-flow').innerHTML = `${$('#lora').value ? '<span>Checkpoint Loader</span><i>↓</i><span>LoRA Loader</span><i>↓</i>' : '<span>Checkpoint Loader</span><i>↓</i>'}<span>Text Encode × 2</span><i>↓</i><span>Empty Latent Image</span><i>↓</i><span>KSampler</span><i>↓</i><span>VAE Decode</span><i>↓</i><span>Save Image</span>`;
}

function clearInputImage() {
  if (state.inputImage?.previewUrl) URL.revokeObjectURL(state.inputImage.previewUrl);
  state.inputImage = null;
  $('#input-image-card').classList.add('hidden');
  $('#input-image-preview').removeAttribute('src');
  $('#input-image-name').textContent = '';
  $('#input-image-size').textContent = '';
  $('#input-image-file').value = '';
  setGenerationEnabled(Boolean(state.activeProfile?.enabled && state.activeProfile.capabilities[selectedGenerationType()]));
}

async function uploadInputImage(file) {
  if (file.size > 20 * 1024 * 1024) { showMessage('画像ファイルは20MB以下にしてください。', 'error'); return; }
  const form = new FormData();
  form.append('image', file);
  clearInputImage();
  $('#input-image-file').disabled = true;
  showMessage('入力画像をComfyUIへ送信しています…', 'info');
  try {
    const response = await fetch('/api/uploads/image', { method: 'POST', body: form });
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.detail || `画像アップロードに失敗しました (${response.status})`);
    clearInputImage();
    state.inputImage = { ...body, previewUrl: URL.createObjectURL(file) };
    $('#input-image-preview').src = state.inputImage.previewUrl;
    $('#input-image-name').textContent = body.name || file.name;
    $('#input-image-size').textContent = `${body.width} × ${body.height}`;
    $('#input-image-card').classList.remove('hidden');
    showMessage('入力画像をComfyUIへアップロードしました。', 'success');
  } catch (error) {
    showMessage(error.message, 'error');
  } finally {
    $('#input-image-file').disabled = false;
    setGenerationEnabled(Boolean(state.activeProfile?.enabled && state.activeProfile.capabilities[selectedGenerationType()]));
  }
}

function renderModelComponents(profile) {
  const holder = $('#model-components');
  const components = profile.ui?.model_components || [];
  if (!components.length) { holder.classList.add('hidden'); holder.innerHTML = ''; return; }
  holder.classList.remove('hidden');
  holder.innerHTML = `<p class="field-label">モデル構成</p><div class="field-grid two">${components.map((component) => {
    const items = state.models.filter((item) => item.type === component.asset_type && item.family === component.family && (!component.name_pattern || new RegExp(component.name_pattern, 'i').test(item.comfy_name || item.name)));
    return `<label class="field-label">${escapeHtml(component.label)}${component.required ? ' <span class="required">必須</span>' : ''}<select id="component-${escapeHtml(component.key)}"><option value="">${items.length ? '選択してください' : 'ComfyUIに対応モデルがありません'}</option>${items.map((item) => `<option value="${escapeHtml(item.comfy_name || item.name)}">${escapeHtml(item.name)}</option>`).join('')}</select></label>`;
  }).join('')}</div><p class="help-text">Fluxは対応UNET、2つのText Encoder、VAEが揃うまでQueue実行できません。</p>`;
  components.forEach((component) => $(`#component-${component.key}`).addEventListener('change', () => setGenerationEnabled(Boolean(profile.enabled && profile.capabilities.txt2img))));
}

async function loadModelProfile(model) {
  const requestId = ++state.profileRequest;
  state.activeProfile = null;
  setGenerationEnabled(false);
  try {
    const profile = await api(`/api/model-profiles/${encodeURIComponent(model.family)}`);
    if (requestId !== state.profileRequest || state.selected?.path !== model.path || state.selected?.family !== model.family) return;
    state.activeProfile = profile;
    syncGenerationMode(profile);
    const supported = profile.enabled && Boolean(profile.capabilities[selectedGenerationType()]);
    renderModelComponents(profile);
    $('#cfg-field').classList.toggle('hidden', profile.ui?.show_cfg === false);
    $('#negative-field').classList.toggle('hidden', profile.ui?.show_negative_prompt === false);
    $('#guidance-field').classList.toggle('hidden', !profile.ui?.show_guidance);
    const loraEnabled = supported && profile.capabilities.lora;
    $('#lora').disabled = !loraEnabled;
    $('#lora-weight').disabled = !loraEnabled;
    if (!loraEnabled) $('#lora').value = '';
    if (supported) {
      const defaults = profile.defaults;
      const resolution = `${defaults.width},${defaults.height}`;
      $('#width').value = defaults.width;
      $('#height').value = defaults.height;
      if ([...$('#resolution').options].some((option) => option.value === resolution)) {
        $('#resolution').value = resolution;
        $('#custom-size').classList.add('hidden');
      } else {
        $('#resolution').value = 'custom';
        $('#width').value = defaults.width;
        $('#height').value = defaults.height;
        $('#custom-size').classList.remove('hidden');
      }
      $('#steps').value = defaults.steps;
      $('#cfg').value = defaults.cfg;
      $('#guidance').value = profile.default_guidance ?? '';
      if (![...$('#sampler').options].some((option) => option.value === defaults.sampler)) {
        $('#sampler').add(new Option(defaults.sampler, defaults.sampler));
      }
      $('#sampler').value = defaults.sampler;
    }
    renderNodeFlow();
    setGenerationEnabled(supported);
    if (!supported) showMessage(`${profile.name}の${selectedGenerationType()}は現在対応していません。`, 'info');
    else if (profile.ui?.model_components?.length && !selectedComponentsReady()) showMessage('Flux Workflow構成を利用できます。実行・保存にはComfyUIにFlux用モデル一式が必要です。', 'info');
  } catch (error) {
    if (requestId !== state.profileRequest) return;
    $('#lora').disabled = true;
    $('#lora-weight').disabled = true;
    $('#model-components').classList.add('hidden');
    $('#cfg-field').classList.remove('hidden');
    $('#negative-field').classList.remove('hidden');
    $('#guidance-field').classList.add('hidden');
    setGenerationEnabled(false);
    showMessage(error.message || 'モデルProfileを読み込めませんでした。', 'error');
  }
}

async function refresh() {
  $('#scan-button').disabled = true;
  $('#scan-button').textContent = 'スキャン中…';
  try {
    const [status, models, profiles] = await Promise.all([api('/api/status'), api('/api/models'), api('/api/model-profiles')]);
    state.models = models;
    state.profiles = new Map(profiles.map((profile) => [profile.id, profile]));
    renderFamilyFilters();
    const lora = $('#lora');
    const selectedLora = lora.value;
    lora.innerHTML = '<option value="">使わない</option>' + models.filter((model) => model.type === 'lora').map((model) => `<option value="${escapeHtml(model.comfy_name || model.name)}">${escapeHtml(model.name)}</option>`).join('');
    if ([...lora.options].some((option) => option.value === selectedLora)) lora.value = selectedLora;
    $('#status-dot').classList.toggle('online', status.comfy.online);
    $('#status-label').textContent = status.comfy.online ? 'ComfyUI 接続中' : 'ComfyUI 未接続';
    $('#root-label').textContent = status.comfy_path ? status.comfy_path.split(/[\\/]/).slice(-2).join('/') : (status.model_source === 'comfy_api' ? 'ComfyUI API' : '環境未検出');
    if (status.settings) { $('#comfy-url').value = status.settings.comfy_url; $('#comfy-path').value = status.settings.comfy_path || ''; $('#web-port').value = status.settings.port || 7865; $('#open-browser').checked = status.settings.open_browser !== false; }
    if (state.selected && state.selected.type !== 'profile') state.selected = state.models.find((model) => model.path === state.selected.path) || null;
    renderModels(); renderSelected();
    if (state.selected) await loadModelProfile(state.selected);
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
  const generationType = selectedGenerationType();
  if (generationType === 'img2img' && !state.inputImage?.upload_id) throw new Error('img2imgには入力画像が必要です。画像を選択してください。');
  let width = null; let height = null;
  if (generationType === 'txt2img') {
    [width, height] = $('#resolution').value.split(',').map(Number);
    if ($('#resolution').value === 'custom') { width = Number($('#width').value); height = Number($('#height').value); }
  }
  const seed = $('#seed-mode').value === 'fixed' ? Number($('#seed').value) : Math.floor(Math.random() * 4294967296);
  const payload = { model: state.selected.type === 'profile' ? '' : (state.selected.comfy_name || state.selected.name), profile_id: state.selected.family,
    prompt,
    generation_type: generationType, input_image_id: generationType === 'img2img' ? state.inputImage.upload_id : null,
    denoise: Number($('#denoise').value),
    negative_prompt: state.activeProfile?.ui?.show_negative_prompt === false ? '' : $('#negative').value, width, height,
    steps: Number($('#steps').value), cfg: Number($('#cfg').value), sampler: $('#sampler').value,
    guidance: state.activeProfile?.ui?.show_guidance ? Number($('#guidance').value) : null,
    scheduler: state.activeProfile?.defaults.scheduler, seed,
    lora: $('#lora').value || null, lora_weight: Number($('#lora-weight').value) };
  for (const component of state.activeProfile?.ui?.model_components || []) payload[component.key] = $(`#component-${component.key}`)?.value || null;
  if (generationType === 'txt2img' && (!Number.isInteger(width) || !Number.isInteger(height) || width < 64 || height < 64 || width > 4096 || height > 4096)) throw new Error('解像度は64〜4096の整数で入力してください。');
  return payload;
}

async function run(payload = null) {
  if (state.isRunning) return;
  state.isRunning = true;
  try {
    const request = payload || getPayload();
    state.lastPayload = request;
    $('#run-button').disabled = true;
    $('#regenerate-button').disabled = true;
    showMessage('ComfyUIへWorkflowを送信しています…', 'info');
    const queued = await api('/api/workflow/run', { method: 'POST', body: JSON.stringify(request) });
    request.seed = queued.seed;
    request.input_image = queued.input_image || request.input_image || null;
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
    const modelLabel = request.profile_id === 'flux' ? `${request.diffusion_model} · ${request.clip_name1} · ${request.clip_name2} · ${request.vae_model}` : (state.selected?.name || request.model);
    const guidanceLabel = request.profile_id === 'flux' ? ` · Guidance ${request.guidance}` : '';
    const imageLabel = request.generation_type === 'img2img' ? ` · ${request.input_image?.name || request.input_image_id} · 変化 ${request.denoise}` : '';
    const sizeLabel = request.generation_type === 'img2img' ? `${request.input_image?.width} × ${request.input_image?.height}` : `${request.width} × ${request.height}`;
    $('#result-meta').textContent = `${modelLabel} · ${sizeLabel} · ${request.steps} steps · CFG ${request.cfg} · ${request.sampler}${guidanceLabel}${imageLabel} · Seed ${request.seed}`;
    $('#result').classList.remove('hidden');
    showMessage('画像生成が完了しました。', 'success');
  } catch (error) { showMessage(error.message, 'error'); }
  finally {
    state.isRunning = false;
    setGenerationEnabled(Boolean(state.activeProfile?.enabled && state.activeProfile.capabilities[selectedGenerationType()]));
    $('#regenerate-button').disabled = false;
  }
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
    showMessage(result.ready === false ? 'Flux Workflowテンプレートを生成しました。Flux用モデルが揃っていないため、保存・実行はできません。' : 'WorkflowDefinitionからAPI PromptとComfyUI Workflowを生成しました。', result.ready === false ? 'info' : 'success');
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
  if (state.selected?.type === 'profile' && state.family !== 'flux') { state.selected = null; state.activeProfile = null; renderSelected(); }
  renderModels();
});
$('#model-search').addEventListener('input', renderModels);
$('#scan-button').addEventListener('click', refresh);
$('#run-button').addEventListener('click', () => run());
$('#regenerate-button').addEventListener('click', () => run(state.lastPayload));
$('#resolution').addEventListener('change', () => $('#custom-size').classList.toggle('hidden', $('#resolution').value !== 'custom'));
$('#seed-mode').addEventListener('change', () => { $('#seed').disabled = $('#seed-mode').value !== 'fixed'; });
document.querySelectorAll('input[name="generation_type"]').forEach((radio) => radio.addEventListener('change', updateGenerationModeUI));
$('#input-image-file').addEventListener('change', (event) => { const file = event.target.files?.[0]; if (file) uploadInputImage(file); });
$('#remove-input-image').addEventListener('click', () => { clearInputImage(); showMessage('入力画像の選択を解除しました。', 'info'); });
$('#denoise').addEventListener('input', () => { $('#denoise-value').value = Number($('#denoise').value).toFixed(2); $('#denoise-value').textContent = Number($('#denoise').value).toFixed(2); });
$('#lora').addEventListener('change', renderNodeFlow);
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
