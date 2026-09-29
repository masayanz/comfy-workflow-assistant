const state = { models: [], comfyModels: [], profiles: new Map(), loraCompatibility: new Map(), family: 'all', selected: null, activeProfile: null, profileRequest: 0, lastPayload: null, inputImage: null, isRunning: false, comfyOnline: false, comfyUrl: '', diagnostics: null, importedWorkflowFile: null };
const $ = (selector) => document.querySelector(selector);

async function api(path, options = {}) {
  const headers = { ...(options.headers || {}) };
  if (!(options.body instanceof FormData) && !headers['Content-Type']) headers['Content-Type'] = 'application/json';
  const response = await fetch(path, { ...options, headers });
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body.detail;
    throw new Error(typeof detail === 'object' ? (detail.message || JSON.stringify(detail)) : (detail || `通信に失敗しました (${response.status})`));
  }
  return body;
}

function formatSize(size) {
  if (size == null) return 'サイズ不明';
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
    <span class="model-icon">◈</span><span class="model-info"><strong title="${escapeHtml(model.name)}">${escapeHtml(model.name)}</strong><small>${formatSize(model.size)} <span class="family-pill ${model.family}">${classificationName(model.classification || { family: model.family })}</span></small></span><span class="radio"></span></button>`).join('');
  list.querySelectorAll('.model-card').forEach((button) => button.addEventListener('click', () => {
    state.selected = items[Number(button.dataset.index)];
    state.activeProfile = null;
    renderModels(); renderSelected(); loadModelProfile(state.selected);
  }));
}

function familyName(family) { return ({ sdxl: 'SDXL', sd15: 'SD1.5', flux: 'Flux', unknown: '不明' })[family] || family; }
function classificationName(classification = {}) { return `${classification.variant === 'pony' ? 'Pony' : familyName(classification.family || 'unknown')}`; }
function escapeHtml(value) { return String(value).replace(/[&<>"']/g, (char) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]); }

function renderWorkflowAnalysis(data) {
  const holder = $('#workflow-analysis');
  const formatLabels = { comfyui_ui_workflow: 'ComfyUI UI Workflow', comfyui_api_prompt: 'ComfyUI API Prompt', unknown: '不明な形式' };
  const kindLabels = { txt2img: 'txt2img', img2img: 'img2img', upscale: 'Upscale', inpaint: 'inpaint', controlnet: 'ControlNet', mixed: '複合Workflow', unknown: '用途不明' };
  const assetLabels = { checkpoint: 'Checkpoint', lora: 'LoRA', vae: 'VAE', diffusion_model: 'Diffusion Model / UNET', text_encoder: 'Text Encoder', upscale_model: 'Upscale Model', controlnet: 'ControlNet', clip_vision: 'CLIP Vision', ipadapter: 'IPAdapter' };
  const familyLabels = { sdxl: 'SDXL', sd15: 'SD1.5', flux: 'Flux', unknown: '不明' };
  const confidenceLabels = { high: '高', low: '低', unknown: '不明' };
  const assetStatus = { found: '✓ 利用可能', missing: '✕ 不足', unknown: '? 照合不明' };
  const nodeStatus = { available: '✓ 登録済み', missing: '✕ 不足', unknown: '? 照合不明' };
  const compatibilityLabel = { compatible: '互換性あり', compatible_with_warning: '要確認', incompatible: '非互換', unknown: '不明' };
  const modelRows = (data.models || []).map((item) => `<li><span>${escapeHtml(assetLabels[item.type] || item.type)} · ${escapeHtml(item.name)}${item.family && item.family !== 'unknown' ? `<br><small>${escapeHtml(familyLabels[item.family] || item.family)}${item.classification?.variant === 'pony' ? ' / Pony' : ''} · 確度 ${escapeHtml(confidenceLabels[item.classification?.confidence] || '不明')}</small>` : ''}</span><small class="analysis-state ${escapeHtml(item.status)}">${assetStatus[item.status] || '不明'}</small></li>`).join('');
  const nodeRows = (data.nodes || []).slice(0, 80).map((node) => `<li><span>${escapeHtml(node.title || node.type)} <small>(${escapeHtml(node.type)} · ID ${escapeHtml(node.id)})</small></span><small class="analysis-state ${escapeHtml(node.availability)}">${nodeStatus[node.availability] || ''}</small></li>`).join('');
  const compatibilityRows = (data.compatibility || []).map((item) => `<li><span>${escapeHtml(item.checkpoint)} + ${escapeHtml(item.lora)}<br><small>${escapeHtml(item.message || '')}</small></span><small class="analysis-state ${escapeHtml(item.status)}">${compatibilityLabel[item.status] || '不明'}</small></li>`).join('');
  const missingNodeRows = (data.missing_nodes || []).map((node) => `<li><span>${escapeHtml(node.type)} · ID ${escapeHtml(node.node_id)}</span><small class="analysis-state missing">不足</small></li>`).join('');
  const customRows = (data.custom_nodes || []).map((node) => `<li><span>${escapeHtml(node.type)} · ID ${escapeHtml(node.node_id)}</span><small class="analysis-state ${escapeHtml(node.status)}">${nodeStatus[node.status] || '不明'}</small></li>`).join('');
  const unknownRows = (data.unknown_nodes || []).map((node) => `<li><span>${escapeHtml(node.type)} · ID ${escapeHtml(node.node_id)}</span><small class="analysis-state ${escapeHtml(node.status)}">解析対象外</small></li>`).join('');
  const warningRows = (data.warnings || []).map((warning) => `<li>${escapeHtml(warning)}</li>`).join('');
  const flowRows = (data.flow || []).map((item) => `<li>${escapeHtml(item.title || item.type)}${item.from?.length ? ` ← ${escapeHtml(item.from.join(', '))}` : ''}</li>`).join('');
  const parameters = JSON.stringify(data.parameters || {}, null, 2);
  const inventoryStatus = data.comfy_online === null ? '形式不明のため親機照合なし' : data.comfy_online ? '親ComfyUIに接続中' : '親ComfyUIに接続できず、在庫照合は不明';
  holder.innerHTML = `
    <div class="analysis-summary"><strong>${escapeHtml(data.summary?.title || 'Workflowを解析しました。')}</strong>
      <div>形式: ${escapeHtml(formatLabels[data.workflow_type] || data.workflow_type)} · 用途: ${escapeHtml(kindLabels[data.generation_type] || data.generation_type)}</div>
      <div>Nodes: ${Number(data.node_count) || 0} · Links: ${Number(data.link_count) || 0}</div>
      <div>${escapeHtml(inventoryStatus)}</div></div>
    ${data.summary?.steps?.length ? `<section><h4>処理の概要</h4><ol>${data.summary.steps.map((step) => `<li>${escapeHtml(step)}</li>`).join('')}</ol></section>` : ''}
    <section><h4>使用モデル</h4>${modelRows ? `<ul class="analysis-list">${modelRows}</ul>` : '<p>モデル参照を検出しませんでした。</p>'}</section>
    ${compatibilityRows ? `<section><h4>Checkpoint / LoRA互換性</h4><ul class="analysis-list">${compatibilityRows}</ul></section>` : ''}
    ${missingNodeRows ? `<section><h4>親ComfyUIにないノード</h4><ul class="analysis-list">${missingNodeRows}</ul></section>` : ''}
    ${customRows ? `<section><h4>Custom Node</h4><ul class="analysis-list">${customRows}</ul></section>` : ''}
    ${unknownRows ? `<section><h4>解析対象外のノード</h4><ul class="analysis-list">${unknownRows}</ul></section>` : ''}
    <section><h4>ノード一覧</h4><ul class="analysis-list">${nodeRows || '<li>ノードがありません。</li>'}</ul>${data.node_count > 80 ? `<p>先頭80件を表示しました（全${Number(data.node_count)}件）。</p>` : ''}</section>
    ${flowRows ? `<section><h4>簡易フロー</h4><ol>${flowRows}</ol></section>` : ''}
    <details><summary>認識した生成パラメータ</summary><pre>${escapeHtml(parameters)}</pre></details>
    ${data.vram_warnings?.length ? `<section><h4>RTX 3060 12GB向け注意</h4><ul class="analysis-list">${data.vram_warnings.map((item) => `<li class="analysis-warning">${escapeHtml(item)}</li>`).join('')}</ul><p>VRAM使用量の数値予測ではありません。</p></section>` : ''}
    ${warningRows ? `<section><h4>確認事項</h4><ul class="analysis-list">${warningRows}</ul></section>` : ''}
    <div class="analysis-actions"><button class="button secondary analysis-download" type="button">読み込んだJSONをダウンロード</button><button class="button secondary analysis-open-comfy" type="button" ${state.comfyUrl ? '' : 'disabled'}>ComfyUIを開く</button><button class="button secondary analysis-open-diagnostics" type="button">環境診断を開く</button></div>
    <p class="help-text">解析のみ実行しました。Queueへの送信やWorkflowの変更は行っていません。ComfyUIで編集する場合は、キャンバスのWorkflow読み込みからJSONを選択してください。</p>`;
  holder.classList.remove('hidden');
}

function loraCompatibilityLabel(status) {
  return ({ compatible: '互換', compatible_with_warning: '要確認', incompatible: '非互換', unknown: 'Unknown / 要確認' })[status] || '未判定 / 要確認';
}

function renderLoraOptions() {
  const select = $('#lora');
  if (!select) return;
  const selected = select.value;
  select.innerHTML = '<option value="">使わない</option>' + state.models.filter((model) => model.type === 'lora').map((model) => {
    const name = model.comfy_name || model.name;
    const entry = state.loraCompatibility.get(name);
    const compatibility = entry?.compatibility;
    const status = compatibility?.status || 'unknown';
    const label = `${model.name} · ${classificationName(entry?.classification || model.classification || { family: model.family })} / ${loraCompatibilityLabel(status)}`;
    return `<option value="${escapeHtml(name)}" ${status === 'incompatible' ? 'disabled' : ''}>${escapeHtml(label)}</option>`;
  }).join('');
  if ([...select.options].some((option) => option.value === selected)) select.value = selected;
  renderLoraCompatibility();
}

function renderLoraCompatibility() {
  const holder = $('#lora-compatibility');
  if (!holder) return;
  const name = $('#lora')?.value;
  if (!name) { holder.textContent = 'LoRAを選ぶとCheckpointとの互換性を表示します。Unknownは警告のみでQueueできます。'; holder.className = 'lora-compatibility help-text'; return; }
  const entry = state.loraCompatibility.get(name);
  const result = entry?.compatibility;
  const classification = entry?.classification || {};
  const choices = [...state.profiles.values()].map((profile) => [profile.id, profile.name]);
  choices.push(['unknown', '不明']);
  holder.className = `lora-compatibility ${result?.status || 'unknown'}`;
  holder.innerHTML = `<strong>${escapeHtml(loraCompatibilityLabel(result?.status))}</strong>: ${escapeHtml(result?.message || '互換性を判定できません。Queue前に確認してください。')}<div class="lora-classification"><label>LoRA系統<select id="lora-family-select">${choices.map(([value,label]) => `<option value="${escapeHtml(value)}" ${(classification.family || entry?.family || 'unknown') === value ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('')}</select></label><label>variant<select id="lora-variant-select" ${(classification.family || entry?.family) !== 'sdxl' ? 'disabled' : ''}><option value="" ${classification.variant !== 'pony' ? 'selected' : ''}>一般SDXL</option><option value="pony" ${classification.variant === 'pony' ? 'selected' : ''}>Pony</option></select></label></div>`;
  const saveClassification = async () => {
    try {
      const family = $('#lora-family-select').value;
      const variant = family === 'sdxl' ? $('#lora-variant-select').value || null : null;
      await api('/api/models/classify', { method: 'POST', body: JSON.stringify({ model: name, asset_type: 'lora', family, variant }) });
      state.models = state.models.map((model) => (model.comfy_name || model.name) === name && model.type === 'lora'
        ? { ...model, family, variant, classification: { family, variant, confidence: 'high', source: 'manual', evidence: ['ユーザーが手動分類'] } } : model);
      await loadLoraCompatibility(state.selected);
      showMessage('LoRAの分類を保存しました。', 'success');
    } catch (error) { showMessage(error.message, 'error'); }
  };
  $('#lora-family-select').addEventListener('change', () => { $('#lora-variant-select').disabled = $('#lora-family-select').value !== 'sdxl'; saveClassification(); });
  $('#lora-variant-select').addEventListener('change', saveClassification);
}

async function loadLoraCompatibility(model) {
  if (!model || model.type !== 'checkpoint') {
    state.loraCompatibility.clear(); renderLoraOptions(); return;
  }
  try {
    const result = await api(`/api/models/compatibility?checkpoint=${encodeURIComponent(model.comfy_name || model.name)}`);
    state.loraCompatibility = new Map((result.loras || []).map((item) => [item.comfy_name || item.name, item]));
    const checkpointClassification = result.checkpoint_classification || result.loras?.[0]?.compatibility?.checkpoint;
    state.models = state.models.map((item) => {
      const name = item.comfy_name || item.name;
      if (item.type === 'checkpoint' && name === (model.comfy_name || model.name) && checkpointClassification) {
        return { ...item, family: checkpointClassification.family, variant: checkpointClassification.variant, classification: checkpointClassification };
      }
      const apiLora = state.loraCompatibility.get(name);
      return item.type === 'lora' && apiLora ? { ...item, ...apiLora } : item;
    });
    const selectedCheckpoint = state.models.find((item) => item.type === 'checkpoint' && (item.comfy_name || item.name) === (model.comfy_name || model.name));
    if (selectedCheckpoint && state.selected?.type === 'checkpoint' && (state.selected.comfy_name || state.selected.name) === (model.comfy_name || model.name)) {
      state.selected = selectedCheckpoint;
      renderModels(); renderSelected();
    }
    renderLoraOptions();
  } catch (error) {
    state.loraCompatibility.clear(); renderLoraOptions();
  }
}

function renderSelected() {
  const holder = $('#selected-model');
  if (selectedGenerationType() === 'upscale') {
    renderUpscaleSelection();
    setGenerationEnabled(true);
    return;
  }
  if (!state.selected) {
    holder.innerHTML = '<span class="model-placeholder">左からCheckpointを選択してください</span>';
    $('#model-family-tag').textContent = '未選択';
    $('#model-components').classList.add('hidden');
    setGenerationEnabled(false);
    return;
  }
  $('#model-family-tag').textContent = classificationName(state.selected.classification || { family: state.selected.family, variant: state.selected.variant });
  renderNodeFlow();
  setGenerationEnabled(false);
  if (state.selected.type === 'profile') {
    holder.innerHTML = `<span class="model-icon large">◈</span><div class="selected-model-info"><strong>${escapeHtml(state.selected.name)}</strong><small>ComfyUIにある個別モデルを下で指定してください</small></div>`;
    return;
  }
  const classificationOptions = [...state.profiles.values()].map((profile) => [profile.id, profile.name]);
  classificationOptions.push(['unknown', '不明']);
  const classification = state.selected.classification || {};
  const variant = classification.variant || state.selected.variant || '';
  const sourceLabel = classification.source === 'metadata' ? 'safetensors metadata' : classification.source === 'manual' ? '手動分類' : classification.source === 'filename' ? 'ファイル名候補' : '未判定';
  holder.innerHTML = `<span class="model-icon large">◈</span><div class="selected-model-info"><strong>${escapeHtml(state.selected.name)}</strong><small>${formatSize(state.selected.size)} · 分類元: ${sourceLabel} (${classification.confidence || 'unknown'})</small></div><div class="classification-controls"><select class="family-select" id="family-select" aria-label="モデル系統">${classificationOptions.map(([value,label]) => `<option value="${escapeHtml(value)}" ${state.selected.family === value ? 'selected' : ''}>${escapeHtml(label)}</option>`).join('')}</select><select class="variant-select" id="variant-select" aria-label="モデルvariant" ${state.selected.family !== 'sdxl' ? 'disabled' : ''}><option value="" ${!variant ? 'selected' : ''}>一般SDXL</option><option value="pony" ${variant === 'pony' ? 'selected' : ''}>Pony</option></select></div>`;
  const saveClassification = async () => {
    try {
      const family = $('#family-select').value;
      const selectedVariant = family === 'sdxl' ? $('#variant-select').value || null : null;
      const updated = await api('/api/models/classify', { method: 'POST', body: JSON.stringify({ model: state.selected.comfy_name || state.selected.name, asset_type: 'checkpoint', family, variant: selectedVariant }) });
      state.selected = updated; state.models = state.models.map((model) => model.path === updated.path ? updated : model); renderModels(); renderSelected();
      await loadModelProfile(updated); await loadLoraCompatibility(updated);
      showMessage('Checkpointの分類を保存しました。', 'success');
    } catch (error) { showMessage(error.message, 'error'); }
  };
  $('#family-select').addEventListener('change', () => { $('#variant-select').disabled = $('#family-select').value !== 'sdxl'; saveClassification(); });
  $('#variant-select').addEventListener('change', saveClassification);
}

function renderFamilyFilters() {
  const filters = $('#filters');
  const entries = [['all', 'すべて'], ...[...state.profiles.values()].map((profile) => [profile.id, familyName(profile.id)]), ['unknown', '不明']];
  filters.innerHTML = entries.map(([id, label]) => `<button class="filter ${state.family === id ? 'active' : ''}" data-filter="${escapeHtml(id)}">${escapeHtml(label)}</button>`).join('');
}

function setGenerationEnabled(enabled) {
  const mode = selectedGenerationType();
  if (mode === 'upscale') {
    enabled = state.comfyOnline && Boolean(state.inputImage?.upload_id) && Boolean($('#upscale-model').value)
      && state.comfyModels.some((model) => model.type === 'upscale_model' && (model.comfy_name || model.name) === $('#upscale-model').value);
  }
  const imageReady = !['img2img', 'upscale'].includes(mode) || Boolean(state.inputImage?.upload_id);
  const buildReady = enabled && imageReady && !state.isRunning;
  const assetsReady = buildReady && selectedComponentsReady();
  $('#build-button').disabled = !buildReady;
  for (const selector of ['#save-ui-button', '#save-api-button', '#run-button']) $(selector).disabled = !assetsReady;
}

function renderUpscaleModelOptions() {
  const select = $('#upscale-model');
  const previous = select.value;
  const models = state.comfyModels.filter((model) => model.type === 'upscale_model');
  select.innerHTML = `<option value="">${models.length ? 'モデルを選択してください' : '親機ComfyUIにUpscale Modelがありません'}</option>` +
    models.map((model) => {
      const scaleLabel = model.scale ? ` · 推定 ${model.scale}x` : '';
      return `<option value="${escapeHtml(model.comfy_name || model.name)}">${escapeHtml(model.name)} · ${formatSize(model.size)}${scaleLabel}</option>`;
    }).join('');
  if (models.some((model) => (model.comfy_name || model.name) === previous)) select.value = previous;
  updateUpscaleEstimate();
  renderUpscaleSelection();
}

function renderUpscaleSelection() {
  if (selectedGenerationType() !== 'upscale') return;
  const model = state.comfyModels.find((item) => item.type === 'upscale_model' && (item.comfy_name || item.name) === $('#upscale-model').value);
  const input = state.inputImage;
  $('#selected-model').innerHTML = `<span class="model-icon large">⇧</span><div class="selected-model-info"><strong>${escapeHtml(model?.name || 'Upscale Model未選択')}</strong><small>${input ? `${escapeHtml(input.name)} · ${input.width} × ${input.height}` : '入力画像をアップロードしてください'}</small></div>`;
  $('#model-family-tag').textContent = 'Upscale';
}

function updateUpscaleEstimate() {
  const estimate = $('#upscale-estimate');
  if (!estimate) return;
  const input = state.inputImage;
  const selected = state.comfyModels.find((model) => model.type === 'upscale_model' && (model.comfy_name || model.name) === $('#upscale-model').value);
  if (!input?.width || !input?.height) {
    estimate.textContent = '入力画像を選ぶと予想出力サイズを表示します。';
    return;
  }
  if (!selected?.scale) {
    estimate.textContent = `入力 ${input.width} × ${input.height} · このモデル名から倍率を特定できないため出力サイズは実行後に表示します。`;
    return;
  }
  estimate.textContent = `入力 ${input.width} × ${input.height} → 予想 ${input.width * selected.scale} × ${input.height * selected.scale}（モデル名からの推定 ${selected.scale}x）`;
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
  const mode = selectedGenerationType();
  const imageMode = mode === 'img2img' || mode === 'upscale';
  const upscaleMode = mode === 'upscale';
  $('#generation-type-label').textContent = upscaleMode ? 'UPSCALE' : imageMode ? 'IMG2IMG' : 'TXT2IMG';
  $('.layout').classList.toggle('upscale-mode', upscaleMode);
  document.body.classList.toggle('upscale-mode', upscaleMode);
  $('#prompt-fields').classList.toggle('hidden', upscaleMode);
  $('#generation-settings').classList.toggle('hidden', upscaleMode);
  $('#image-input-section').classList.toggle('hidden', !imageMode);
  $('#denoise-section').classList.toggle('hidden', upscaleMode);
  $('#upscale-options').classList.toggle('hidden', !upscaleMode);
  $('#resolution-grid').classList.toggle('hidden', imageMode);
  $('.builder-panel h2').textContent = upscaleMode ? '画像をアップスケール' : 'ワークフローを作成';
  $('.preview-panel .explain').textContent = upscaleMode
    ? '入力画像をComfyUIのUpscale Modelで拡大し、Save Imageへ保存します。CheckpointやPromptは使いません。'
    : '選択したCheckpointからモデル・CLIP・VAEを読み込み、プロンプトと生成設定をKSamplerへ渡します。';
  renderUpscaleModelOptions();
  renderNodeFlow();
  const profile = state.activeProfile;
  setGenerationEnabled(upscaleMode || Boolean(profile?.enabled && profile.capabilities[mode]));
}

function selectedComponentsReady() {
  const components = state.activeProfile?.ui?.model_components || [];
  return components.every((component) => !component.required || Boolean($(`#component-${component.key}`)?.value));
}

function renderNodeFlow() {
  if (selectedGenerationType() === 'upscale') {
    $('#model-family-tag').textContent = 'Upscale';
    $('#node-flow').innerHTML = '<span>Load Image</span><i>↓</i><span>Load Upscale Model</span><i>↓</i><span>Upscale Image</span><i>↓</i><span>Save Image</span>';
    return;
  }
  $('#model-family-tag').textContent = state.selected ? familyName(state.selected.family) : 'SDXL';
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
  updateUpscaleEstimate();
  renderUpscaleSelection();
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
    updateUpscaleEstimate();
    renderUpscaleSelection();
    showMessage('入力画像をComfyUIへアップロードしました。', 'success');
  } catch (error) {
    showMessage(error.message, 'error');
  } finally {
    $('#input-image-file').disabled = false;
    renderUpscaleSelection();
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
    const supported = selectedGenerationType() === 'upscale' || (profile.enabled && Boolean(profile.capabilities[selectedGenerationType()]));
    if (selectedGenerationType() === 'upscale') {
      $('#model-components').classList.add('hidden');
      $('#model-components').innerHTML = '';
    } else renderModelComponents(profile);
    $('#cfg-field').classList.toggle('hidden', profile.ui?.show_cfg === false);
    $('#negative-field').classList.toggle('hidden', profile.ui?.show_negative_prompt === false);
    $('#guidance-field').classList.toggle('hidden', !profile.ui?.show_guidance);
    const loraEnabled = selectedGenerationType() !== 'upscale' && supported && profile.capabilities.lora;
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
    await loadLoraCompatibility(model);
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
    const [status, models, comfyModels, profiles] = await Promise.all([
      api('/api/status'), api('/api/models'), api('/api/comfy/models').catch(() => []), api('/api/model-profiles'),
    ]);
    state.models = models;
    state.comfyModels = comfyModels;
    state.comfyOnline = Boolean(status.comfy.online);
    state.comfyUrl = status.settings?.comfy_url || '';
    state.profiles = new Map(profiles.map((profile) => [profile.id, profile]));
    renderFamilyFilters();
    renderLoraOptions();
    renderUpscaleModelOptions();
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
  const generationType = selectedGenerationType();
  if (generationType === 'upscale') {
    if (!state.inputImage?.upload_id) throw new Error('upscaleには入力画像が必要です。画像を選択してください。');
    if (!state.comfyModels.some((model) => model.type === 'upscale_model')) throw new Error('親機ComfyUIにUpscale Modelがありません。環境診断から配置先と導入手順を確認してください。');
    if (!$('#upscale-model').value) throw new Error('Upscale Modelを選択してください。');
    return { generation_type: 'upscale', input_image_id: state.inputImage.upload_id, upscale_model: $('#upscale-model').value };
  }
  if (!state.selected) throw new Error('Checkpointを選択してください。');
  const prompt = $('#prompt').value.trim();
  if (!prompt) throw new Error('作りたいものを入力してください。');
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

function waitForImage(img) {
  if (img.complete) return Promise.resolve();
  return new Promise((resolve) => {
    img.addEventListener('load', resolve, { once: true });
    img.addEventListener('error', resolve, { once: true });
    setTimeout(resolve, 30000);
  });
}

async function run(payload = null) {
  if (state.isRunning) return;
  state.isRunning = true;
  const startedAt = performance.now();
  try {
    const request = payload || getPayload();
    state.lastPayload = request;
    const isUpscale = request.generation_type === 'upscale';
    $('#run-button').disabled = true;
    $('#regenerate-button').disabled = true;
    showMessage(isUpscale ? 'Upscale WorkflowをComfyUIへ送信しています…' : 'ComfyUIへWorkflowを送信しています…', 'info');
    const queued = await api('/api/workflow/run', { method: 'POST', body: JSON.stringify(request) });
    if (queued.seed != null) request.seed = queued.seed;
    request.input_image = queued.input_image || request.input_image || null;
    request.upscale_scale = queued.upscale_scale ?? request.upscale_scale;
    state.lastPayload = request;
    showMessage(isUpscale ? 'UpscaleをQueueへ登録しました。' : `Queue登録済み · Seed ${queued.seed}`, 'info');
    const deadline = Date.now() + 15 * 60 * 1000;
    let result;
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      result = await api(`/api/workflow/status/${encodeURIComponent(queued.prompt_id)}`);
      if (result.status === 'COMPLETED') break;
      if (result.status === 'ERROR') throw new Error(result.message || 'ComfyUIで画像生成に失敗しました。');
      showMessage(result.status === 'RUNNING' ? (isUpscale ? '画像をアップスケール中です…' : '画像を生成中です…') : 'Queue待機中です…', 'info');
    }
    if (!result || result.status !== 'COMPLETED') throw new Error('画像生成がタイムアウトしました。ComfyUIの状態を確認してください。');
    if (!result.images?.length) throw new Error('ComfyUIは処理を完了しましたが、画像が見つかりませんでした。SaveImageノードと出力を確認してください。');
    $('#result-images').innerHTML = result.images.map((image) => `<a href="${image.url}" target="_blank"><img src="${image.url}" alt="生成画像"></a>`).join('');
    if (isUpscale) {
      const outputImages = [...$('#result-images').querySelectorAll('img')];
      await Promise.all(outputImages.map(waitForImage));
      const input = request.input_image || state.inputImage || {};
      const firstOutput = outputImages[0];
      const inputSize = `${input.width || '?'} × ${input.height || '?'}`;
      const outputSize = firstOutput?.naturalWidth ? `${firstOutput.naturalWidth} × ${firstOutput.naturalHeight}` : '出力サイズ不明';
      const model = state.comfyModels.find((item) => item.type === 'upscale_model' && (item.comfy_name || item.name) === request.upscale_model);
      const scaleLabel = request.upscale_scale ? ` · 倍率 ${request.upscale_scale}x` : '';
      const seconds = ((performance.now() - startedAt) / 1000).toFixed(1);
      $('#result-meta').textContent = `${inputSize} → ${outputSize} · ${model?.name || request.upscale_model}${scaleLabel} · ${seconds}秒`;
      $('#result').classList.remove('hidden');
      showMessage('アップスケールが完了しました。', 'success');
      return;
    }
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

function renderDiagnosticAssets(models, kind) {
  const items = models.filter((item) => item.type === kind);
  if (!items.length) return '<p class="help-text">ComfyUI APIの認識一覧は0件です。</p>';
  return `<ul class="diagnostic-assets">${items.map((item) => `<li><span>${escapeHtml(item.comfy_name || item.name)}</span><small>${formatSize(item.size)}${item.family ? ` · ${escapeHtml(classificationName(item.classification || { family: item.family }))}` : ''}${item.classification ? ` · ${escapeHtml(item.classification.source)} / ${escapeHtml(item.classification.confidence)}` : ''}${item.inventory_source === 'loader_enum' ? ' · Loader選択肢' : ''}</small></li>`).join('')}</ul>`;
}

async function loadDiagnostics() {
  $('#diagnostics-status').textContent = '親機ComfyUIから取得中…';
  $('#diagnostics-content').innerHTML = '<div class="empty-state">モデルとノード情報を問い合わせています…</div>';
  try {
    const data = await api('/api/diagnostics');
    state.diagnostics = data;
    state.comfyModels = data.models || state.comfyModels;
    renderUpscaleModelOptions();
    if (!data.online) {
      const issues = (data.issues || []).map((issue) => `<div class="diagnostic-issue ${escapeHtml(issue.severity.toLowerCase())}"><strong>${escapeHtml(issue.severity)}</strong><span>${escapeHtml(issue.message)}</span></div>`).join('');
      $('#diagnostics-content').innerHTML = `${issues || ''}<div class="diagnostic-warning">ComfyUIがオフラインです。接続設定と親機の起動状態を確認してください。</div>`;
      $('#diagnostics-status').textContent = 'OFFLINE';
      return;
    }
    const server = data.server || {};
    const device = server.devices?.[0] || {};
    const vram = device.vram_total ? `${(device.vram_total / 1024 ** 3).toFixed(1)} GB` : '不明';
    const folders = data.model_folder_paths || {};
    const folderDetails = Object.entries(folders).map(([category, paths]) => `<li><strong>${escapeHtml(category)}</strong><span>${paths.length ? paths.map(escapeHtml).join('<br>') : 'APIから保存先を取得できません'}</span></li>`).join('');
    const counts = data.model_counts || {};
    const categories = [
      ['Checkpoints', 'checkpoints', 'checkpoint'], ['LoRA', 'loras', 'lora'], ['VAE', 'vae', 'vae'],
      ['UNET / Diffusion Models', 'diffusion_models', 'diffusion_model'], ['Text Encoders', 'text_encoders', 'text_encoder'],
      ['Upscale Models', 'upscale_models', 'upscale_model'], ['ControlNet', 'controlnet', 'controlnet'],
    ];
    const modelSections = categories.map(([label, key, kind]) => `<details class="diagnostic-section"><summary>${escapeHtml(label)} <span>${counts[key] ?? 0}</span></summary>${renderDiagnosticAssets(data.models || [], kind)}</details>`).join('');
    const requiredNodes = Object.entries(data.upscale?.node_available || {}).map(([name, available]) => `<li>${escapeHtml(name)} <span class="diagnostic-state ${available ? 'ready' : 'missing'}">${available ? '利用可能' : '不足'}</span></li>`).join('');
    const upscaleFolder = folders.upscale_models?.[0] || (server.comfyui_root ? `${server.comfyui_root}\\models\\upscale_models` : 'models\\upscale_models');
    const upscaleModels = data.upscale?.models || [];
    const upscaleHint = upscaleModels.length
      ? '<p class="diagnostic-ok">親機ComfyUIがUpscale Modelを認識しています。</p>'
      : `<div class="diagnostic-warning">Upscale Modelは0件です。親機の保存先: <code>${escapeHtml(upscaleFolder)}</code><br><code>scripts/Install-RealESRGAN-On-Parent.ps1</code>を親機へコピーし、親機PowerShellで実行してください。ComfyUIを再起動してから再取得します。</div>`;
    const sd15 = data.sd15 || {};
    const misplacedLoras = data.checkpoint_metadata?.misplaced_loras || [];
    const misplacedLoraSummary = misplacedLoras.length
      ? `Checkpointフォルダ内にLoRA候補が${misplacedLoras.length}件あります（LoRA Loaderの認識対象外）。推奨保存先: models/loras。自動移動はしません。 ${misplacedLoras.map((item) => escapeHtml(item.comfy_name || item.name)).join(', ')}`
      : 'Checkpointフォルダ内にLoRAメタデータの候補はありません。';
    const sd15Summary = sd15.checkpoint_count
      ? `SD1.5として分類済み: ${sd15.checkpoint_count}件 (${(sd15.checkpoints || []).map(escapeHtml).join(', ')})`
      : `SD1.5と確認できたCheckpointは0件です。未分類: ${(sd15.unclassified_checkpoint_names || []).map(escapeHtml).join(', ') || 'なし'}`;
    const flux = data.flux || {};
    const fluxSummary = flux.missing_assets?.length
      ? `不足: ${flux.missing_assets.map(escapeHtml).join('、')}`
      : 'Profileが要求するFluxモデル構成を認識しています。';
    const packs = data.custom_nodes?.packs || [];
    const issues = data.issues || [];
    const issueHtml = issues.length ? issues.map((issue) => `<div class="diagnostic-issue ${escapeHtml(issue.severity.toLowerCase())}"><strong>${escapeHtml(issue.severity)}</strong><span>${escapeHtml(issue.message)}</span></div>`).join('') : '<p>確認された問題はありません。</p>';
    $('#diagnostics-content').innerHTML = `
      <section class="diagnostic-summary"><strong>ComfyUI ${escapeHtml(server.version || 'version不明')} · ONLINE</strong><span>URL: ${escapeHtml(server.url || data.url || '不明')}</span><span>親機ルート: ${escapeHtml(server.comfyui_root || '不明')}</span><span>GPU: ${escapeHtml(device.name || server.os || '不明')} · VRAM ${vram}</span><span>Python ${escapeHtml(server.python_version || '不明')} · PyTorch ${escapeHtml(server.pytorch_version || '不明')}</span><small>モデル一覧: 親機ComfyUI API · ノード型 ${data.node_count ?? 0}</small></section>
      <section><h3>診断結果</h3><div class="diagnostic-issues">${issueHtml}</div></section>
      <section><h3>Upscale 実行要件</h3><ul class="diagnostic-nodes">${requiredNodes}</ul>${upscaleHint}</section>
      <section><h3>モデル棚卸し</h3>${modelSections}</section>
      <section><h3>SD1.5確認</h3><p>対応Workflow: 実装済み · 実Checkpoint: ${sd15.checkpoint_count || 0}件 · 実機確認: ${sd15.checkpoint_count ? '可能' : '未実施'}</p><p>${sd15Summary}</p><p class="help-text">未分類Checkpointは名前だけでSD1.5と判断せず、必要ならユーザーが分類してください。</p><p>${misplacedLoraSummary}</p></section>
      <section><h3>Flux不足</h3><p>${fluxSummary}</p>${Object.entries(flux.available_components || {}).map(([key, names]) => `<p class="help-text">${escapeHtml(key)}: ${names.length ? names.map(escapeHtml).join(', ') : '認識なし'}</p>`).join('')}</section>
      <section><h3>ComfyUI登録済みCustom Node</h3><p>${data.custom_nodes?.pack_count ?? 0} packs · ${data.custom_nodes?.node_count ?? 0} node types</p><p class="help-text">object_infoのpython_moduleから集計しています。</p><ul class="diagnostic-assets">${packs.map((pack) => `<li><span>${escapeHtml(pack.name)}</span><small>${pack.node_count} nodes</small></li>`).join('') || '<li><span>登録済みCustom Nodeなし</span></li>'}</ul></section>
      <section><h3>親機モデル保存先</h3><ul class="diagnostic-paths">${folderDetails || '<li>ComfyUI APIから保存先一覧を取得できません。</li>'}</ul></section>`;
    $('#diagnostics-status').textContent = 'ComfyUI APIで照合済み';
  } catch (error) {
    $('#diagnostics-status').textContent = '取得エラー';
    $('#diagnostics-content').innerHTML = `<div class="diagnostic-warning">${escapeHtml(error.message)}</div>`;
  }
}

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
$('#lora').addEventListener('change', () => { renderNodeFlow(); renderLoraCompatibility(); });
$('#upscale-model').addEventListener('change', () => { updateUpscaleEstimate(); renderUpscaleSelection(); setGenerationEnabled(true); });
$('#settings-open').addEventListener('click', () => $('#settings-dialog').showModal());
$('#diagnostics-open').addEventListener('click', () => { $('#diagnostics-dialog').showModal(); loadDiagnostics(); });
$('#diagnostics-close').addEventListener('click', () => $('#diagnostics-dialog').close());
$('#diagnostics-refresh').addEventListener('click', loadDiagnostics);
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
$('#workflow-import-file').addEventListener('change', () => {
  state.importedWorkflowFile = null;
  $('#workflow-analysis').classList.add('hidden');
  $('#workflow-analysis').replaceChildren();
});
$('#workflow-import-button').addEventListener('click', async (event) => {
  const button = event.currentTarget;
  const file = $('#workflow-import-file').files?.[0];
  if (!file) { showMessage('解析するComfyUI Workflow JSONを選択してください。', 'error'); return; }
  if (!file.name.toLowerCase().endsWith('.json')) { showMessage('JSONファイルを選択してください。', 'error'); return; }
  if (file.size > 10 * 1024 * 1024) { showMessage('Workflow JSONは10MB以下にしてください。', 'error'); return; }
  $('#workflow-analysis').classList.add('hidden');
  $('#workflow-analysis').replaceChildren();
  button.disabled = true;
  button.textContent = '親ComfyUIと照合中…';
  try {
    const form = new FormData();
    form.append('file', file, file.name);
    const result = await api('/api/workflow/import', { method: 'POST', body: form });
    state.importedWorkflowFile = file;
    renderWorkflowAnalysis(result);
    showMessage('Workflowの読み込みと解析が完了しました。Queueへは送信していません。', result.valid ? 'success' : 'info');
  } catch (error) {
    showMessage(error.message, 'error');
  } finally {
    button.disabled = false;
    button.textContent = 'Workflowを読み込んで解析';
  }
});
$('#workflow-analysis').addEventListener('click', (event) => {
  if (event.target.closest('.analysis-download')) {
    if (!state.importedWorkflowFile) return;
    const url = URL.createObjectURL(state.importedWorkflowFile);
    const link = document.createElement('a');
    link.href = url; link.download = state.importedWorkflowFile.name; link.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }
  if (event.target.closest('.analysis-open-comfy')) {
    if (!/^https?:\/\//i.test(state.comfyUrl)) { showMessage('ComfyUI URLを設定してください。', 'error'); return; }
    window.open(state.comfyUrl, '_blank', 'noopener,noreferrer');
  }
  if (event.target.closest('.analysis-open-diagnostics')) {
    $('#diagnostics-dialog').showModal();
    loadDiagnostics();
  }
});
refresh();
