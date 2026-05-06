'use strict';

// ── Charge metadata ────────────────────────────────────────────────────────
const CHARGE_META = {
  light_dues:       { label: 'Light Dues',       icon: '☀' },
  vts:              { label: 'VTS',               icon: '📡' },
  pilotage:         { label: 'Pilotage',          icon: '🧭' },
  tug_assistance:   { label: 'Tug Assistance',    icon: '⚙' },
  port_dues:        { label: 'Port Dues',         icon: '⚓' },
  cargo_dues:       { label: 'Cargo Dues',        icon: '📦' },
  berth_dues:       { label: 'Berth Dues',        icon: '🏗' },
  running_of_lines: { label: 'Running of Lines',  icon: '🔗' },
};

const TYPE_LABELS = {
  // Chat agent events
  llm_call:    'LLM',
  tool_call:   'TOOL',
  tool_result: 'RESULT',
  response:    'REPLY',
  // Legacy SSE events (kept for compatibility)
  profile:  'PROFILE',
  rules:    'RULES',
  fetch:    'FETCH',
  calc:     'CALC',
  error:    'ERROR',
  complete: 'DONE',
};

// ── State ──────────────────────────────────────────────────────────────────
let sessionId   = initSessionId();
let debugOpen   = false;
let stepCount   = 0;
let isStreaming  = false;

// ── Session management ─────────────────────────────────────────────────────
function initSessionId() {
  let id = localStorage.getItem('tariff_session_id');
  if (!id) {
    id = makeUUID();
    localStorage.setItem('tariff_session_id', id);
  }
  return id;
}

function makeUUID() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, c => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

function newChat() {
  // Generate a fresh session (server forgets old history)
  sessionId = makeUUID();
  localStorage.setItem('tariff_session_id', sessionId);

  // Clear the messages list and show welcome again
  const inner = document.getElementById('messagesInner');
  // Remove all children except the welcome state
  [...inner.children].forEach(child => {
    if (child.id !== 'welcomeState') child.remove();
  });
  document.getElementById('welcomeState').classList.remove('hidden');

  clearDebug(null);
  document.getElementById('chatInput').focus();
}

// ── Debug panel ────────────────────────────────────────────────────────────
function toggleDebug() {
  debugOpen = !debugOpen;
  document.getElementById('debugPanel').classList.toggle('open', debugOpen);
  document.getElementById('layout').classList.toggle('debug-open', debugOpen);
  document.getElementById('debugToggle').classList.toggle('active', debugOpen);
}

function clearDebug(e) {
  if (e) e.stopPropagation();
  document.getElementById('debugSteps').innerHTML = '';
  document.getElementById('debugEmpty').classList.remove('hidden');
  stepCount = 0;
  document.getElementById('debugStepCount').textContent = '0 steps';
}

function addDebugStep(evt) {
  const container = document.getElementById('debugSteps');
  document.getElementById('debugEmpty').classList.add('hidden');

  stepCount++;
  document.getElementById('debugStepCount').textContent =
    `${stepCount} step${stepCount !== 1 ? 's' : ''}`;

  const type    = evt.type;
  const badge   = TYPE_LABELS[type] || type.toUpperCase();
  const ms      = evt.elapsed_ms != null ? `+${evt.elapsed_ms}ms` : '';
  const n       = stepCount;

  // Build detail payload (omit the display fields)
  const detail = {};
  for (const [k, v] of Object.entries(evt)) {
    if (!['type', 'step', 'description', 'elapsed_ms'].includes(k)) {
      detail[k] = v;
    }
  }
  const detailStr = JSON.stringify(detail, null, 2);

  const div = document.createElement('div');
  div.className = `debug-step type-${type}`;
  div.innerHTML = `
    <span class="debug-step-num">${String(n).padStart(2, '0')}</span>
    <span class="debug-step-badge">${badge}</span>
    <div class="debug-step-body">
      <div class="debug-step-name">${esc(evt.step || type)}</div>
      <div class="debug-step-desc">${esc(evt.description || '')}</div>
      <div class="debug-step-detail" id="dd-${n}">
        <pre>${esc(detailStr)}</pre>
      </div>
    </div>
    <span class="debug-step-time">${ms}</span>`;

  div.addEventListener('click', () => {
    div.querySelector('.debug-step-detail').classList.toggle('open');
  });

  container.appendChild(div);
  container.scrollTop = container.scrollHeight;
}

// ── Suggestion chips ───────────────────────────────────────────────────────
function useSuggestion(btn) {
  const text  = btn.textContent.trim();
  const input = document.getElementById('chatInput');
  input.value = text;
  // Trigger resize + enable send button
  input.dispatchEvent(new Event('input'));
  sendMessage();
}

// ── Input initialisation ───────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  const input   = document.getElementById('chatInput');
  const sendBtn = document.getElementById('sendBtn');

  // Auto-grow textarea
  input.addEventListener('input', () => {
    input.style.height = 'auto';
    input.style.height = Math.min(input.scrollHeight, 160) + 'px';
    sendBtn.disabled = !input.value.trim() || isStreaming;
  });

  // Enter to send, Shift+Enter for newline
  input.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      if (!sendBtn.disabled) sendMessage();
    }
  });

  input.focus();
});

// ── Send a message ─────────────────────────────────────────────────────────
async function sendMessage() {
  const input = document.getElementById('chatInput');
  const text  = input.value.trim();
  if (!text || isStreaming) return;

  isStreaming = true;
  document.getElementById('sendBtn').disabled = true;

  // Hide welcome, add user bubble
  document.getElementById('welcomeState').classList.add('hidden');
  appendUserBubble(text);

  // Reset input
  input.value = '';
  input.style.height = 'auto';

  // Show typing indicator
  const typingEl = appendTypingIndicator();

  // Auto-open debug panel on first use
  if (!debugOpen) toggleDebug();

  try {
    const res = await fetch('/chat', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ session_id: sessionId, message: text }),
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: 'Request failed' }));
      throw new Error(err.detail || `HTTP ${res.status}`);
    }

    const reader  = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer    = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // keep the incomplete last line

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const evt = JSON.parse(line.slice(6));
            handleEvent(evt, typingEl);
          } catch (_) { /* skip malformed JSON */ }
        }
      }
    }

  } catch (err) {
    typingEl.remove();
    appendSystemMsg('⚠ ' + err.message);
  } finally {
    isStreaming = false;
    document.getElementById('sendBtn').disabled =
      !document.getElementById('chatInput').value.trim();
    document.getElementById('chatInput').focus();
  }
}

// ── Event routing ──────────────────────────────────────────────────────────
function handleEvent(evt, typingEl) {
  // Every event goes to the debug panel
  addDebugStep(evt);

  // The 'response' event carries the final reply and optional calc data
  if (evt.type === 'response') {
    typingEl.remove();
    appendAgentMessage(evt.content || '', evt.calc_data || null);
  }
}

// ── Message rendering ──────────────────────────────────────────────────────
function getInner() {
  return document.getElementById('messagesInner');
}

function appendUserBubble(text) {
  const group = document.createElement('div');
  group.className = 'message-group user';
  group.innerHTML = `
    <div class="message-row">
      <div class="bubble user-bubble">${esc(text)}</div>
    </div>`;
  getInner().appendChild(group);
  scrollBottom();
}

function appendAgentMessage(content, calcData) {
  const group = document.createElement('div');
  group.className = 'message-group agent';

  let inner = `
    <div class="message-row">
      <div class="bubble agent-bubble">${renderMarkdown(content)}</div>
    </div>`;

  if (calcData) {
    inner += buildCalcCard(calcData);
  }

  group.innerHTML = inner;
  getInner().appendChild(group);
  scrollBottom();
}

function appendTypingIndicator() {
  const group = document.createElement('div');
  group.className = 'message-group agent';
  group.innerHTML = `
    <div class="message-row">
      <div class="typing-bubble">
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
        <span class="typing-dot"></span>
      </div>
    </div>`;
  getInner().appendChild(group);
  scrollBottom();
  return group;
}

function appendSystemMsg(text) {
  const div = document.createElement('div');
  div.className = 'system-msg';
  div.textContent = text;
  getInner().appendChild(div);
  scrollBottom();
}

// ── Calculation card builder ───────────────────────────────────────────────
function buildCalcCard(data) {
  const items  = data.line_items || [];
  const errors = data.errors     || [];
  const total  = data.total_zar  || 0;
  const gt     = data.gt ? Number(data.gt).toLocaleString('en-ZA') : '';
  const port   = data.port
    ? data.port.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
    : '';

  const itemsHtml = items.map(item => {
    const meta = CHARGE_META[item.charge_type] || { label: item.charge_type, icon: '•' };
    return `
      <div class="calc-item">
        <span class="calc-item-icon">${meta.icon}</span>
        <div class="calc-item-body">
          <span class="calc-item-name">${meta.label}</span>
          <span class="calc-item-formula">${esc(item.formula || '')}</span>
        </div>
        <span class="calc-item-amount">${formatZAR(item.charge_zar)}</span>
      </div>`;
  }).join('');

  const errHtml = errors.length > 0 ? `
    <div class="calc-card-errors">
      ${errors.map(e => `<div class="calc-error-item">⚠ <strong>${esc(e.charge_type)}</strong>: ${esc(e.error)}</div>`).join('')}
    </div>` : '';

  const sub = [
    items.length + ' charge' + (items.length !== 1 ? 's' : ''),
    gt ? gt + ' GT' : '',
    port || '',
    '2024/25 Tariff',
  ].filter(Boolean).join(' · ');

  return `
    <div class="calc-card">
      <div class="calc-card-total">
        <div class="calc-total-label">Total Port Dues</div>
        <div class="calc-total-amount">${formatZAR(total)}</div>
        <div class="calc-total-sub">${sub}</div>
      </div>
      <div class="calc-card-divider"></div>
      <div class="calc-card-body">${itemsHtml}</div>
      ${errHtml}
    </div>`;
}

// ── Minimal Markdown renderer ──────────────────────────────────────────────
// Escapes HTML first, then selectively replaces markdown tokens with safe HTML.
function renderMarkdown(text) {
  if (!text) return '';
  return esc(text)
    // Bold: **text** or __text__
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/__(.+?)__/g, '<strong>$1</strong>')
    // Italic: *text* or _text_
    .replace(/\*([^*\n]+?)\*/g, '<em>$1</em>')
    // Inline code: `code`
    .replace(/`([^`]+?)`/g, '<code>$1</code>')
    // Line breaks
    .replace(/\n/g, '<br>');
}

// ── Helpers ────────────────────────────────────────────────────────────────
function scrollBottom() {
  const area = document.getElementById('messagesArea');
  // Use requestAnimationFrame to scroll after DOM paint
  requestAnimationFrame(() => { area.scrollTop = area.scrollHeight; });
}

function formatZAR(n) {
  return 'R ' + Number(n).toLocaleString('en-ZA', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function esc(str) {
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}
