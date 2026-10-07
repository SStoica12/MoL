'use strict';

const $ = (selector) => document.querySelector(selector);
const $$ = (selector) => [...document.querySelectorAll(selector)];
const routing = {
  layer: {
    tag: 'ONE DECISION FOR THE WHOLE IMAGE',
    title: 'One image. One shared mixture.',
    copy: 'Pool the patch features at each depth, score them against the text query, and select the top-k layers. Every patch uses the same layer selection and weights.',
    formula: 'routed image = Σ selected weight × layer features',
    note: 'A shared routing decision provides an image-level preference over visual depth.',
    colors: ['#749776', '#749776', '#749776', '#749776'],
  },
  patch: {
    tag: 'A SEPARATE CHOICE FOR EACH PATCH',
    title: 'Different regions. Different depths.',
    copy: 'Score every visual patch against the text query across encoder layers. Each region selects its own top-k layers, allowing local evidence to draw from different levels of abstraction.',
    formula: 'routed patch = Σ patch-specific weight × layer features',
    note: 'The number of output patch tokens stays unchanged.',
    colors: ['#749776', '#d69370', '#749776', '#668b9b'],
  },
  hybrid: {
    tag: 'GLOBAL + LOCAL + RESERVE',
    title: 'The detail and the bigger picture.',
    copy: 'Combine image-level and patch-level routing with a weighted product of experts. A learned gate uses their disagreement to blend in a stable reserve layer.',
    formula: 'routed patch = (1 − γ) · selected layers + γ · reserve',
    note: 'The reserve is the final or penultimate representation used by the original backbone.',
    colors: ['#749776', '#d69370', '#749776', '#668b9b'],
  },
};

function renderRouter(key) {
  const variant = routing[key];
  $$('[data-router]').forEach((button) => button.setAttribute('aria-pressed', String(button.dataset.router === key)));
  ['tag', 'title', 'copy', 'formula', 'note'].forEach((field) => $(`#router-${field}`).textContent = variant[field]);
  const layers = Array.from({ length: 6 }, (_, i) => {
    const selected = key === 'layer' ? [1, 3].includes(i) : [1, 3, 4].includes(i);
    const reserve = key === 'hybrid' && i === 5;
    return `<rect x="55" y="${20 + i * 29}" width="150" height="20" rx="3" fill="${reserve ? '#dbe4ca' : selected ? '#d8e6d4' : '#eceee6'}" stroke="${reserve ? '#173f38' : selected ? '#709577' : '#cbd3c2'}" ${reserve ? 'stroke-dasharray="3 2"' : ''}/><text x="65" y="${34 + i * 29}" font-size="9" fill="#4f6658">${reserve ? 'Reserve layer' : `Layer ${i + 1}`}</text>`;
  }).join('');
  const connections = variant.colors.map((color, i) => {
    const y = 30 + (key === 'layer' ? (i % 2 ? 3 : 1) : [1, 3, 1, 4][i]) * 29;
    return `<path d="M205 ${y}C260 ${y} 258 ${58 + i * 36} 306 ${58 + i * 36}" fill="none" stroke="${color}" stroke-width="1.7"/><rect x="307" y="${47 + i * 36}" width="38" height="23" rx="3" fill="${color}"/>`;
  }).join('');
  $('#route-illustration').innerHTML = `<svg viewBox="0 0 430 240" xmlns="http://www.w3.org/2000/svg"><text x="55" y="10" font-size="9" font-family="monospace" fill="#647068">VISION LAYERS</text><text x="292" y="25" font-size="9" font-family="monospace" fill="#647068">OUTPUT PATCHES</text>${layers}${connections}${key === 'hybrid' ? '<path d="M205 175Q260 205 326 200V179" fill="none" stroke="#173f38" stroke-dasharray="4 3"/><text x="229" y="219" fill="#647068" font-size="9">+ gated reserve</text>' : ''}<text x="55" y="222" font-size="10" fill="#647068">${key === 'layer' ? 'Shared layer weights' : 'Patch-specific layer weights'}</text></svg>`;
}
$$('[data-router]').forEach((button) => button.addEventListener('click', () => renderRouter(button.dataset.router)));
renderRouter('hybrid');

function renderResults(key) {
  const { metrics, groups } = window.MOL_RESULTS;
  const group = groups[key];
  const best = Math.max(...group.rows.map((row) => row.scores[0]));
  const winner = group.rows.find((row) => row.scores[0] === best);
  const delta = best - group.rows[0].scores[0];
  const colors = { Baseline: '#aab4a4', 'Interleaved-MoF': '#aab4a4', 'MoL layer': '#9bb49a', 'MoL patch': '#6e9981', 'MoL hybrid': '#1c6b59' };
  $('#results-chart').innerHTML = group.rows.map((row) => `<div class="chart-row ${row.scores[0] === best ? 'best' : ''}"><span class="bar-label">${row.method}${row.scores[0] === best ? '<small>BEST V* SCORE</small>' : ''}</span><div class="bar-track" aria-hidden="true"><div class="bar-fill" style="--score:${row.scores[0]}%;--bar-color:${colors[row.method]}"></div></div><span class="bar-number">${row.scores[0].toFixed(2)}</span></div>`).join('');
  $('#result-takeaway').innerHTML = `<strong>+${delta.toFixed(2)} percentage points</strong> on V* · ${winner.method} vs. ${group.rows[0].method.toLowerCase()} · ${group.label}`;
  $('#results-caption').textContent = `${group.label} · Accuracy (%) · Best displayed score in each column is bold`;
  $('#results-table thead').innerHTML = `<tr><th scope="col">Method</th>${metrics.map((metric) => `<th scope="col">${metric}</th>`).join('')}</tr>`;
  const maxima = metrics.map((_, i) => Math.max(...group.rows.map((row) => row.scores[i])));
  const variantMaxima = metrics.map((_, i) => Math.max(...group.rows.slice(1).map((row) => row.scores[i])));
  $('#results-table tbody').innerHTML = group.rows.map((row) => `<tr class="${row === winner ? 'best-row' : ''}"><th scope="row">${row.method}</th>${row.scores.map((score, i) => `<td class="${score === maxima[i] ? 'metric-best' : ''}">${score.toFixed(2)}</td>`).join('')}</tr>`).join('') + `<tr class="delta-row"><th scope="row">Best MoL − baseline</th>${variantMaxima.map((score, i) => { const difference = Number((score - group.rows[0].scores[i]).toFixed(2)); return `<td class="${difference < 0 ? 'negative' : 'positive'}">${difference >= 0 ? '+' : '−'}${Math.abs(difference).toFixed(2)}</td>`; }).join('')}</tr>`;
  $('#results-note').textContent = key === 'multi'
    ? 'Source: the manuscript’s multi-encoder table. Interleaved-MoF uses CLIP + DINOv2; MoL uses CLIP + DINOv2 w/ Txt with hybrid routing and concatenation/upsampling. This is a complete-system comparison, not a controlled routing-only ablation. Deltas are percentage points.'
    : 'Source: the manuscript’s main backbone comparison. Deltas are percentage points calculated from displayed scores; “Best MoL” may use a different routing variant for each metric. Results vary by benchmark and backbone.';
}
if (window.MOL_RESULTS) {
  renderResults('clip');
  $('#backbone').addEventListener('change', (event) => renderResults(event.target.value));
  $('#show-multi').addEventListener('click', () => {
    $('#backbone').value = 'multi';
    renderResults('multi');
    $('#backbone').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'center' });
    $('#backbone').focus({ preventScroll: true });
  });
}

$$('[data-sampling]').forEach((button) => button.addEventListener('click', () => {
  const clip = button.dataset.sampling === 'clip';
  $$('[data-sampling]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
  const path = `assets/${clip ? 'layer_sampling_by_category' : 'vstar_mmstar_gqa_layer_sampling_rf_background_dinov2'}.webp`;
  $('#sampling-image').src = path;
  $('#sampling-image').height = clip ? 958 : 943;
  $('#sampling-image').alt = `Measured ${clip ? 'CLIP' : 'DINOv2'} routing probabilities across layers for V*, MMStar, and GQA categories, with receptive-field backgrounds and top-k selection markers.`;
  $('#sampling-link').href = path;
  $('#sampling-expand').href = path;
}));

$$('[data-attention]').forEach((button) => button.addEventListener('click', () => {
  const clip = button.dataset.attention === 'clip';
  $$('[data-attention]').forEach((item) => item.setAttribute('aria-pressed', String(item === button)));
  const path = `assets/case_study_attention_map${clip ? '' : '_dinov2'}.webp`;
  $('#attention-image').src = path;
  $('#attention-image').height = clip ? 765 : 548;
  $('#attention-image').alt = `${clip ? 'CLIP' : 'DINOv2'} selected-layer self-attention examples for localized text, attributes, and spatial relationships, reproduced from the paper.`;
  $('#attention-link').href = path;
  $('#attention-caption').innerHTML = `<span>QUALITATIVE EVIDENCE · ${clip ? 'CLIP' : 'DINOv2'}</span> Self-attention maps from selected vision layers. These visualizations illustrate attention patterns; they do not by themselves establish a causal explanation.`;
}));

const dialog = $('#figure-dialog');
$$('[data-lightbox]').forEach((link) => link.addEventListener('click', (event) => {
  if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey || typeof dialog.showModal !== 'function') return;
  event.preventDefault();
  $('#dialog-image').src = link.href;
  $('#dialog-image').alt = link.querySelector('img')?.alt || link.closest('figure').querySelector('img').alt;
  dialog.showModal();
  document.body.classList.add('dialog-open');
}));
$('#close-figure').addEventListener('click', () => dialog.close());
dialog.addEventListener('click', (event) => { if (event.target === dialog) dialog.close(); });
dialog.addEventListener('close', () => document.body.classList.remove('dialog-open'));

$('#copy-citation').addEventListener('click', async () => {
  const button = $('#copy-citation');
  try {
    await navigator.clipboard.writeText($('#bibtex').textContent);
    button.textContent = 'Copied ✓';
    $('#copy-status').textContent = 'BibTeX copied to clipboard.';
    setTimeout(() => { button.textContent = 'Copy BibTeX ⧉'; }, 2500);
  } catch {
    const range = document.createRange();
    range.selectNodeContents($('#bibtex'));
    const selection = window.getSelection();
    selection.removeAllRanges();
    selection.addRange(range);
    $('#copy-status').textContent = 'Citation selected. Press Control+C or Command+C to copy.';
    button.textContent = 'Selected — press Ctrl/Cmd+C';
  }
});

if ('IntersectionObserver' in window) {
  const sections = $$('.site-header nav a').map((link) => document.querySelector(link.getAttribute('href')));
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      $$('.site-header nav a').forEach((link) => {
        if (link.hash === `#${entry.target.id}`) link.setAttribute('aria-current', 'location');
        else link.removeAttribute('aria-current');
      });
    });
  }, { rootMargin: '-15% 0px -65% 0px', threshold: 0 });
  sections.forEach((section) => observer.observe(section));
  observer.observe($('.hero'));
}
