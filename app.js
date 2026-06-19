'use strict';
/* ================================================================
   app.js  –  AI Fragment Reassembly Engine  (fixed build)
   Bugs fixed:
   1. TF-IDF → Character trigram TF-IDF for non-zero cross-sentence sim
   2. Heatmap drawn after card is shown (fixes 200×200 canvas bug)
   3. Parenthesis syntax fix in drawHeatmap cell calculation
   4. Compare-card match logic correctly checks position vs ground truth
   5. Added Google Fonts (Inter) preconnect already in HTML
   ================================================================ */

const DEMO=[
  "The study began by identifying the primary sources of pollution in the watershed.",
  "Water samples were collected from twelve different locations along the river.",
  "Each sample was preprocessed to remove outliers and normalise chemical readings.",
  "Machine learning classifiers, including Random Forest and SVM, were then applied.",
  "Feature engineering reduced dimensionality while retaining 97% of the variance.",
  "The team concluded that ensemble methods outperform any single classifier.",
  "These findings suggest AI-driven monitoring can serve as an early-warning system.",
  "The result was a significant improvement in water quality across all tested sites.",
];

/* ── Character trigram tokeniser ────────────────────────────────
   Uses overlapping char 3-grams instead of words.
   Captures morphological similarity even when sentences share no
   full words — dramatically improves cosine sim on short texts.   */
function charTrigrams(text) {
  const t = text.toLowerCase().replace(/[^a-z0-9 ]/g,' ').replace(/\s+/g,' ').trim();
  const ng = [];
  for (let i = 0; i < t.length - 2; i++) ng.push(t.substring(i, i + 3));
  return ng;
}

function buildTFIDF(sentences) {
  const N = sentences.length;
  const td = sentences.map(charTrigrams);
  const df = {};
  td.forEach(toks => new Set(toks).forEach(t => { df[t] = (df[t] || 0) + 1; }));
  const vocab = Object.keys(df), V = vocab.length, idf = {};
  vocab.forEach(t => { idf[t] = Math.log((N + 1) / (df[t] + 1)) + 1; });
  return td.map(toks => {
    const tf = {};
    toks.forEach(t => { tf[t] = (tf[t] || 0) + 1; });
    const v = new Float32Array(V);
    vocab.forEach((t, i) => { v[i] = ((tf[t] || 0) / Math.max(toks.length, 1)) * idf[t]; });
    const nm = Math.sqrt(v.reduce((s, x) => s + x * x, 0)) || 1;
    for (let i = 0; i < V; i++) v[i] /= nm;
    return v;
  });
}

function cos(a, b) {
  let d = 0;
  for (let i = 0; i < a.length; i++) d += a[i] * b[i];
  return Math.max(0, Math.min(1, d));
}

function simMatrix(vecs) {
  const n = vecs.length;
  const S = Array.from({ length: n }, () => new Float32Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i; j < n; j++) {
      const s = cos(vecs[i], vecs[j]);
      S[i][j] = s; S[j][i] = s;
    }
  }
  return S;
}

/* ── Ordering algorithms ──────────────────────────────────────── */
function greedy(sim) {
  const n = sim.length;
  const means = sim.map((r, i) => {
    let s = 0;
    for (let j = 0; j < n; j++) if (j !== i) s += r[j];
    return s / (n - 1);
  });
  let start = means.indexOf(Math.min(...means));
  const vis = new Array(n).fill(false), order = [start];
  vis[start] = true;
  for (let step = 1; step < n; step++) {
    const cur = order[order.length - 1];
    let best = -1, bs = -Infinity;
    for (let j = 0; j < n; j++) if (!vis[j] && sim[cur][j] > bs) { bs = sim[cur][j]; best = j; }
    order.push(best); vis[best] = true;
  }
  return order;
}

function beamSearch(sim, w) {
  const n = sim.length;
  const means = sim.map((r, i) => {
    let s = 0;
    for (let j = 0; j < n; j++) if (j !== i) s += r[j];
    return { i, m: s / (n - 1) };
  }).sort((a, b) => a.m - b.m);
  let beams = means.slice(0, w).map(({ i }) => ({ order: [i], score: 0 }));
  for (let step = 1; step < n; step++) {
    const cands = [];
    beams.forEach(b => {
      const vis = new Set(b.order), cur = b.order[b.order.length - 1];
      for (let j = 0; j < n; j++) {
        if (!vis.has(j)) cands.push({ order: [...b.order, j], score: b.score + sim[cur][j] });
      }
    });
    cands.sort((a, b) => b.score - a.score);
    beams = cands.slice(0, w);
  }
  return beams[0].order;
}

function chainScore(sim, order) {
  if (order.length <= 1) return 1;
  let t = 0;
  for (let i = 0; i < order.length - 1; i++) t += sim[order[i]][order[i + 1]];
  return t / (order.length - 1);
}

/* ── Kendall Tau ─────────────────────────────────────────────── */
function kendallTau(truth, predicted) {
  const n = truth.length;
  if (n <= 1) return 1;
  // Build rank lookup for truth array (value -> position)
  const truthRank = {};
  truth.forEach((v, i) => { truthRank[v] = i; });
  let c = 0, d = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const ti = truthRank[predicted[i]] ?? i;
      const tj = truthRank[predicted[j]] ?? j;
      if (ti < tj) c++;
      else if (ti > tj) d++;
    }
  }
  const den = n * (n - 1) / 2;
  return den === 0 ? 1 : (c - d) / den;
}

/* ── Main reassembly pipeline ───────────────────────────────── */
function reassemble(frags, bw = 3) {
  if (!frags.length) return { ordered: [], indices: [], confidence: 0, sim: [] };
  if (frags.length === 1) return { ordered: [...frags], indices: [0], confidence: 1, sim: [[1]] };
  const vecs = buildTFIDF(frags);
  const sim = simMatrix(vecs);
  const order = (bw > 1 && frags.length <= 14) ? beamSearch(sim, bw) : greedy(sim);
  return { ordered: order.map(i => frags[i]), indices: order, confidence: chainScore(sim, order), sim };
}

/* ── Heatmap canvas renderer ────────────────────────────────── */
function drawHeatmap(canvas, sim) {
  const n = sim.length;
  const pad = 4;
  // FIX: use canvas.offsetParent or fixed calculation instead of clientWidth
  // when parentElement might be hidden. We use a minimum sensible size.
  const availW = canvas.parentElement.offsetWidth || canvas.parentElement.clientWidth || 400;
  const cell = Math.max(28, Math.min(64, Math.floor((availW - pad * 2 - 16) / n)));
  canvas.width  = cell * n + pad * 2;
  canvas.height = cell * n + pad * 2;
  canvas.style.width  = canvas.width  + 'px';
  canvas.style.height = canvas.height + 'px';

  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = sim[i][j];
      // Indigo → Cyan gradient
      const r = Math.round(99  + (6   - 99)  * v);
      const g = Math.round(102 + (182 - 102) * v);
      const b = Math.round(241 + (212 - 241) * v);
      ctx.fillStyle = 'rgb(' + r + ',' + g + ',' + b + ')';
      ctx.globalAlpha = 0.25 + v * 0.75;
      ctx.fillRect(pad + j * cell, pad + i * cell, cell - 1, cell - 1);
    }
  }
  ctx.globalAlpha = 1;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.font = 'bold ' + Math.max(9, Math.floor(cell * 0.32)) + 'px Inter,sans-serif';
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      const v = sim[i][j];
      ctx.fillStyle = v > 0.5 ? 'rgba(10,10,15,0.85)' : 'rgba(240,240,248,0.75)';
      ctx.fillText(v.toFixed(2), pad + j * cell + cell / 2, pad + i * cell + cell / 2);
    }
  }
}

/* ── Seeded shuffle ─────────────────────────────────────────── */
function seededRNG(seed) {
  let s = seed % 2147483647; if (s <= 0) s += 2147483646;
  return () => { s = s * 16807 % 2147483647; return (s - 1) / 2147483646; };
}
function shuffle(arr, seed) {
  const a = [...arr], rng = seededRNG(seed);
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

/* ── DOM helpers ────────────────────────────────────────────── */
const $ = id => document.getElementById(id);
const show = (el, v) => { el.style.display = v ? '' : 'none'; };

function animNum(el, val, dec = 0, suf = '') {
  const dur = 700, t0 = performance.now();
  const tick = now => {
    const p = Math.min((now - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
    el.textContent = (val * e).toFixed(dec) + suf;
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function fItem(num, text) {
  const d = document.createElement('div');
  d.className = 'fragment-item';
  d.style.animationDelay = (num * 0.04) + 's';
  d.innerHTML = '<div class="fragment-num">' + num + '</div><div class="fragment-text">' + esc(text) + '</div>';
  return d;
}

function cItem(num, text, match) {
  const d = document.createElement('div');
  d.className = 'compare-item' + (match ? ' match' : ' mismatch');
  d.innerHTML = '<span class="citem-num">' + num + '</span><span class="citem-text">' + esc(text) + '</span>';
  return d;
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

/* ── Fragment count update ──────────────────────────────────── */
function updateCount() {
  const n = $('fragmentInput').value.split('\n').filter(l => l.trim()).length;
  $('fragmentCount').textContent = n + ' fragment' + (n !== 1 ? 's' : '') + ' detected';
  $('reassembleBtn').disabled = n < 2;
}

function loadDemo() {
  $('fragmentInput').value = DEMO.join('\n');
  updateCount();
}

function clearAll() {
  $('fragmentInput').value = '';
  updateCount();
  ['outputCard','metricsCard','compareCard','statsCard'].forEach(id => show($(id), false));
}

/* ── Core run function ──────────────────────────────────────── */
async function run() {
  const raw = $('fragmentInput').value.split('\n').filter(l => l.trim());
  if (raw.length < 2) return;

  const bw  = parseInt($('beamWidth').value);
  const seed = parseInt($('randomSeed').value) || 42;
  const showM = $('showMetrics').checked;
  const doSh  = $('autoShuffle').checked;

  show($('loadingOverlay'), true);
  await new Promise(r => setTimeout(r, 60));

  try {
    const orig = [...raw];
    const inp  = doSh ? shuffle(raw, seed) : [...raw];

    const t0 = performance.now();
    const { ordered, indices, confidence, sim } = reassemble(inp, bw);
    const elapsed = performance.now() - t0;

    /* Stats sidebar */
    show($('statsCard'), true);
    $('statFragments').textContent = ordered.length;
    $('statTime').textContent = elapsed.toFixed(1) + ' ms';
    $('statConf').textContent = confidence.toFixed(3);

    /* Output list */
    const ol = $('outputList');
    ol.innerHTML = '';
    ordered.forEach((t, i) => ol.appendChild(fItem(i + 1, t)));
    show($('outputCard'), true);

    /* Evaluation metrics */
    if (showM) {
      /* Build ground-truth lookup: text -> original position in orig */
      const oMap = Object.fromEntries(orig.map((t, i) => [t, i]));

      /* predicted[i] = which original position sits at output slot i */
      const predicted = ordered.map(t => oMap[t] ?? -1);

      /* ground truth: output slot i should contain original item i */
      const truth = Array.from({ length: orig.length }, (_, i) => i);

      const correct = predicted.filter((p, i) => p === truth[i]).length;
      const acc     = correct / orig.length;
      const tau     = kendallTau(truth, predicted);
      const perf    = correct === orig.length;

      /* FIX: Show metricsCard BEFORE drawing heatmap so clientWidth > 0 */
      show($('metricsCard'), true);
      // Allow DOM to reflow before reading clientWidth
      await new Promise(r => requestAnimationFrame(() => requestAnimationFrame(r)));

      $('matchVal').textContent = perf ? 'Yes' : 'No';
      $('matchVal').style.color = perf ? 'var(--success)' : 'var(--danger)';
      animNum($('accuracyVal'), acc * 100, 1, '%');
      animNum($('tauVal'), tau, 4);

      /* Draw heatmap AFTER card is visible */
      if (sim && sim.length) drawHeatmap($('heatmapCanvas'), sim);

      /* Before vs After comparison */
      const sl = $('shuffledList'), rl = $('orderedList');
      sl.innerHTML = ''; rl.innerHTML = '';
      inp.forEach((t, i) => sl.appendChild(cItem(i + 1, t, false)));
      ordered.forEach((t, i) => {
        const origPos = oMap[t] ?? -1;
        rl.appendChild(cItem(i + 1, t, origPos === i));
      });
      show($('compareCard'), true);
    }

    $('outputCard').scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } finally {
    show($('loadingOverlay'), false);
  }
}

/* ── Copy output ────────────────────────────────────────────── */
function copyOut() {
  const items = document.querySelectorAll('#outputList .fragment-text');
  const txt = Array.from(items).map((el, i) => (i + 1) + '. ' + el.textContent).join('\n');
  navigator.clipboard.writeText(txt).then(() => {
    const b = $('copyBtn'), o = b.textContent;
    b.textContent = 'Copied!'; b.style.color = 'var(--success)';
    setTimeout(() => { b.textContent = o; b.style.color = ''; }, 2000);
  }).catch(() => {
    /* Fallback for browsers without clipboard API */
    const ta = document.createElement('textarea');
    ta.value = txt; document.body.appendChild(ta); ta.select();
    document.execCommand('copy'); document.body.removeChild(ta);
    const b = $('copyBtn'); b.textContent = 'Copied!';
    setTimeout(() => { b.textContent = 'Copy'; }, 2000);
  });
}

/* ── Bootstrap ──────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  loadDemo();
  $('fragmentInput').addEventListener('input', updateCount);
  $('loadDemo').addEventListener('click', loadDemo);
  $('clearBtn').addEventListener('click', clearAll);
  $('reassembleBtn').addEventListener('click', run);
  $('copyBtn').addEventListener('click', copyOut);
  $('beamWidth').addEventListener('input', () => {
    $('beamHint').textContent = $('beamWidth').value;
  });
});
