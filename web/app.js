/* ============================================================
   CLS Finder — app.js
   Pyodide initialization + UI state + rendering
   ============================================================ */

'use strict';

// ─── Global State ─────────────────────────────────────────────────────────────
let pyReady = false;
let focusedSymCell = null;  // last focused symbolic matrix input
let libraryModels = []; // Cache library presets
let bandPreviewTimer = null; // debounce timer for band preview
let bandPreviewAbort = null; // AbortController for band preview
let simplifiedMatrix = null;
let simplifyMatrixTimer = null;

// Reserved words that should NOT be detected as parameters
const RESERVED_SYMBOLS = new Set([
  'kx', 'ky', 'kz', 'I', 'pi', 'e',
  'exp', 'cos', 'sin', 'tan', 'sqrt', 'abs', 'ln', 'log',
  'conj', 'conjugate', 're', 'im',
  'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'mu', 'sigma', 'lambda', 'omega',
  'theta', 'phi', 'psi', 'nu', 'tau', 'eta', 'zeta', 'xi', 'rho', 'kappa'
]);
// Greek letters that should be treated as parameters (not reserved math functions)
const GREEK_PARAM_NAMES = new Set([
  'alpha', 'beta', 'gamma', 'delta', 'epsilon', 'mu', 'sigma', 'lambda', 'omega',
  'theta', 'phi', 'psi', 'nu', 'tau', 'eta', 'zeta', 'xi', 'rho', 'kappa'
]);
const MATH_FUNCTIONS = new Set([
  'exp', 'cos', 'sin', 'tan', 'sqrt', 'abs', 'ln', 'log', 'conj', 'conjugate'
]);

const state = {
  dimension: 2,
  primitiveVectors: [[1.0, 0.0], [0.0, 1.0]],
  orbitals: [
    { label: 'A', position: [0.0, 0.0] },
    { label: 'B', position: [0.5, 0.0] }
  ],
  hamiltonianMode: 'hopping',   // 'hopping' | 'matrix'
  hoppings: [],
  symbolicMatrix: [],           // Q×Q array of strings
  kGrid: 30,
  flatTol: 1e-5,
  parameters: {},               // name → value map for Hamiltonian params
  parameterRanges: {},          // name → { min, max, step }
  kPathStr: '',                 // Custom k-path string
  kPointsOverride: {},          // Custom high-symmetry points overrides
  plotN: 60,                    // 2D grid resolution for BZ and Projector plots
};

// ─── API Client (native Python server) ───────────────────────────────────────
async function apiFetch(endpoint, body = null) {
  const opts = body
    ? { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) }
    : { method: 'GET' };
  const resp = await fetch('/api/' + endpoint, opts);
  if (!resp.ok) {
    const txt = await resp.text().catch(() => resp.statusText);
    throw new Error(`API ${resp.status}: ${txt}`);
  }
  return resp.json();
}

// ─── Server Initialization ────────────────────────────────────────────────────
async function initApiClient() {
  const overlay  = document.getElementById('loading-overlay');
  const progFill = document.getElementById('prog-fill');
  const statusEl = document.getElementById('loading-status');
  const badge    = document.getElementById('py-status');

  function setProgress(pct, msg) {
    progFill.style.width = pct + '%';
    statusEl.textContent = msg;
  }

  setProgress(30, '로컬 서버에 연결 중...');

  try {
    setProgress(70, '모델 목록 로드 중...');
    const models = await apiFetch('models');

    setProgress(100, '완료!');
    setTimeout(() => {
      pyReady = true;
      badge.textContent = '준비 완료';
      badge.className   = 'status-badge ready';
      overlay.classList.add('hidden');

      document.getElementById('run-btn').disabled = false;
      document.getElementById('band-preview-btn').disabled = false;
      document.getElementById('load-preset-btn').disabled = false;

      loadPresetList(models);
      updateHamiltonianMatrixPreview();
    }, 200);
  } catch (err) {
    setProgress(100, '서버 연결 실패');
    statusEl.style.color = '#f06060';
    statusEl.textContent = '오류: 로컬 서버에 연결할 수 없습니다. start.bat으로 서버를 먼저 실행하세요.';
    badge.textContent = '오류';
    badge.className   = 'status-badge error';
    console.error('Server init failed:', err);
  }
}

// ─── Preset List ─────────────────────────────────────────────────────────────
function getUserModels() {
  try {
    const raw = localStorage.getItem('cls_user_models');
    return raw ? JSON.parse(raw) : [];
  } catch (_) {
    return [];
  }
}

function saveUserModels(models) {
  try {
    localStorage.setItem('cls_user_models', JSON.stringify(models));
  } catch (e) {
    console.error('Failed to save to localStorage', e);
  }
}

function loadPresetList(models = null) {
  if (!models) {
    models = libraryModels || [];
  } else {
    libraryModels = models;
  }
  const sel     = document.getElementById('preset-select');
  sel.innerHTML = '<option value="">— 모델 선택 —</option>';
  
  // 1. Add Library Presets Group
  const libGroup = document.createElement('optgroup');
  libGroup.label = '기본 라이브러리 프리셋';
  for (const m of models) {
    const opt      = document.createElement('option');
    opt.value      = m.id;
    opt.textContent = `${m.name}  (${m.dim}D, Q=${m.Q})`;
    opt.dataset.desc = m.desc;
    opt.dataset.isUser = 'false';
    libGroup.appendChild(opt);
  }
  sel.appendChild(libGroup);
  
  // 2. Add User Models Group
  const userModels = getUserModels();
  if (userModels.length > 0) {
    const userGroup = document.createElement('optgroup');
    userGroup.label = '저장된 사용자 모델';
    for (const m of userModels) {
      const opt      = document.createElement('option');
      opt.value      = m.id;
      opt.textContent = `${m.name}  (${m.dim}D, Q=${m.Q})`;
      opt.dataset.desc = m.desc || '사용자 저장 모델';
      opt.dataset.isUser = 'true';
      userGroup.appendChild(opt);
    }
    sel.appendChild(userGroup);
  }
}

async function applyPreset(modelId) {
  console.log('[CLS] Loading preset:', modelId);
  const userModels = getUserModels();
  const userModel = userModels.find(m => m.id === modelId);
  
  let spec;
  if (userModel) {
    spec = userModel.spec;
  } else {
    spec = await apiFetch('model_spec/' + encodeURIComponent(modelId));
  }
  
  if (spec.error) { alert('모델 로드 실패: ' + spec.error); return; }
  console.log('[CLS] Preset spec:', JSON.stringify(spec.lattice || spec, null, 2));
  applySpecToState(spec);
  rebuildLatticeUI();
  rebuildHamiltonianEditor();

  // Show detected lattice type
  const lat = spec.lattice || spec;
  const descEl = document.getElementById('preset-desc');
  const opt = document.getElementById('preset-select').selectedOptions[0];
  const baseDesc = opt?.dataset.desc || '';
  const typeStr = detectLatticeType(lat);
  descEl.textContent = baseDesc + (typeStr ? '  ▸ ' + typeStr : '');
}

function detectLatticeType(lat) {
  const d = lat.dimension;
  const Q = lat.orbitals ? lat.orbitals.length : 0;
  
  // Count distinct sublattices (by position proximity)
  const sublattices = [];
  if (lat.orbitals) {
    lat.orbitals.forEach(orb => {
      const pos = orb.position;
      const match = sublattices.find(s =>
        s.pos.length === pos.length && s.pos.every((v, i) => Math.abs(v - pos[i]) < 1e-6)
      );
      if (match) {
        match.count++;
        match.labels.push(orb.label);
      } else {
        sublattices.push({ pos: [...pos], count: 1, labels: [orb.label] });
      }
    });
  }
  const nSub = sublattices.length;
  const isMultiOrb = sublattices.some(s => s.count > 1);

  let bravaisStr = '';
  if (d === 1) {
    bravaisStr = '1D 사슬';
  } else if (d === 3) {
    bravaisStr = '3D 격자';
  } else {
    const v1 = lat.primitive_vectors[0], v2 = lat.primitive_vectors[1];
    const l1 = Math.sqrt(v1.reduce((s,x) => s+x*x, 0));
    const l2 = Math.sqrt(v2.reduce((s,x) => s+x*x, 0));
    const dot = v1.reduce((s,x,i) => s+x*v2[i], 0);
    const ang = Math.acos(Math.max(-1, Math.min(1, dot/(l1*l2)))) * 180 / Math.PI;
    if (Math.abs(l1-l2) < 0.01 && (Math.abs(ang-60) < 1 || Math.abs(ang-120) < 1))
      bravaisStr = '육각격자 (Hexagonal) — Γ→M→K→Γ';
    else if (Math.abs(l1-l2) < 0.01 && Math.abs(ang-90) < 1)
      bravaisStr = '정방격자 (Square) — Γ→X→M→Γ';
    else if (Math.abs(ang-90) < 1)
      bravaisStr = '직사각격자 (Rectangular) — Γ→X→S→Y→Γ';
    else
      bravaisStr = '사각격자 (Oblique) — Γ→X→M→Y→Γ';
  }

  let subInfo = `서브라티스 ${nSub}개`;
  if (isMultiOrb) {
    const maxOrb = Math.max(...sublattices.map(s => s.count));
    subInfo += ` (다중 오비탈: 최대 ${maxOrb}개/사이트)`;
  }

  return `${bravaisStr} | ${subInfo} | Q=${Q}`;
}

function getDefaultKPath(dim, primitiveVectors) {
  if (dim === 1) return '-π - π';
  if (dim === 2) {
    // Detect hexagonal vs square vs rectangular
    const A = primitiveVectors;
    const v1 = A[0], v2 = A[1];
    if (!v1 || !v2) return 'Γ - X - M - Γ';
    const l1 = Math.hypot(v1[0], v1[1]);
    const l2 = Math.hypot(v2[0], v2[1]);
    const dot = v1[0]*v2[0] + v1[1]*v2[1];
    const ca = dot / (l1 * l2);
    const ang = Math.acos(Math.max(-1.0, Math.min(1.0, ca))) * (180.0 / Math.PI);
    const isHex = (Math.abs(ang - 60.0) < 1.0 || Math.abs(ang - 120.0) < 1.0) && Math.abs(l1 - l2) < 1e-3;
    const isSquare = Math.abs(l1 - l2) < 1e-3 && Math.abs(ang - 90.0) < 1.0;
    const isRect = !isSquare && Math.abs(ang - 90.0) < 1.0;
    if (isHex) return 'Γ - M - K - K\' - Γ';
    if (isSquare) return 'Γ - X - M - Γ';
    if (isRect) return 'Γ - X - S - Y - Γ';
    return 'Γ - X - M - Y - Γ';
  }
  // dim === 3
  const A = primitiveVectors;
  if (!A[0] || !A[1] || !A[2]) return 'Γ - X - M - R - Γ';
  const l1 = Math.hypot(A[0][0], A[0][1], A[0][2]);
  const l2 = Math.hypot(A[1][0], A[1][1], A[1][2]);
  const l3 = Math.hypot(A[2][0], A[2][1], A[2][2]);
  const ang_12 = Math.acos((A[0][0]*A[1][0] + A[0][1]*A[1][1] + A[0][2]*A[1][2]) / (l1 * l2)) * (180.0 / Math.PI);
  const ang_23 = Math.acos((A[1][0]*A[2][0] + A[1][1]*A[2][1] + A[1][2]*A[2][2]) / (l2 * l3)) * (180.0 / Math.PI);
  const ang_31 = Math.acos((A[2][0]*A[0][0] + A[2][1]*A[0][1] + A[2][2]*A[0][2]) / (l3 * l1)) * (180.0 / Math.PI);
  const isFCC = Math.abs(l1 - l2) < 1e-3 && Math.abs(l2 - l3) < 1e-3 &&
                Math.abs(ang_12 - 60.0) < 2.0 && Math.abs(ang_23 - 60.0) < 2.0 && Math.abs(ang_31 - 60.0) < 2.0;
  const isBCC = Math.abs(l1 - l2) < 1e-3 && Math.abs(l2 - l3) < 1e-3 &&
                Math.abs(ang_12 - 109.47) < 2.0 && Math.abs(ang_23 - 109.47) < 2.0 && Math.abs(ang_31 - 109.47) < 2.0;
  if (isFCC) return 'Γ - X - W - L - Γ';
  if (isBCC) return 'Γ - H - P - N - Γ';
  return 'Γ - X - M - R - Γ';
}

function getDefaultKPointsMap(dim, primitiveVectors) {
  if (dim === 1) {
    return {
      "Γ": [0.0],
      "X": [0.5],
      "π": [0.5],
      "-π": [-0.5]
    };
  }
  if (dim === 2) {
    const A = primitiveVectors;
    const v1 = A[0], v2 = A[1];
    if (!v1 || !v2) return { "Γ": [0.0, 0.0], "X": [0.5, 0.0], "M": [0.5, 0.5] };
    const l1 = Math.hypot(v1[0], v1[1]);
    const l2 = Math.hypot(v2[0], v2[1]);
    const dot = v1[0]*v2[0] + v1[1]*v2[1];
    const ca = dot / (l1 * l2);
    const ang = Math.acos(Math.max(-1.0, Math.min(1.0, ca))) * (180.0 / Math.PI);
    const isHex = (Math.abs(ang - 60.0) < 1.0 || Math.abs(ang - 120.0) < 1.0) && Math.abs(l1 - l2) < 1e-3;
    const isSquare = Math.abs(l1 - l2) < 1e-3 && Math.abs(ang - 90.0) < 1.0;
    
    if (isHex) {
      return {
        "Γ": [0.0, 0.0],
        "M": [0.5, 0.0],
        "K": [2/3, 1/3],
        "K'": [1/3, 2/3]
      };
    } else if (isSquare) {
      return {
        "Γ": [0.0, 0.0],
        "X": [0.5, 0.0],
        "M": [0.5, 0.5]
      };
    } else {
      return {
        "Γ": [0.0, 0.0],
        "X": [0.5, 0.0],
        "Y": [0.0, 0.5],
        "M": [0.5, 0.5],
        "S": [0.5, 0.5]
      };
    }
  }
  // dim === 3
  const A = primitiveVectors;
  if (!A[0] || !A[1] || !A[2]) return { "Γ": [0.0, 0.0, 0.0], "X": [0.5, 0.0, 0.0] };
  const l1 = Math.hypot(A[0][0], A[0][1], A[0][2]);
  const l2 = Math.hypot(A[1][0], A[1][1], A[1][2]);
  const l3 = Math.hypot(A[2][0], A[2][1], A[2][2]);
  const ang_12 = Math.acos((A[0][0]*A[1][0] + A[0][1]*A[1][1] + A[0][2]*A[1][2]) / (l1 * l2)) * (180.0 / Math.PI);
  const ang_23 = Math.acos((A[1][0]*A[2][0] + A[1][1]*A[2][1] + A[1][2]*A[2][2]) / (l2 * l3)) * (180.0 / Math.PI);
  const ang_31 = Math.acos((A[2][0]*A[0][0] + A[2][1]*A[0][1] + A[2][2]*A[0][2]) / (l3 * l1)) * (180.0 / Math.PI);
  const isFCC = Math.abs(l1 - l2) < 1e-3 && Math.abs(l2 - l3) < 1e-3 &&
                Math.abs(ang_12 - 60.0) < 2.0 && Math.abs(ang_23 - 60.0) < 2.0 && Math.abs(ang_31 - 60.0) < 2.0;
  const isBCC = Math.abs(l1 - l2) < 1e-3 && Math.abs(l2 - l3) < 1e-3 &&
                Math.abs(ang_12 - 109.47) < 2.0 && Math.abs(ang_23 - 109.47) < 2.0 && Math.abs(ang_31 - 109.47) < 2.0;
  if (isFCC) {
    return {
      "Γ": [0.0, 0.0, 0.0],
      "X": [0.5, 0.5, 0.0],
      "W": [0.5, 0.75, 0.25],
      "L": [0.5, 0.5, 0.5]
    };
  }
  if (isBCC) {
    return {
      "Γ": [0.0, 0.0, 0.0],
      "H": [0.5, -0.5, 0.5],
      "P": [0.25, 0.25, 0.25],
      "N": [0.0, 0.0, 0.5]
    };
  }
  return {
    "Γ": [0.0, 0.0, 0.0],
    "X": [0.5, 0.0, 0.0],
    "Y": [0.0, 0.5, 0.0],
    "Z": [0.0, 0.0, 0.5],
    "M": [0.5, 0.5, 0.0],
    "R": [0.5, 0.5, 0.5]
  };
}

function rebuildKPointsOverrideUI() {
  const container = document.getElementById('kpoints-override-list');
  if (!container) return;
  container.innerHTML = '';

  const dim = state.dimension;
  const entries = Object.entries(state.kPointsOverride);

  if (entries.length === 0) {
    container.innerHTML = '<div style="font-size:0.7rem;color:#889;text-align:center;padding:0.2rem 0;">정의된 고대칭점이 없습니다.</div>';
    return;
  }

  entries.forEach(([label, coords]) => {
    const row = document.createElement('div');
    row.style.cssText = 'display:flex;align-items:center;gap:0.3rem;background:#fff;padding:0.2rem;border-radius:3px;border:1px solid #e2e8f0;';

    // 1. Label Input
    const labelInp = document.createElement('input');
    labelInp.type = 'text';
    labelInp.value = label;
    labelInp.style.cssText = 'width:35px;font-size:0.75rem;font-weight:600;text-align:center;border:1px solid #cbd5e1;border-radius:2px;padding:1px 0;box-sizing:border-box;';
    labelInp.maxLength = 6;
    labelInp.addEventListener('change', e => {
      const newLabel = e.target.value.trim();
      if (!newLabel || newLabel === label) {
        e.target.value = label;
        return;
      }
      if (state.kPointsOverride[newLabel]) {
        alert('이미 존재하는 고대칭점 이름입니다.');
        e.target.value = label;
        return;
      }
      state.kPointsOverride[newLabel] = state.kPointsOverride[label];
      delete state.kPointsOverride[label];
      triggerAutoSave();
      debouncedBandPreview();
      rebuildKPointsOverrideUI();
    });

    row.appendChild(labelInp);

    const colon = document.createElement('span');
    colon.textContent = ':';
    colon.style.fontSize = '0.75rem';
    row.appendChild(colon);

    // 2. Coordinate Inputs (dim boxes)
    const coordInpsWrap = document.createElement('div');
    coordInpsWrap.style.cssText = 'display:flex;gap:0.2rem;flex:1;';

    coords.forEach((coordVal, ci) => {
      const inp = document.createElement('input');
      inp.type = 'number';
      inp.step = 'any';
      inp.value = Number(coordVal.toFixed(5));
      inp.style.cssText = 'flex:1;min-width:25px;font-size:0.7rem;padding:1px;border:1px solid #cbd5e1;border-radius:2px;text-align:right;box-sizing:border-box;';
      inp.addEventListener('change', e => {
        const val = parseFloat(e.target.value);
        if (isNaN(val)) {
          e.target.value = state.kPointsOverride[label][ci];
          return;
        }
        state.kPointsOverride[label][ci] = val;
        triggerAutoSave();
        debouncedBandPreview();
      });
      coordInpsWrap.appendChild(inp);
    });

    row.appendChild(coordInpsWrap);

    // 3. Remove Button
    const rmBtn = document.createElement('button');
    rmBtn.type = 'button';
    rmBtn.className = 'btn-remove';
    rmBtn.innerHTML = '×';
    rmBtn.style.cssText = 'padding:0 0.2rem;font-size:0.9rem;font-weight:bold;line-height:1;margin-left:0.1rem;';
    rmBtn.addEventListener('click', () => {
      delete state.kPointsOverride[label];
      triggerAutoSave();
      debouncedBandPreview();
      rebuildKPointsOverrideUI();
    });

    row.appendChild(rmBtn);
    container.appendChild(row);
  });
}

function applySpecToState(spec) {
  const lat = spec.lattice || spec;
  state.dimension        = lat.dimension;
  state.primitiveVectors = lat.primitive_vectors.map(v => [...v]);
  state.orbitals         = lat.orbitals.map(o => {
    const entry = {
      label:    o.label,
      position: [...o.position]
    };
    // Preserve sublattice hint if present
    if (o.sublattice !== undefined) entry.sublattice = o.sublattice;
    return entry;
  });

  if (spec.hoppings) {
    state.hamiltonianMode = 'hopping';
    state.hoppings        = spec.hoppings.map(h => ({...h, R: [...h.R]}));
    state.symbolicMatrix  = [];
  } else if (spec.H_symbolic) {
    state.hamiltonianMode = 'matrix';
    state.symbolicMatrix  = spec.H_symbolic.map(row => [...row]);
    state.hoppings        = [];
  }
  // Sync tab UI
  const mode = state.hamiltonianMode;
  document.querySelectorAll('.tab-btn').forEach(b =>
    b.classList.toggle('active', b.dataset.tab === mode));
  document.getElementById('tab-hopping')?.classList.toggle('hidden', mode !== 'hopping');
  document.getElementById('tab-matrix')?.classList.toggle('hidden', mode !== 'matrix');

  if (spec.options) {
    if (spec.options.k_grid) {
      const g = spec.options.k_grid;
      state.kGrid = Array.isArray(g) ? g[0] : g;
    }
    if (spec.options.flat_tol) state.flatTol = spec.options.flat_tol;
    if (spec.options.plot_n) state.plotN = spec.options.plot_n;
    if (spec.options.k_path_str) {
      state.kPathStr = spec.options.k_path_str;
    } else {
      state.kPathStr = getDefaultKPath(state.dimension, state.primitiveVectors);
    }
  } else {
    state.kPathStr = getDefaultKPath(state.dimension, state.primitiveVectors);
  }

  // Sync param inputs
  document.getElementById('param-kgrid').value = state.kGrid;
  document.getElementById('param-tol').value   = state.flatTol;
  document.getElementById('param-plotn').value = state.plotN || 60;
  const kpathInput = document.getElementById('param-kpath');
  if (kpathInput) {
    kpathInput.value = state.kPathStr;
  }

  // Restore Hamiltonian parameters
  if (spec.parameters) {
    state.parameters = { ...spec.parameters };
  } else {
    state.parameters = {};
  }
  if (spec.parameterRanges) {
    state.parameterRanges = {};
    for (const [k, v] of Object.entries(spec.parameterRanges)) {
      state.parameterRanges[k] = { ...v };
    }
  } else {
    state.parameterRanges = {};
  }
  if (spec.options && spec.options.k_points_override) {
    state.kPointsOverride = {};
    for (const [k, v] of Object.entries(spec.options.k_points_override)) {
      state.kPointsOverride[k] = [...v];
    }
  } else {
    state.kPointsOverride = getDefaultKPointsMap(state.dimension, state.primitiveVectors);
  }
  rebuildKPointsOverrideUI();
}

// ─── Build Spec JSON from UI ──────────────────────────────────────────────────
function buildSpec() {
  const spec = {
    lattice: {
      dimension: state.dimension,
      primitive_vectors: state.primitiveVectors,
      orbitals: state.orbitals
    },
    options: {
      k_grid:   Array(state.dimension).fill(state.kGrid),
      flat_tol: state.flatTol,
      k_path_str: state.kPathStr,
      k_points_override: state.kPointsOverride,
      plot_n: state.plotN || 60
    }
  };

  // Inject parameters
  if (Object.keys(state.parameters).length > 0) {
    spec.parameters = { ...state.parameters };
  }

  if (state.hamiltonianMode === 'hopping') {
    const cards = document.querySelectorAll('#hop-cards-ui .hop-card');
    const hops = [];
    cards.forEach(card => {
      const selI   = card.querySelector('.hop-sel-i');
      const selJ   = card.querySelector('.hop-sel-j');
      const rInps  = card.querySelectorAll('.hop-r-inp');
      const inpExpr = card.querySelector('.hop-expr-input');
      if (!selI || !selJ) return;
      const i   = parseInt(selI.value);
      const j   = parseInt(selJ.value);
      const R   = Array.from(rInps, inp => parseInt(inp.value) || 0);

      // Support expression strings for hopping amplitude
      const exprVal = inpExpr ? inpExpr.value.trim() : '0';
      let t;
      if (exprVal === '' || exprVal === '0') {
        t = 0;
      } else if (hasSymbolicParams(exprVal)) {
        // Contains symbolic parameters → send as string
        t = exprVal;
      } else {
        // Try to evaluate as number
        try {
          const numVal = evalMathExpr(exprVal);
          t = numVal;
        } catch (_) {
          // Might contain I (imaginary) or complex expressions → send as string
          t = exprVal;
        }
      }
      hops.push({ i, j, R, t });
    });
    spec.hoppings = hops;
  } else {
    const Q   = state.orbitals.length;
    const mat = [];
    for (let r = 0; r < Q; r++) {
      const row = [];
      for (let c = 0; c < Q; c++) {
        const el  = document.getElementById(`sym-${r}-${c}`);
        row.push(el ? (el.value || '0') : '0');
      }
      mat.push(row);
    }
    spec.H_symbolic = mat;
  }

  return spec;
}

// ─── Parameter Detection ──────────────────────────────────────────────────────
function extractParameters(exprStr) {
  if (!exprStr || exprStr === '0') return [];
  // Find all word tokens
  const tokens = exprStr.match(/\b[a-zA-Z_]\w*\b/g) || [];
  const params = new Set();
  for (const tok of tokens) {
    // Skip math reserved words and momentum variables
    if (MATH_FUNCTIONS.has(tok)) continue;
    if (tok === 'kx' || tok === 'ky' || tok === 'kz') continue;
    if (tok === 'I' || tok === 'pi' || tok === 'e') continue;
    // Greek letter names are parameters
    if (GREEK_PARAM_NAMES.has(tok)) {
      params.add(tok);
      continue;
    }
    // Other identifiers that aren't reserved
    if (!RESERVED_SYMBOLS.has(tok)) {
      params.add(tok);
    }
  }
  return [...params];
}

function hasSymbolicParams(exprStr) {
  return extractParameters(exprStr).length > 0;
}

function collectAllParameters() {
  const allParams = new Set();

  if (state.hamiltonianMode === 'hopping') {
    const cards = document.querySelectorAll('#hop-cards-ui .hop-card');
    cards.forEach(card => {
      const inpExpr = card.querySelector('.hop-expr-input');
      if (inpExpr) {
        extractParameters(inpExpr.value).forEach(p => allParams.add(p));
      }
    });
  } else {
    const Q = state.orbitals.length;
    for (let r = 0; r < Q; r++) {
      for (let c = 0; c < Q; c++) {
        const el = document.getElementById(`sym-${r}-${c}`);
        if (el) {
          extractParameters(el.value).forEach(p => allParams.add(p));
        }
      }
    }
  }

  return [...allParams].sort();
}

function rebuildParameterPanel() {
  const params = collectAllParameters();
  const container = document.getElementById('param-sliders-ui');
  const badge = document.getElementById('param-count-badge');
  if (!container) return;

  // Remove parameters that no longer exist in expressions
  for (const key of Object.keys(state.parameters)) {
    if (!params.includes(key)) {
      delete state.parameters[key];
      delete state.parameterRanges[key];
    }
  }

  // Add defaults for new parameters
  for (const p of params) {
    if (!(p in state.parameters)) {
      state.parameters[p] = 1.0;
    }
    if (!(p in state.parameterRanges)) {
      state.parameterRanges[p] = { min: -5, max: 5, step: 0.01 };
    }
  }

  if (params.length === 0) {
    container.innerHTML = '<div class="param-slider-empty">수식에 파라미터를 사용하면 여기에 슬라이더가 자동 생성됩니다.<br><code>예: t1, delta, mu</code></div>';
    if (badge) badge.style.display = 'none';
    return;
  }

  if (badge) {
    badge.textContent = `${params.length}개`;
    badge.style.display = 'inline';
  }

  container.innerHTML = '';
  for (const p of params) {
    const range = state.parameterRanges[p];
    const val = state.parameters[p];

    const row = document.createElement('div');
    row.className = 'param-slider-row';
    row.style.position = 'relative';

    // Name label (with Greek letter rendering)
    const nameEl = document.createElement('span');
    nameEl.className = 'param-name';
    if (GREEK_PARAM_NAMES.has(p)) {
      try { katex.render('\\' + p, nameEl, { throwOnError: false }); }
      catch (_) { nameEl.textContent = p; }
    } else {
      // Render subscripted names: t1 -> t_1
      const match = p.match(/^([a-zA-Z]+)(\d+)$/);
      if (match) {
        try { katex.render(`${match[1]}_{${match[2]}}`, nameEl, { throwOnError: false }); }
        catch (_) { nameEl.textContent = p; }
      } else {
        nameEl.textContent = p;
      }
    }

    // Slider
    const slider = document.createElement('input');
    slider.type = 'range';
    slider.className = 'param-slider';
    slider.min = range.min;
    slider.max = range.max;
    slider.step = range.step;
    slider.value = val;

    // Value input
    const valInput = document.createElement('input');
    valInput.type = 'number';
    valInput.className = 'param-value-input';
    valInput.step = range.step;
    valInput.value = parseFloat(val.toFixed(4));

    // Range config button
    const rangeBtn = document.createElement('button');
    rangeBtn.className = 'param-range-btn';
    rangeBtn.textContent = '⚙';
    rangeBtn.title = '범위 설정';

    // Sync slider ↔ input
    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      state.parameters[p] = v;
      valInput.value = parseFloat(v.toFixed(4));
      debouncedBandPreview();
    });

    valInput.addEventListener('change', () => {
      const v = parseFloat(valInput.value);
      if (!isNaN(v)) {
        state.parameters[p] = v;
        slider.value = v;
        debouncedBandPreview();
      }
    });

    // Range config popup
    rangeBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      // Remove existing popups
      document.querySelectorAll('.param-range-popup').forEach(pp => pp.remove());

      const popup = document.createElement('div');
      popup.className = 'param-range-popup';
      popup.innerHTML = `
        <label>최소값</label>
        <input type="number" class="range-min" value="${range.min}" step="0.1">
        <label>최대값</label>
        <input type="number" class="range-max" value="${range.max}" step="0.1">
        <label>단계</label>
        <input type="number" class="range-step" value="${range.step}" step="0.001">
        <div class="range-actions">
          <button class="btn btn-secondary btn-sm range-apply">적용</button>
        </div>
      `;
      popup.querySelector('.range-apply').addEventListener('click', () => {
        const newMin = parseFloat(popup.querySelector('.range-min').value);
        const newMax = parseFloat(popup.querySelector('.range-max').value);
        const newStep = parseFloat(popup.querySelector('.range-step').value);
        if (!isNaN(newMin) && !isNaN(newMax) && !isNaN(newStep) && newMin < newMax && newStep > 0) {
          state.parameterRanges[p] = { min: newMin, max: newMax, step: newStep };
          rebuildParameterPanel();
        }
        popup.remove();
      });

      // Close on outside click
      const closeHandler = (ev) => {
        if (!popup.contains(ev.target) && ev.target !== rangeBtn) {
          popup.remove();
          document.removeEventListener('click', closeHandler);
        }
      };
      setTimeout(() => document.addEventListener('click', closeHandler), 10);

      row.appendChild(popup);
    });

    row.appendChild(nameEl);
    row.appendChild(slider);
    row.appendChild(valInput);
    row.appendChild(rangeBtn);
    container.appendChild(row);
  }
}

// ─── Band Preview (Real-time) ────────────────────────────────────────────────
function debouncedBandPreview() {
  if (!document.getElementById('auto-band-refresh')?.checked) return;
  if (bandPreviewTimer) clearTimeout(bandPreviewTimer);
  bandPreviewTimer = setTimeout(() => runBandPreview(), 300);
}

async function runBandPreview() {
  if (!pyReady) return;

  const loading = document.getElementById('band-loading');
  const previewBtn = document.getElementById('band-preview-btn');
  const previewLabel = document.getElementById('band-preview-label');
  const liveBadge = document.getElementById('band-live-badge');

  // Show loading state
  if (loading) loading.classList.remove('hidden');
  if (previewBtn) previewBtn.disabled = true;
  if (previewLabel) previewLabel.textContent = '계산 중...';

  // Abort previous request
  if (bandPreviewAbort) bandPreviewAbort.abort();
  bandPreviewAbort = new AbortController();

  let spec;
  try { spec = buildSpec(); }
  catch (err) {
    if (loading) loading.classList.add('hidden');
    if (previewBtn) previewBtn.disabled = false;
    if (previewLabel) previewLabel.textContent = '밴드 미리보기';
    return;
  }

  try {
    const resp = await fetch('/api/band_data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spec }),
      signal: bandPreviewAbort.signal
    });

    if (!resp.ok) throw new Error(`API ${resp.status}`);
    const data = await resp.json();

    if (data.error) {
      console.error('[Band Preview]', data.error);
      document.getElementById('run-status').textContent = '밴드 계산 오류: ' + data.error;
    } else {
      renderBandPlot(data);
      if (data.reciprocal_vectors) {
        renderReciprocalSpaceInfo(data);
      } else {
        const el = document.getElementById('reciprocal-space-section');
        if (el) el.style.display = 'none';
      }
      // Show bands panel
      showPanel('bands');
      if (liveBadge) {
        liveBadge.style.display = 'inline';
        liveBadge.textContent = '실시간';
        liveBadge.className = 'status-badge ready';
      }
    }
  } catch (err) {
    if (err.name !== 'AbortError') {
      console.error('[Band Preview] Error:', err);
    }
  }

  if (loading) loading.classList.add('hidden');
  if (previewBtn) previewBtn.disabled = false;
  if (previewLabel) previewLabel.textContent = '밴드 미리보기';
}


// ─── Run Analysis ─────────────────────────────────────────────────────────────
async function runAnalysis() {
  if (!pyReady) return;
  const runBtn   = document.getElementById('run-btn');
  const runLabel = document.getElementById('run-label');
  const runStat  = document.getElementById('run-status');
  const badge    = document.getElementById('py-status');

  runBtn.disabled   = true;
  runLabel.textContent = '분석 중...';
  badge.textContent    = '실행 중';
  badge.className      = 'status-badge running';
  runStat.textContent  = '';

  clearResults();
  showPanel('process');
  addLogEntry({ name: '분석 시작', status: 'running', message: '계산 준비 중...' });

  let spec;
  try { spec = buildSpec(); }
  catch (err) {
    runStat.textContent = '입력 오류: ' + err.message;
    runBtn.disabled     = false;
    runLabel.textContent = '분석 실행';
    badge.textContent    = '준비 완료';
    badge.className      = 'status-badge ready';
    return;
  }

  try {
    const response = await fetch('/api/run_analysis_stream', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({ spec })
    });
    
    if (!response.ok) {
      const txt = await response.text().catch(() => response.statusText);
      throw new Error(`API ${response.status}: ${txt}`);
    }
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      
      // Keep the last partial line in the buffer
      buffer = lines.pop();
      
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const data = JSON.parse(line);
          if (data.type === 'log') {
            addLogEntry(data.step);
          } else if (data.type === 'result') {
            displayResults(data.result);
          }
        } catch (e) {
          console.error('Failed to parse line:', line, e);
        }
      }
    }
    
    // Process final buffer if any
    if (buffer.trim()) {
      try {
        const data = JSON.parse(buffer);
        if (data.type === 'log') {
          addLogEntry(data.step);
        } else if (data.type === 'result') {
          displayResults(data.result);
        }
      } catch (e) {
        console.error('Failed to parse trailing buffer:', buffer, e);
      }
    }

  } catch (err) {
    addLogEntry({ name: '실행 오류', status: 'error', message: err.message });
    console.error(err);
  }

  runBtn.disabled      = false;
  runLabel.textContent = '분석 실행';
  badge.textContent    = '준비 완료';
  badge.className      = 'status-badge ready';
}

// ─── Display Results ─────────────────────────────────────────────────────────
function displayResults(result) {
  // Process log
  const logEl = document.getElementById('process-log');
  logEl.innerHTML = '';
  if (result.steps) result.steps.forEach(addLogEntry);

  if (result.error) {
    addLogEntry({ name: '오류', status: 'error',
                  message: result.error,
                  data: { traceback: result.traceback } });
  }

  // Band structure
  if (result.band_plot) {
    renderBandPlot(result.band_plot);
    activateResultsTab('bands');
  }

  // CLS + Classification
  if (result.flat_bands && result.flat_bands.length > 0) {
    const labels = state.orbitals.map(o => o.label);
    renderCLSContent(result.flat_bands, labels);
    renderClassification(result.flat_bands, labels);
    activateResultsTab('bands');
  }

  // Nearly flat bands info (perturbation-broken flat bands)
  if (result.nearly_flat_bands && result.nearly_flat_bands.length > 0) {
    renderNearlyFlatInfo(result.nearly_flat_bands);
  } else {
    const el = document.getElementById('nearly-flat-section');
    if (el) el.style.display = 'none';
  }

  // Reciprocal space info
  if (result.reciprocal_space) {
    renderReciprocalSpaceInfo(result.reciprocal_space);
  } else {
    const el = document.getElementById('reciprocal-space-section');
    if (el) el.style.display = 'none';
  }

  // Show backend lattice classification in the info bar
  if (result.lattice_info) {
    const li = result.lattice_info;
    const bar = document.getElementById('lattice-info-bar');
    const content = document.getElementById('lattice-info-content');
    if (bar && content) {
      content.textContent = li.description_ko || li.description_en || '';
      bar.style.display = 'flex';
    }
  }

  // Auto-switch to bands tab if successful
  if (result.band_plot && !result.error) {
    showPanel('bands');
  }
}

function clearResults() {
  document.getElementById('process-log').innerHTML = '';
  document.getElementById('band-plot').innerHTML   = '';
  document.getElementById('cls-content').innerHTML = '';
  document.getElementById('classify-content').innerHTML = '';
  const el = document.getElementById('reciprocal-space-section');
  if (el) {
    el.innerHTML = '';
    el.style.display = 'none';
  }
}

// ─── Process Log ─────────────────────────────────────────────────────────────
const ICONS = {
  running: '⟳', success: '✓', warning: '⚠', error: '✗', info: 'ℹ'
};

function addLogEntry(step) {
  const logEl = document.getElementById('process-log');
  // Remove welcome message on first entry
  const welcome = logEl.querySelector('.welcome-msg');
  if (welcome) welcome.remove();

  // Find if there is an existing log-entry with the same name
  let existingDiv = null;
  const entries = logEl.querySelectorAll('.log-entry');
  for (const entry of entries) {
    const nameEl = entry.querySelector('.log-name');
    if (nameEl && nameEl.textContent === step.name) {
      existingDiv = entry;
      break;
    }
  }

  const hasDetail = step.data && Object.keys(step.data).length > 0;
  const detailStr = hasDetail ? JSON.stringify(step.data, null, 2) : '';

  const innerHTML = `
    <span class="log-icon">${ICONS[step.status] || '•'}</span>
    <div class="log-body">
      <div class="log-name">${escHtml(step.name)}</div>
      <div class="log-msg">${escHtml(step.message)}</div>
      ${hasDetail ? `
        <span class="log-detail-toggle" onclick="toggleDetail(this)">▸ 세부 정보 보기</span>
        <div class="log-detail hidden">${escHtml(detailStr)}</div>
      ` : ''}
    </div>`;

  if (existingDiv) {
    existingDiv.className = `log-entry ${step.status}`;
    existingDiv.innerHTML = innerHTML;
  } else {
    const div = document.createElement('div');
    div.className = `log-entry ${step.status}`;
    div.innerHTML = innerHTML;
    logEl.appendChild(div);
  }
  logEl.scrollTop = logEl.scrollHeight;
}

function toggleDetail(el) {
  const detail = el.nextElementSibling;
  if (detail) {
    detail.classList.toggle('hidden');
    el.textContent = detail.classList.contains('hidden')
      ? '▸ 세부 정보 보기' : '▾ 접기';
  }
}

// ─── Nearly Flat Band Info ────────────────────────────────────────────────────
function renderNearlyFlatInfo(nfBands) {
  let sec = document.getElementById('nearly-flat-section');
  if (!sec) return;
  sec.style.display = '';

  const rows = nfBands.map(b => {
    const pct = (b.flatness_ratio * 100).toFixed(1);
    return `<tr>
      <td>밴드 ${b.band_index}</td>
      <td>${b.energy.toFixed(4)}</td>
      <td>${b.bandwidth.toExponential(2)}</td>
      <td>${pct}%</td>
    </tr>`;
  }).join('');

  sec.innerHTML = `
    <div class="nearly-flat-card">
      <div class="nearly-flat-header">
        ≈ 거의 평탄 밴드 <span class="nearly-flat-badge">${nfBands.length}개</span>
        <span class="nearly-flat-hint">perturbation으로 인한 flatness 파괴 의심</span>
      </div>
      <table class="nearly-flat-table">
        <thead><tr><th>밴드</th><th>평균 에너지</th><th>대역폭</th><th>전체 대비</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}

// ─── Reciprocal Space & Brillouin Zone Info ──────────────────────────────────
function renderReciprocalSpaceInfo(rs) {
  const sec = document.getElementById('reciprocal-space-section');
  if (!sec) return;
  sec.style.display = 'block';

  // 1. Reciprocal Lattice Vectors
  const vectorsHtml = rs.reciprocal_vectors.map((vec, idx) => {
    const formatted = vec.map(v => v.toFixed(4)).join(', ');
    return `<div>
      <span style="font-weight: 600; color: #4f46e5;">b<sub>${idx+1}</sub></span> = [ ${formatted} ]
    </div>`;
  }).join('');

  // 2. High Symmetry Points
  const hsRows = (rs.high_symmetry_points || []).map(pt => {
    const fracStr = pt.frac.map(v => v.toFixed(4)).join(', ');
    const cartStr = pt.cart.map(v => v.toFixed(4)).join(', ');
    return `<tr>
      <td style="font-weight: 700; color: #1e293b;">${escHtml(pt.label)}</td>
      <td style="font-family: monospace;">(${fracStr})</td>
      <td style="font-family: monospace;">(${cartStr})</td>
    </tr>`;
  }).join('');

  // 3. Brillouin Zone Coordinates
  let bzHtml = '';
  const bz = rs.brillouin_zone;
  if (bz) {
    if (bz.dimension === 1 && bz.vertices) {
      const vals = bz.vertices.map(v => v[0].toFixed(4));
      bzHtml = `<div>1차원 BZ 경계: [ ${vals.join(' ~ ')} ]</div>`;
    } else if (bz.dimension === 2 && bz.vertices) {
      const uniqueVerts = bz.vertices.slice(0, -1);
      const vertStrings = uniqueVerts.map((v, i) => `꼭짓점 ${i+1}: (${v[0].toFixed(4)}, ${v[1].toFixed(4)})`);
      bzHtml = `<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 0.25rem; font-family: monospace; font-size: 0.76rem;">
        ${vertStrings.map(s => `<div>${s}</div>`).join('')}
      </div>`;
    } else if (bz.dimension === 3 && bz.faces) {
      bzHtml = `<div style="font-size: 0.76rem; color: #475569;">
        3차원 Brillouin Zone 페이스 수: ${bz.faces.length}개 
        <span style="font-size:0.7rem; color:#889;">(마우스 드래그 플롯 참고)</span>
      </div>`;
    }
  }

  sec.innerHTML = `
    <div class="nearly-flat-card" style="border-color: #a5b4fc; background: #fafbff; margin-top: 1.5rem;">
      <div class="nearly-flat-header" style="color: #312e81; border-bottom: 1.5px solid #e0e7ff; padding-bottom: 0.4rem; margin-bottom: 0.6rem;">
        🌐 역격자 및 브릴루앙 구역 정보 (Reciprocal Space & BZ)
      </div>
      
      <div style="display: flex; flex-direction: column; gap: 0.9rem;">
        <!-- Reciprocal vectors -->
        <div>
          <div style="font-size: 0.78rem; font-weight: bold; color: #475569; margin-bottom: 0.35rem;">■ 역격자 벡터 (Reciprocal Lattice Vectors)</div>
          <div style="display: flex; gap: 1.5rem; font-family: monospace; font-size: 0.8rem; background: #f8fafc; padding: 0.4rem 0.6rem; border-radius: 6px; border: 1px solid #e2e8f0; flex-wrap: wrap;">
            ${vectorsHtml}
          </div>
        </div>

        <!-- High symmetry points -->
        <div>
          <div style="font-size: 0.78rem; font-weight: bold; color: #475569; margin-bottom: 0.35rem;">■ 고대칭점 좌표 (High Symmetry Points)</div>
          <table class="nearly-flat-table" style="width: 100%;">
            <thead>
              <tr style="border-bottom: 1px solid #cbd5e1;">
                <th style="color: #475569; text-align: left;">레이블</th>
                <th style="color: #475569; text-align: left;">Fractional 좌표</th>
                <th style="color: #475569; text-align: left;">Cartesian 좌표</th>
              </tr>
            </thead>
            <tbody>
              ${hsRows}
            </tbody>
          </table>
        </div>

        <!-- BZ Boundaries -->
        ${bzHtml ? `
        <div>
          <div style="font-size: 0.78rem; font-weight: bold; color: #475569; margin-bottom: 0.35rem;">■ 브릴루앙 구역 좌표 (Brillouin Zone Coordinates)</div>
          <div style="background: #f8fafc; padding: 0.5rem 0.7rem; border-radius: 6px; border: 1px solid #e2e8f0;">
            ${bzHtml}
          </div>
        </div>
        ` : ''}
      </div>
    </div>
  `;
}

// ─── Band Structure Plot ───────────────────────────────────────────────────────
function renderBandPlot(bd) {
  const container = document.getElementById('band-plot');
  const flatSet   = new Set(bd.flat_energies || []);
  const isFlat    = n => {
    if (!bd.bands || !bd.bands[n]) return false;
    const vals = bd.bands[n];
    const mean = vals.reduce((a,b)=>a+b,0)/vals.length;
    return Math.abs(mean - [...flatSet].find(e=>Math.abs(e-mean)<0.01)) < 0.01
           || (flatSet.size > 0 && flatSet.has(mean));
  };

  // Simpler flat detection
  const variances = (bd.bands||[]).map(band => {
    const m = band.reduce((a,b)=>a+b,0)/band.length;
    return band.reduce((a,b)=>a+(b-m)**2,0)/band.length;
  });

  const nearlyFlatSet = new Set(bd.nearly_flat_indices || []);

  const traces = (bd.bands || []).map((band, n) => {
    const flat = variances[n] < 1e-6;
    const nearlyFlat = !flat && nearlyFlatSet.has(n);
    return {
      x: bd.x, y: band, mode: 'lines', type: 'scatter',
      name: flat ? `밴드 ${n} ★ 평탄` : nearlyFlat ? `밴드 ${n} ≈ 거의평탄` : `밴드 ${n}`,
      line: {
        color: flat ? '#e04040' : nearlyFlat ? '#f59e0b' : '#aabbdd',
        width: flat ? 2.5 : nearlyFlat ? 2.0 : 1.2,
        dash: nearlyFlat ? 'dash' : 'solid',
      },
      hovertemplate: `k: %{x}<br>E: %{y:.4f}<extra>밴드 ${n}</extra>`
    };
  });

  // High-symmetry tick marks
  const tickVals = [], tickText = [];
  if (bd.k_ticks) {
    for (const [k, v] of Object.entries(bd.k_ticks)) {
      tickVals.push(parseInt(k));
      tickText.push(v);
    }
  }

  const shapes = tickVals.map(x => ({
    type: 'line', x0: x, x1: x, y0: 0, y1: 1,
    yref: 'paper', line: { color: '#9099bb', width: 1, dash: 'dot' }
  }));

  Plotly.newPlot(container, traces, {
    margin: { t: 30, l: 55, r: 20, b: 50 },
    xaxis: { tickvals: tickVals, ticktext: tickText,
             showgrid: false, zeroline: false,
             title: { text: '' } },
    yaxis: { title: { text: 'Energy' }, gridcolor: '#eef', zeroline: false },
    plot_bgcolor:  '#fff',
    paper_bgcolor: '#fff',
    shapes,
    showlegend: true,
    legend: { x: 1, xanchor: 'right', y: 1 },
    font: { family: 'Segoe UI, sans-serif', size: 12 }
  }, { responsive: true, displayModeBar: false });
}

// ─── CLS Content ────────────────────────────────────────────────────────────
function buildReprSelector(reprs) {
  const sel = document.createElement('div');
  sel.className = 'repr-selector';
  sel.innerHTML = `
    <div class="repr-selector-label">CLS 표현 선택 <span class="repr-selector-hint">(게이지 자유도: 전체 위상 θ 선택)</span></div>
    <div class="repr-btn-group">
      ${reprs.map((r, ri) => `
        <button class="repr-btn ${ri === 0 ? 'active' : ''}"
                data-repr-idx="${ri}"
                title="${escHtml(r.description)}">
          <div class="repr-btn-name">${escHtml(r.label)}</div>
          <div class="repr-btn-meta">
            <span class="repr-mode-badge repr-mode-${r.display_mode}">${modeLabel(r.display_mode)}</span>
            ${r.phase_pattern ? `<span class="repr-phase-tag">${escHtml(r.phase_pattern)}</span>` : ''}
            <span class="repr-realness">${Math.round(r.realness * 100)}% 실수</span>
          </div>
        </button>`).join('')}
    </div>`;
  return sel;
}

function renderGaugeDetails(fb, selectedGaugeId, detailContainer, bandIdx, orbLabels) {
  detailContainer.innerHTML = '';

  const g = fb.gauges.find(x => x.gauge_id === selectedGaugeId) || fb.gauges[0];
  if (!g) return;

  const safeGaugeId = String(g.gauge_id).replace(/[^a-zA-Z0-9_-]/g, '_');
  const reprs = g.representations || [];

  // A. Laurent Polynomial Formula
  const formulaDiv = document.createElement('div');
  formulaDiv.className = 'cls-formula';
  formulaDiv.innerHTML = `<h4>x(k) — 최소화된 CLS 로랑 다항식 벡터 [기반: ${g.method_name}]</h4>
    <div class="laurent-vec">
      ${(g.x_k_min || []).map((expr, q) =>
        `<div class="laurent-comp">
           <span class="laurent-idx">${orbLabels[q] || q}:</span>
           <span class="laurent-expr">${laurentToHtml(expr)}</span>
         </div>`).join('')}
    </div>`;
  detailContainer.appendChild(formulaDiv);

  // B. Representation Selector (only when multiple candidates exist)
  if (reprs.length > 1) {
    detailContainer.appendChild(buildReprSelector(reprs));
  }

  // C. Amplitude Table — container updated when repr changes
  const tableContainer = document.createElement('div');
  detailContainer.appendChild(tableContainer);

  // D. Validation Status
  const valDiv = document.createElement('div');
  valDiv.style.cssText = 'margin: 0.5rem 0; font-size: 0.8rem; font-weight: 500;';
  if (g.validation) {
    valDiv.innerHTML = g.validation.success
      ? `<span class="xcheck-ok" style="background:#ecfdf5;color:#059669;padding:0.3rem 0.6rem;border-radius:4px;display:inline-block;">✓ 고유벡터 검증 성공 — ${escHtml(g.validation.message)}</span>`
      : `<span class="xcheck-fail" style="background:#fef2f2;color:#dc2626;padding:0.3rem 0.6rem;border-radius:4px;display:inline-block;">✗ 고유벡터 검증 실패 — ${escHtml(g.validation.message)}</span>`;
  }
  detailContainer.appendChild(valDiv);

  // E. Cross-check Status
  if (fb.cross_check) {
    const cc = document.createElement('div');
    cc.style.cssText = 'margin: 0.5rem 0; font-size: 0.8rem;';
    cc.innerHTML = fb.cross_check.success
      ? `<span class="xcheck-ok" style="background:#eff6ff;color:#2563eb;padding:0.3rem 0.6rem;border-radius:4px;display:inline-block;">✓ 해석적/수치적 교차검증 통과 — ${escHtml(fb.cross_check.message)}</span>`
      : `<span class="xcheck-fail" style="background:#fff7ed;color:#ea580c;padding:0.3rem 0.6rem;border-radius:4px;display:inline-block;">✗ 교차검증 — ${escHtml(fb.cross_check.message)}</span>`;
    detailContainer.appendChild(cc);
  }

  // F. Plot Row containers
  const rowDiv = document.createElement('div');
  rowDiv.className = 'plots-row';
  const clsDiv = document.createElement('div');
  clsDiv.id = `cls-plot-${bandIdx}-${safeGaugeId}`;
  clsDiv.className = 'cls-plot-container-half';
  const bzDiv = document.createElement('div');
  bzDiv.id = `bz-plot-${bandIdx}-${safeGaugeId}`;
  bzDiv.className = 'cls-plot-container-half';
  rowDiv.appendChild(clsDiv);
  rowDiv.appendChild(bzDiv);
  detailContainer.appendChild(rowDiv);

  function updateReprContent(reprIdx) {
    const repr = reprs[reprIdx] || null;
    const ampData = repr ? repr.amplitudes : g.amplitudes;
    const displayMode = repr ? repr.display_mode : 'complex';

    tableContainer.innerHTML = '';
    tableContainer.appendChild(buildAmplitudeTable(ampData, orbLabels, displayMode));

    if (!g.plot) return;

    // Attempt amplitude patching; fall back to base plot on any failure.
    let plotData = g.plot;
    let activeMode = displayMode;
    if (repr) {
      try {
        const patched = applyReprAmplitudes(g.plot, ampData);
        // Only adopt the patched data if it actually has CLS sites
        if (patched.sites.some(s => s.is_cls)) {
          plotData = patched;
        } else {
          // Patching produced no CLS sites — fall back silently
          activeMode = 'complex';
        }
      } catch (e) {
        activeMode = 'complex';
      }
    }

    const title = `평탄 밴드 #${bandIdx + 1} CLS [${g.method_name}]`;
    setTimeout(() => {
      try {
        renderLatticePlot(plotData, clsDiv.id, title, activeMode);
      } catch (e) {
        // Ultimate fallback: render base plot in complex mode
        try { renderLatticePlot(g.plot, clsDiv.id, title, 'complex'); } catch (_) {}
      }
      if (g.bz_plot) {
        try { renderBZPlot(g.bz_plot, bzDiv.id, `평탄 밴드 #${bandIdx + 1} BZ`); } catch (_) {}
      }
    }, 50);
  }

  // H. Wire up repr buttons
  if (reprs.length > 1) {
    const btns = detailContainer.querySelectorAll('.repr-btn');
    btns.forEach(btn => {
      btn.addEventListener('click', () => {
        btns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        updateReprContent(parseInt(btn.dataset.reprIdx));
      });
    });
  }

  // I. Initial render (best repr = index 0)
  updateReprContent(0);
}

function renderCLSContent(flatBands, orbLabels) {
  const container = document.getElementById('cls-content');
  container.innerHTML = '';

  flatBands.forEach((fb, i) => {
    const sec = document.createElement('div');
    sec.className = 'band-section';

    const singLabel = fb.singular === true  ? '특이형' :
                      fb.singular === false ? '비특이형' : '?';

    sec.innerHTML = `
      <div class="band-section-title">
        <span>평탄 밴드 #${i+1}</span>
        <span class="energy-badge">E = ${fb.energy.toFixed(4)}</span>
        <span class="energy-badge">${singLabel}</span>
      </div>`;

    if (fb.gauges && fb.gauges.length > 0) {
      // Render optimal canonical card at the top
      const primaryGauge = fb.gauges.find(x => x.is_primary) || fb.gauges[0];
      let mc = null;
      if (primaryGauge.canonical) {
        mc = primaryGauge.canonical;
      } else if (primaryGauge.representations && primaryGauge.representations.length > 0) {
        mc = buildCanonicalFromRepr(primaryGauge.representations[0]);
      }
      if (mc) {
        const mcCard = buildMinimalCLSCard(mc, orbLabels);
        if (mcCard) {
          sec.appendChild(mcCard);
        }
      }

      // Gauge select row
      const gaugeSelectRow = document.createElement('div');
      gaugeSelectRow.className = 'gauge-select-row';
      gaugeSelectRow.style.cssText = 'margin: 0.5rem 0 1rem 0; display: flex; gap: 0.5rem; align-items: center;';
      
      const selectLbl = document.createElement('label');
      selectLbl.textContent = '게이지 선택 (Gauge Choice):';
      selectLbl.style.cssText = 'font-size: 0.8rem; font-weight: bold; color: #475569;';
      
      const selectEl = document.createElement('select');
      selectEl.className = 'input-num';
      selectEl.style.cssText = 'flex: 1; font-size: 0.82rem; padding: 0.3rem 0.5rem; border-radius: 6px; cursor: pointer;';
      
      fb.gauges.forEach((g) => {
        const opt = document.createElement('option');
        opt.value = g.gauge_id;
        const primaryMark = g.is_primary ? ' ★ 추천' : '';
        const singularMark = g.singular ? ' ⚠ 특이성 있음' : ' (비특이)';
        const sizeMark = ` [크기: ${g.support_size}]`;
        opt.textContent = `${g.method_name}${primaryMark}${singularMark}${sizeMark}`;
        if (g.is_primary) opt.selected = true;
        selectEl.appendChild(opt);
      });
      
      gaugeSelectRow.appendChild(selectLbl);
      gaugeSelectRow.appendChild(selectEl);
      sec.appendChild(gaugeSelectRow);
      
      const detailContainer = document.createElement('div');
      sec.appendChild(detailContainer);
      
      // Render initially
      renderGaugeDetails(fb, primaryGauge.gauge_id, detailContainer, i, orbLabels);
      
      // Change listener
      selectEl.addEventListener('change', (e) => {
        renderGaugeDetails(fb, e.target.value, detailContainer, i, orbLabels);
      });
    } else if (fb.cls) {
      // Fallback for old spec format (compatibility)
      const formulaDiv = document.createElement('div');
      formulaDiv.className = 'cls-formula';
      formulaDiv.innerHTML = `<h4>x(k) — 최소화된 CLS 로랑 다항식 벡터</h4>
        <div class="laurent-vec">
          ${(fb.cls.x_k_min || []).map((expr, q) =>
            `<div class="laurent-comp">
               <span class="laurent-idx">${orbLabels[q] || q}:</span>
               <span class="laurent-expr">${laurentToHtml(expr)}</span>
             </div>`).join('')}
        </div>`;
      sec.appendChild(formulaDiv);

      const tableDiv = buildAmplitudeTable(fb.cls.amplitudes, orbLabels);
      sec.appendChild(tableDiv);

      if (fb.cross_check) {
        const cc = document.createElement('div');
        cc.innerHTML = fb.cross_check.success
          ? `<span class="xcheck-ok">✓ 해석적/수치적 교차검증 통과 — ${escHtml(fb.cross_check.message)}</span>`
          : `<span class="xcheck-fail">✗ 교차검증 — ${escHtml(fb.cross_check.message)}</span>`;
        sec.appendChild(cc);
      }

      if (fb.cls.plot) {
        const rowDiv = document.createElement('div');
        rowDiv.className = 'plots-row';
        
        const clsDiv  = document.createElement('div');
        clsDiv.id     = `cls-plot-${i}`;
        clsDiv.className = 'cls-plot-container-half';
        
        const bzDiv  = document.createElement('div');
        bzDiv.id     = `bz-plot-${i}`;
        bzDiv.className = 'cls-plot-container-half';
        
        rowDiv.appendChild(clsDiv);
        rowDiv.appendChild(bzDiv);
        sec.appendChild(rowDiv);
        
        setTimeout(() => {
          renderLatticePlot(fb.cls.plot, `cls-plot-${i}`, `평탄 밴드 #${i+1} CLS 실공간 격자`);
          if (fb.cls.bz_plot) {
            renderBZPlot(fb.cls.bz_plot, `bz-plot-${i}`, `평탄 밴드 #${i+1} 브릴루앙 구역 (BZ)`);
          }
        }, 50);
      }
    } else {
      sec.innerHTML += '<p style="color:#888;font-size:.8rem">해석적 CLS 도출 실패</p>';
    }

    container.appendChild(sec);
  });
}

// ─── Symbolic Number Recognition ─────────────────────────────────────────────
function _gcd(a, b) { a = Math.abs(a); b = Math.abs(b); while (b) { [a, b] = [b, a % b]; } return a; }

/**
 * Try to express |x| as a nice symbolic string.
 * Returns e.g. "√3/2", "3/4", "1", "√2" or null if unrecognized.
 */
function toNiceNum(x, tol = 1e-3) {
  if (!isFinite(x)) return null;
  const ax = Math.abs(x);
  if (ax < tol * 0.5) return '0';

  // Simple fractions: n/d for small n, d
  for (const d of [1, 2, 3, 4, 6, 8, 12]) {
    const n = Math.round(ax * d);
    if (n > 0 && n <= 4 * d && Math.abs(ax - n / d) < tol) {
      const g = _gcd(n, d);
      const ns = n / g, ds = d / g;
      return ds === 1 ? `${ns}` : `${ns}/${ds}`;
    }
  }

  // n·√k / d
  const sqRoots = [[2, Math.SQRT2], [3, Math.sqrt(3)], [5, Math.sqrt(5)], [6, Math.sqrt(6)]];
  for (const [k, sq] of sqRoots) {
    for (const d of [1, 2, 3, 4, 6]) {
      for (const n of [1, 2, 3, 4]) {
        if (Math.abs(ax - n * sq / d) < tol) {
          const g = _gcd(n, d);
          const ns = n / g, ds = d / g;
          const sym = k === 2 ? '√2' : k === 3 ? '√3' : k === 5 ? '√5' : '√6';
          const num_str = ns === 1 ? sym : `${ns}${sym}`;
          return ds === 1 ? num_str : `${num_str}/${ds}`;
        }
      }
    }
  }

  return null;
}

/** Format a real number: symbolic if possible, else 4dp decimal. */
function niceReal(x) {
  const s = toNiceNum(x);
  if (s === null) return x.toFixed(4);
  return (x < 0 && s !== '0') ? `-${s}` : s;
}

/** Format a complex number with symbolic recognition. */
function niceComplex(re, im) {
  if (Math.abs(im) < 1e-6) return niceReal(re);
  if (Math.abs(re) < 1e-6) {
    const is = toNiceNum(im);
    const iStr = is !== null ? ((im < 0 ? '-' : '') + (is === '1' ? '' : is)) : im.toFixed(4);
    return `${iStr}i`;
  }
  const rs = niceReal(re);
  const iAbs = Math.abs(im);
  const is = toNiceNum(iAbs);
  const iStr = is !== null ? is : iAbs.toFixed(4);
  const sign = im > 0 ? ' + ' : ' - ';
  return `${rs}${sign}${iStr === '1' ? '' : iStr}i`;
}

// ─── Amplitude Table ──────────────────────────────────────────────────────────
function formatComplex(re, im) {
  return niceComplex(re, im);
}

function buildAmplitudeTable(ampData, orbLabels, displayMode = 'complex') {
  const wrap = document.createElement('div');
  wrap.className = 'amp-table-wrap';

  const showPhase = displayMode !== 'amplitude';
  const phaseHeader = showPhase ? '<th>위상</th>' : '';
  const ampHeader = displayMode === 'amplitude' ? '진폭 (부호)' : '진폭 A';

  let rows = '';
  for (const [qi, qdata] of Object.entries(ampData)) {
    const label = qdata.label || orbLabels[parseInt(qi)] || qi;
    for (const a of qdata.amplitudes || []) {
      const cellStr = a.cell.join(', ');
      const abs = niceReal(a.abs);
      let valStr, rowCls, phaseCell = '';

      if (displayMode === 'amplitude') {
        if (Math.abs(a.im) < 1e-4) {
          valStr = niceReal(a.re);
          rowCls = a.re >= 0 ? 'amp-pos' : 'amp-neg';
        } else {
          valStr = niceReal(a.abs);
          rowCls = 'amp-cmp';
        }
      } else {
        rowCls = Math.abs(a.im) < 1e-6
          ? (a.re > 0 ? 'amp-pos' : 'amp-neg')
          : 'amp-cmp';
        valStr = formatComplex(a.re, a.im);
        if (showPhase) {
          const deg = (Math.atan2(a.im, a.re) * 180 / Math.PI + 360) % 360;
          phaseCell = `<td>${deg.toFixed(1)}°</td>`;
        }
      }

      rows += `<tr>
        <td>${escHtml(label)}</td>
        <td>(${escHtml(cellStr)})</td>
        <td class="${rowCls}">${escHtml(valStr)}</td>
        ${phaseCell || (showPhase ? '<td>—</td>' : '')}
        <td>${escHtml(abs)}</td>
      </tr>`;
    }
  }

  wrap.innerHTML = `
    <table class="amp-table">
      <thead><tr>
        <th>오비탈</th><th>셀 오프셋</th><th>${ampHeader}</th>
        ${phaseHeader}<th>|A|</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  return wrap;
}

function buildCanonicalFromRepr(repr) {
  if (!repr || !repr.amplitudes) return null;
  
  const sites = [];
  for (const [qStr, qData] of Object.entries(repr.amplitudes)) {
    const q = parseInt(qStr);
    const label = qData.label || String(q);
    for (const amp of qData.amplitudes || []) {
      sites.push({
        orbital: q,
        label: label,
        cell: amp.cell,
        re: amp.re,
        im: amp.im,
        abs: amp.abs,
        phase_deg: ((Math.atan2(amp.im, amp.re) * 180 / Math.PI) + 360) % 360
      });
    }
  }

  if (sites.length === 0) return null;

  // Find reference cell (largest magnitude, tie-break: lower orbital, cell order)
  const sortedForRef = [...sites].sort((a, b) => {
    if (Math.abs(b.abs - a.abs) > 1e-5) return b.abs - a.abs;
    if (a.orbital !== b.orbital) return a.orbital - b.orbital;
    for (let i = 0; i < a.cell.length; i++) {
      if (a.cell[i] !== b.cell[i]) return a.cell[i] - b.cell[i];
    }
    return 0;
  });

  const refCell = sortedForRef[0].cell;
  const refOrbital = sortedForRef[0].orbital;

  // Compute rel_cell offsets
  sites.forEach(s => {
    s.rel_cell = s.cell.map((val, idx) => val - (refCell[idx] || 0));
  });

  // Sort sites by Euclidean distance from reference, then orbital index, then cell
  sites.sort((a, b) => {
    const distA = Math.sqrt(a.rel_cell.reduce((sum, x) => sum + x*x, 0));
    const distB = Math.sqrt(b.rel_cell.reduce((sum, x) => sum + x*x, 0));
    if (Math.abs(distA - distB) > 1e-5) return distA - distB;
    if (a.orbital !== b.orbital) return a.orbital - b.orbital;
    for (let i = 0; i < a.cell.length; i++) {
      if (a.cell[i] !== b.cell[i]) return a.cell[i] - b.cell[i];
    }
    return 0;
  });

  return {
    sites: sites,
    ref_orbital: refOrbital,
    ref_cell: refCell,
    support_size: sites.length,
    global_phase_deg: repr.global_phase_deg,
    display_mode: repr.display_mode,
    realness: repr.realness,
    phase_pattern: repr.phase_pattern
  };
}

// ─── Minimal CLS Card ────────────────────────────────────────────────────────
function formatRelCell(relCell) {
  return '(' + relCell.map(x => x > 0 ? '+' + x : String(x)).join(', ') + ')';
}

function buildMinimalCLSCard(mc, orbLabels) {
  if (!mc || !mc.sites || mc.sites.length === 0) return null;

  const card = document.createElement('div');
  card.className = 'minimal-cls-card';

  const modeTag = `<span class="repr-mode-badge repr-mode-${mc.display_mode}">${modeLabel(mc.display_mode)}</span>`;
  const ppTag = mc.phase_pattern ? `<span class="repr-phase-tag">${escHtml(mc.phase_pattern)}</span>` : '';
  const realPct = Math.round((mc.realness || 0) * 100);
  const thetaStr = mc.global_phase_deg != null ? `θ = ${mc.global_phase_deg.toFixed(1)}°` : '';

  // Compact formula: Label(rel_cell) = value
  const formulaParts = mc.sites.map(s => {
    const lbl = s.label || orbLabels[s.orbital] || String(s.orbital);
    const cellStr = formatRelCell(s.rel_cell);
    let valStr;
    if (Math.abs(s.im) < 1e-4) {
      const niceSym = toNiceNum(Math.abs(s.re));
      if (niceSym !== null) {
        valStr = (s.re >= 0 ? '+' : '-') + niceSym;
      } else {
        const v = parseFloat(s.re.toFixed(4));
        valStr = (v >= 0 ? '+' : '') + v;
      }
    } else {
      const deg = ((Math.atan2(s.im, s.re) * 180 / Math.PI) + 360) % 360;
      const niceSym = toNiceNum(s.abs);
      const absStr = niceSym !== null ? niceSym : s.abs.toFixed(3);
      valStr = `${absStr}∠${deg.toFixed(1)}°`;
    }
    return `<span class="mcls-term">${escHtml(lbl)}<sub>${escHtml(cellStr)}</sub>&thinsp;=&thinsp;${escHtml(valStr)}</span>`;
  });

  // Table rows
  const showPhase = mc.display_mode !== 'amplitude';
  let rows = '';
  mc.sites.forEach((s, si) => {
    const lbl = s.label || orbLabels[s.orbital] || String(s.orbital);
    const relCellStr = formatRelCell(s.rel_cell);
    const isRef = s.rel_cell.every(x => x === 0) && si === 0;
    let valStr, rowCls;
    if (mc.display_mode === 'amplitude' || Math.abs(s.im) < 1e-4) {
      const niceSym = toNiceNum(Math.abs(s.re));
      valStr = niceSym !== null
        ? (s.re >= 0 ? '+' : '-') + niceSym
        : (s.re >= 0 ? '+' : '') + s.re.toFixed(4);
      rowCls = s.re >= 0 ? 'amp-pos' : 'amp-neg';
    } else {
      valStr = formatComplex(s.re, s.im);
      rowCls = 'amp-cmp';
    }
    const phaseDeg = ((Math.atan2(s.im, s.re) * 180 / Math.PI) + 360) % 360;
    const phaseCell = showPhase ? `<td>${phaseDeg.toFixed(1)}°</td>` : '';
    const refMark = isRef ? ' <span class="mcls-ref-badge">REF</span>' : '';
    rows += `<tr${isRef ? ' class="mcls-ref-row"' : ''}>
      <td>${escHtml(lbl)}${refMark}</td>
      <td class="mcls-cell-col">${escHtml(relCellStr)}</td>
      <td class="${rowCls}">${escHtml(valStr)}</td>
      ${phaseCell}
      <td>${niceReal(s.abs)}</td>
    </tr>`;
  });

  const phaseHeader = showPhase ? '<th>위상</th>' : '';

  card.innerHTML = `
    <div class="minimal-cls-header">
      <span class="minimal-cls-title">최소 CLS &thinsp;(크기 ${mc.support_size})</span>
      ${modeTag}${ppTag}
      <span class="minimal-cls-meta">${escHtml(thetaStr)} &middot; 실수율 ${realPct}%</span>
    </div>
    <div class="minimal-cls-formula">${formulaParts.join(',&ensp;')}</div>
    <table class="amp-table">
      <thead><tr>
        <th>오비탈</th><th>상대 셀</th>
        <th>진폭</th>${phaseHeader}<th>|A|</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;

  return card;
}

// ─── Classification Panel ─────────────────────────────────────────────────────
function renderClassification(flatBands, orbLabels) {
  const container = document.getElementById('classify-content');
  container.innerHTML = '';

  flatBands.forEach((fb, i) => {
    const card = document.createElement('div');
    card.className = 'classify-card';

    const singClass = fb.singular ? 'singular' : 'nonsingular';
    const singText  = fb.singular === true  ? '특이형 (Singular)' :
                      fb.singular === false ? '비특이형 (Non-Singular)' :
                      '분류 불가';

    const hasGridPlot = fb.chern && fb.chern.grid_data && (fb.bz_plot || (fb.cls && fb.cls.bz_plot));

    if (hasGridPlot) {
      card.innerHTML = `
        <div class="band-section-title">평탄 밴드 #${i+1}</div>
        <div class="classify-layout-cols" style="display: flex; gap: 1.5rem; flex-wrap: wrap;">
          <div class="classify-col-left" style="flex: 1.1; min-width: 320px;">
            <div class="classify-result" style="margin-bottom: 0.8rem; display: flex; align-items: center; gap: 1rem;">
              <div class="sing-badge ${singClass}" style="font-size: 0.95rem; padding: 0.35rem 0.75rem;">${singText}</div>
              <div class="classify-energy">에너지: ${fb.energy.toFixed(4)} | 밴드 인덱스: ${fb.band_index}</div>
            </div>
            <div class="k0-container"></div>
            <div class="chern-container"></div>
            <div class="explanation-container" style="font-size:.78rem;color:#556;margin-top:.8rem;padding:.6rem;background:#f8f9fc;border-radius:6px;line-height:1.6"></div>
          </div>
          <div class="classify-col-right" style="flex: 0.9; min-width: 320px; display: flex; flex-direction: column;">
            <div class="bz-plot-controls-container"></div>
            <div class="bz-plot-container-wrapper" style="position: relative;">
              <div id="classify-bz-plot-${i}" class="plotly-container" style="height: 380px; width: 100%;"></div>
            </div>
          </div>
        </div>
        
        <div class="projector-analysis-section" style="margin-top: 1.5rem; border-top: 1px dashed #cbd5e1; padding-top: 1.2rem;">
          <div style="font-weight: bold; color: #1e1b4b; font-size: 1.05rem; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem;">
            <span>🔬</span> 사영자 & 베리 곡률 분석 (Projector & Berry Curvature Analysis)
          </div>
          <p style="font-size: 0.76rem; color: #64748b; margin-bottom: 1rem; line-height: 1.5;">
            평탄 밴드의 <strong>투영 연산자(Projector) P_αβ(k)</strong> 및 <strong>베리 곡률(Berry Curvature) F_xy(k)</strong>을 분석합니다.
            공통 영점 k₀ (vortex) 주위에서 사영자가 연속적인지(즉, 방향에 무관한 극한을 갖는지), 베리 곡률이 어디에 집중되어 있는지 직접 확인하고 해석할 수 있습니다.
          </p>
          
          <div class="projector-controls" style="display: flex; gap: 0.8rem; align-items: center; background: #f8fafc; border: 1.5px solid #e2e8f0; padding: 0.6rem 1rem; border-radius: 8px; flex-wrap: wrap; margin-bottom: 1rem;">
            <div style="display: flex; align-items: center; gap: 0.4rem;">
              <span style="font-size: 0.74rem; font-weight: bold; color: #475569;">오비탈 1 (α):</span>
              <select class="proj-orb-1" style="padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 0.74rem; background: white;"></select>
            </div>
            
            <div style="display: flex; align-items: center; gap: 0.4rem;">
              <span style="font-size: 0.74rem; font-weight: bold; color: #475569;">오비탈 2 (β):</span>
              <select class="proj-orb-2" style="padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 0.74rem; background: white;"></select>
            </div>
            
            <div style="display: flex; align-items: center; gap: 0.4rem;">
              <span style="font-size: 0.74rem; font-weight: bold; color: #475569;">시각화 대상:</span>
              <select class="proj-quantity" style="padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 0.74rem; background: white;">
                <option value="mag" selected>Projector Element Magnitude |P_αβ(k)|</option>
                <option value="real">Projector Element Real Part Re(P_αβ(k))</option>
                <option value="imag">Projector Element Imag Part Im(P_αβ(k))</option>
                <option value="phase">Projector Element Phase arg(P_αβ(k))</option>
                <option value="berry">Berry Curvature F_xy(k)</option>
              </select>
            </div>
            
            <div style="display: flex; align-items: center; gap: 0.4rem;">
              <span style="font-size: 0.74rem; font-weight: bold; color: #475569;">표시 모드:</span>
              <select class="proj-plot-mode" style="padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 0.74rem; background: white;">
                <option value="contour">2D Contour (등고선/히트맵)</option>
                <option value="surface">3D Surface (입체 3D 표면)</option>
                <option value="cut">1D Cut (특이점 통과 1D 단면)</option>
              </select>
            </div>
            
            <div class="proj-singularity-select-div" style="display: none; align-items: center; gap: 0.4rem;">
              <span style="font-size: 0.74rem; font-weight: bold; color: #475569;">기준 특이점 k₀:</span>
              <select class="proj-singularity-select" style="padding: 0.2rem 0.5rem; border-radius: 4px; border: 1px solid #cbd5e1; font-size: 0.74rem; background: white;"></select>
            </div>
          </div>
          
          <div class="projector-plots-row" style="display: flex; gap: 1rem; flex-wrap: wrap;">
            <div class="projector-plot-wrapper" style="flex: 1.3; min-width: 320px; border: 1px solid #e2e8f0; border-radius: 8px; background: white; padding: 0.5rem; position: relative;">
              <div class="proj-plot-container" style="height: 400px; width: 100%;"></div>
            </div>
            <div class="projector-analysis-card" style="flex: 0.7; min-width: 250px; background: #faf5ff; border: 1.5px solid #e9d5ff; border-radius: 8px; padding: 0.8rem 1rem; display: flex; flex-direction: column; gap: 0.6rem;">
              <div style="font-weight: bold; color: #581c87; font-size: 0.84rem; display: flex; align-items: center; gap: 0.3rem;">
                <span>💡</span> 물리적 해석 &amp; 가이드
              </div>
              <div class="proj-interpretation-content" style="font-size: 0.76rem; color: #4b5563; line-height: 1.6; flex: 1;"></div>
            </div>
          </div>
        </div>
      `;
    } else {
      card.innerHTML = `
        <div class="band-section-title">평탄 밴드 #${i+1}</div>
        <div class="classify-result" style="display: flex; align-items: center; gap: 1rem; margin-bottom: 1rem;">
          <div class="sing-badge ${singClass}">${singText}</div>
          <div class="classify-energy">에너지: ${fb.energy.toFixed(4)} | 밴드 인덱스: ${fb.band_index}</div>
        </div>
        <div class="k0-container"></div>
        <div class="chern-container"></div>
        <div class="explanation-container" style="font-size:.78rem;color:#556;margin-top:.8rem;padding:.6rem;background:#f8f9fc;border-radius:6px;line-height:1.6"></div>
      `;
    }

    // Populate k0 list
    const k0Container = card.querySelector('.k0-container');
    if (fb.singular && fb.k0_list && fb.k0_list.length > 0) {
      const k0div = document.createElement('div');
      k0div.className = 'k0-list';
      k0div.innerHTML = `<h4>특이점 k₀ (${fb.k0_list.length}개)</h4>`;
      fb.k0_list.forEach((k0, ki) => {
        const span = document.createElement('span');
        span.className = 'k0-item';
        span.textContent = `k₀[${ki}] = (${k0.map(v => v.toFixed(3)).join(', ')})`;
        k0div.appendChild(span);
      });
      k0Container.appendChild(k0div);
    } else {
      k0Container.innerHTML = '<p style="font-size:.75rem; color:#888; margin-top:.4rem;">발견된 k₀ 특이점 없음</p>';
    }

    // Populate Chern block
    const chernContainer = card.querySelector('.chern-container');
    if (fb.chern && typeof fb.chern.chern_number !== 'undefined') {
      const ch = fb.chern;
      const iso = ch.isolation || {};
      const cdiv = document.createElement('div');
      cdiv.className = 'chern-block';
      cdiv.style.cssText = 'margin-top:.7rem;padding:.6rem .7rem;border-radius:6px;'
        + 'border:1.5px solid ' + (iso.isolated === false ? '#fca5a5' : '#a5b4fc')
        + ';background:' + (iso.isolated === false ? '#fef2f2' : '#eef2ff') + ';';
      const cval = (iso.isolated === false)
        ? `C = ${ch.chern_number} <span style="color:#b91c1c">(고립 밴드 아님 — 미정의)</span>`
        : `C = <strong>${ch.chern_number}</strong>`;
      let detail = '';
      if (ch.analytic) {
        const a = ch.analytic;
        const arrow = '▸';
        detail = `
          <div class="ry-checklist" style="margin-top: 0.5rem; font-size: 0.74rem; color: #4b5563; background: #ffffff; padding: 0.5rem; border-radius: 4px; border: 1px dashed #cbd5e1;">
            <div style="font-weight: bold; color: #374151; margin-bottom: 0.25rem; border-bottom: 1px solid #cbd5e1; padding-bottom: 0.15rem;">Rhim-Yang 조건 분석:</div>
            <div style="display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.15rem;">
              <span style="color: ${a.has_common_zero ? '#10b981' : '#ef4444'}; font-weight: bold;">${a.has_common_zero ? '✓' : '✗'}</span>
              <span>1. 공통 영점 (Common Zeros): <strong>${a.n_common_zeros}개</strong></span>
            </div>
            ${a.has_common_zero ? `
            <div style="display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.15rem;">
              <span style="color: ${a.projector_continuous ? '#10b981' : '#f59e0b'}; font-weight: bold;">${a.projector_continuous ? '✓' : '✗'}</span>
              <span>2. 사영자 연속성 (Continuity): <strong>${a.projector_continuous ? '만족 (Rank=1)' : '불만족 (Rank≠1)'}</strong></span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.15rem;">
              <span style="color: ${a.nonzero_winding ? '#10b981' : '#6b7280'}; font-weight: bold;">${a.nonzero_winding ? '✓' : '✗'}</span>
              <span>3. 와인딩 발생 (Winding): <strong>${a.nonzero_winding ? '발생' : '소멸 (w_i=0)'}</strong></span>
            </div>
            <div style="display: flex; align-items: center; gap: 0.4rem; margin-bottom: 0.15rem; padding-left: 0.8rem; font-size: 0.7rem; color: #6b7280;">
              <span>${arrow} 각 영점별 Winding 수 합 (Σw_i) = <strong>${a.C}</strong></span>
            </div>
            ` : ''}
            <div style="display: flex; align-items: center; gap: 0.4rem; margin-top: 0.25rem; border-top: 1px dotted #cbd5e1; padding-top: 0.2rem; font-size: 0.72rem;">
              <span style="color: ${ch.agreement ? '#10b981' : '#ef4444'}; font-weight: bold;">${ch.agreement ? '✓' : '✗'}</span>
              <span>수치 FHS (C=${ch.numerical.C}) ↔ Winding 합 (Σw_i=${a.C}) 일치</span>
            </div>
          </div>
        `;
      }
      cdiv.innerHTML =
        `<div style="font-weight:bold;color:#3730a3;">Chern 수 (FHS + 유한푸리에 분석)</div>`
        + `<div style="font-size:1.05rem;margin:.2rem 0;">${cval}</div>`
        + (detail ? detail : '')
        + (ch.explanation ? `<div style="font-size:.74rem;color:#444;margin-top:.3rem;">${ch.explanation}</div>` : '');
      chernContainer.appendChild(cdiv);
    }

    // Populate explanation
    const expContainer = card.querySelector('.explanation-container');
    if (fb.singular) {
      expContainer.innerHTML = `
        <strong>특이형 (Singular)</strong>: CLS가 BZ의 특이점 k₀에서 파동함수 불연속성을 가집니다.<br>
        이 평탄 밴드의 CLS는 BZ 전체를 채울 수 없으며 (불완전성),<br>
        비수축 루프 상태 (NLS/NPS)가 존재합니다.<br>
        외부 섭동에 의해 평탄 밴드가 휘어지고 Chern 수가 생성될 수 있습니다.`;
    } else {
      expContainer.innerHTML = `
        <strong>비특이형 (Non-Singular)</strong>: CLS가 BZ 전체에서 연속적인 파동함수를 가집니다.<br>
        이 평탄 밴드의 CLS는 BZ를 완전히 채울 수 있으며 (완전성),<br>
        비수축 상태 (NLS/NPS)가 존재하지 않습니다.<br>
        외부 섭동에 의해 밴드 갭이 열리더라도 위상적으로 자명합니다.`;
    }

    // Populate BZ plot and controls if grid_data is available
    if (hasGridPlot) {
      const controlsContainer = card.querySelector('.bz-plot-controls-container');
      
      const toolbar = document.createElement('div');
      toolbar.className = 'lat-toolbar';
      toolbar.style.cssText = 'margin-bottom: 0.5rem; display: flex; align-items: center; gap: 0.6rem; padding: 0.35rem 0.5rem; border-radius: 6px; border: 1px solid #dde3ec; background: #f8fafc;';
      toolbar.innerHTML = `
        <span class="lat-ctrl-label" style="font-weight: bold; color: #475569; font-size: 0.74rem;">시각화 모드:</span>
        <label class="lat-toggle" style="margin-right: 10px; font-size: 0.74rem;">
          <input type="radio" name="bz-view-mode-${i}" value="amp" checked style="accent-color:#4c7aff; cursor:pointer;">
          <span>진폭 크기 |f(k)|</span>
        </label>
        <label class="lat-toggle" style="font-size: 0.74rem;">
          <input type="radio" name="bz-view-mode-${i}" value="phase" style="accent-color:#4c7aff; cursor:pointer;">
          <span>우세 오비탈 위상 arg(f)</span>
        </label>
      `;
      controlsContainer.appendChild(toolbar);

      const bzPlotId = `classify-bz-plot-${i}`;
      setTimeout(() => {
        renderBZSingularityPlot(fb, bzPlotId, 'amp');
      }, 50);

      toolbar.querySelectorAll(`input[name="bz-view-mode-${i}"]`).forEach(radio => {
        radio.addEventListener('change', (e) => {
          renderBZSingularityPlot(fb, bzPlotId, e.target.value);
        });
      });
    }

    // NLS/NPS section
    if (fb.nls && fb.nls.length > 0) {
      const nlsSec = document.createElement('div');
      nlsSec.className = 'nls-section';
      nlsSec.innerHTML = '<h4>비수축 루프/평면 상태 (NLS/NPS)</h4>';

      const tabBar = document.createElement('div');
      tabBar.className = 'nls-tab-bar';
      fb.nls.forEach((nls, ni) => {
        const btn = document.createElement('button');
        btn.className = `nls-tab ${ni === 0 ? 'active' : ''}`;
        btn.textContent = `축 ${nls.keep_axis} 방향 NLS`;
        btn.onclick = () => {
          tabBar.querySelectorAll('.nls-tab').forEach(b => b.classList.remove('active'));
          btn.classList.add('active');
          nlsSec.querySelectorAll('.nls-plot-container').forEach((p, pi) => {
            p.style.display = pi === ni ? 'block' : 'none';
          });
        };
        tabBar.appendChild(btn);
      });
      nlsSec.appendChild(tabBar);

      fb.nls.forEach((nls, ni) => {
        const pd = document.createElement('div');
        pd.id        = `nls-plot-${i}-${ni}`;
        pd.className = 'nls-plot-container';
        pd.style.display = ni === 0 ? 'block' : 'none';
        nlsSec.appendChild(pd);
        setTimeout(() => renderLatticePlot(nls.plot, `nls-plot-${i}-${ni}`,
          `NLS 축 ${nls.keep_axis}`), 100 * ni + 80);
      });

      card.appendChild(nlsSec);
    }

    if (hasGridPlot) {
      initProjectorAnalysis(fb, card, orbLabels, i);
    }

    container.appendChild(card);
  });
}

function renderBZSingularityPlot(fb, containerId, viewMode) {
  const container = document.getElementById(containerId);
  if (!container || !fb || !fb.chern || !fb.chern.grid_data) return;
  
  const bzData = fb.bz_plot || (fb.cls && fb.cls.bz_plot);
  if (!bzData) return;
  
  const grid_data = fb.chern.grid_data;
  const traces = [];
  
  // 1. Contour plot of the wave function
  if (viewMode === 'amp') {
    traces.push({
      x: grid_data.x,
      y: grid_data.y,
      z: grid_data.z_amp,
      type: 'contour',
      colorscale: 'Viridis',
      reversescale: false,
      line: { width: 0 },
      contours: { coloring: 'heatmap' },
      showscale: true,
      colorbar: {
        title: { text: '|f(k)|', font: { size: 10, weight: 'bold' } },
        thickness: 12,
        len: 0.8,
        y: 0.5
      },
      hoverinfo: 'x+y+z',
      name: '|f(k)| 크기'
    });
  } else {
    traces.push({
      x: grid_data.x,
      y: grid_data.y,
      z: grid_data.z_phase,
      type: 'contour',
      colorscale: 'HSV', // cyclic phase colorscale
      line: { width: 0 },
      contours: { coloring: 'heatmap' },
      showscale: true,
      colorbar: {
        title: { text: 'arg(f) (rad)', font: { size: 10, weight: 'bold' } },
        thickness: 12,
        len: 0.8,
        y: 0.5,
        tickvals: [-Math.PI, -Math.PI/2, 0, Math.PI/2, Math.PI],
        ticktext: ['-π', '-π/2', '0', 'π/2', 'π']
      },
      hoverinfo: 'x+y+z',
      name: '우세오비탈 위상'
    });
  }
  
  // 2. BZ Boundary Polygon
  const polyX = bzData.vertices.map(v => v[0]);
  const polyY = bzData.vertices.map(v => v[1]);
  traces.push({
    x: polyX,
    y: polyY,
    mode: 'lines',
    type: 'scatter',
    line: { color: '#0f172a', width: 2.5 },
    name: '1st BZ 경계',
    hoverinfo: 'none',
    showlegend: true
  });
  
  // 3. Reciprocal Vectors
  const b1 = bzData.recip_vectors[0];
  const b2 = bzData.recip_vectors[1];
  traces.push({
    x: [0, b1[0], null, 0, b2[0]],
    y: [0, b1[1], null, 0, b2[1]],
    mode: 'lines',
    type: 'scatter',
    line: { color: '#475569', width: 1.5, dash: 'dash' },
    name: '역격자 벡터',
    hoverinfo: 'none',
    showlegend: true
  });
  
  const annotations = [
    {
      x: b1[0], y: b1[1], text: 'b₁', showarrow: false,
      xanchor: 'left', yanchor: 'bottom', font: { size: 12, color: '#475569', weight: 'bold' }
    },
    {
      x: b2[0], y: b2[1], text: 'b₂', showarrow: false,
      xanchor: 'left', yanchor: 'bottom', font: { size: 12, color: '#475569', weight: 'bold' }
    }
  ];
  
  // 4. k-path
  if (bzData.sym_points && bzData.sym_points.length > 0) {
    const pathX = bzData.sym_points.map(p => p.x);
    const pathY = bzData.sym_points.map(p => p.y);
    traces.push({
      x: pathX,
      y: pathY,
      mode: 'lines+markers',
      type: 'scatter',
      line: { color: '#f97316', width: 1.2, dash: 'dot' },
      marker: { color: '#f97316', size: 4 },
      name: 'k-경로',
      hoverinfo: 'none',
      showlegend: true
    });
    
    // High symmetry points
    const uniqueSyms = {};
    bzData.sym_points.forEach(p => {
      const key = `${p.x.toFixed(3)},${p.y.toFixed(3)}`;
      if (!uniqueSyms[key]) uniqueSyms[key] = p;
    });
    const uSymList = Object.values(uniqueSyms);
    traces.push({
      x: uSymList.map(p => p.x),
      y: uSymList.map(p => p.y),
      mode: 'markers+text',
      type: 'scatter',
      marker: { size: 6, color: '#e2e8f0', line: { color: '#1e293b', width: 1 } },
      text: uSymList.map(p => p.label),
      textposition: 'top right',
      textfont: { size: 9, color: '#1e293b', weight: 'bold' },
      name: '고대칭점',
      hovertemplate: '대칭점: %{text}<br>k: (%{x:.3f}, %{y:.3f})<extra></extra>',
      showlegend: true
    });
  }
  
  // 5. Singularity Zeros
  if (fb.chern.analytic && fb.chern.analytic.per_zero && fb.chern.analytic.per_zero.length > 0) {
    const zeros = fb.chern.analytic.per_zero;
    
    // Group zeros by winding number
    const windingGroups = {};
    zeros.forEach(z => {
      const w = z.winding || 0;
      if (!windingGroups[w]) {
        windingGroups[w] = [];
      }
      windingGroups[w].push(z);
    });
    
    // Sort winding groups so they appear in a consistent order in the legend (positive first)
    const windings = Object.keys(windingGroups).map(Number).sort((a, b) => b - a);
    
    windings.forEach(w => {
      const zeroGroup = windingGroups[w];
      let color = '#6b7280'; // gray default for w=0
      let symbol = 'circle';
      let label = `w = ${w}`;
      
      if (w > 0) {
        color = '#10b981'; // green for positive winding
        symbol = 'star';
        label = `w = +${w}`;
      } else if (w < 0) {
        color = '#ef4444'; // red for negative winding
        symbol = 'star';
        label = `w = ${w}`;
      }
      
      traces.push({
        x: zeroGroup.map(z => z.k[0]),
        y: zeroGroup.map(z => z.k[1]),
        mode: 'markers+text',
        type: 'scatter',
        marker: {
          size: 13,
          color: color,
          symbol: symbol,
          line: { color: '#0f172a', width: 1.5 }
        },
        text: zeroGroup.map(z => z.label ? `${z.label} (w=${w >= 0 ? '+' : ''}${w})` : label),
        textposition: 'bottom center',
        textfont: { size: 10, color: '#0f172a', weight: 'bold' },
        name: `특이점 (${label})`,
        customdata: zeroGroup.map(z => {
          const cont = (typeof z.projector_continuous !== 'undefined')
            ? (z.projector_continuous ? '연속 (rank-1)' : '불연속 (rank>1, 접촉/특이)')
            : ((z.first_order && z.first_order.rank === 1) ? '연속' : '불연속');
          const ord = z.order ? z.order : 1;
          const lbl = z.label ? z.label : '—';
          const rr = (typeof z.rank_ratio === 'number') ? z.rank_ratio.toExponential(1) : '?';
          return `고대칭점: ${lbl}<br>k 좌표: (${z.k[0].toFixed(4)}, ${z.k[1].toFixed(4)})`
               + `<br>Winding 수: ${z.winding}<br>영점 차수(order): ${ord}`
               + `<br>사영자: ${cont}<br>loop rank-ratio: ${rr}`;
        }),
        hovertemplate:
          '<b>특이점 (vortex)</b><br>' +
          '%{customdata}' +
          '<extra></extra>',
        showlegend: true
      });
    });
  }
  
  const maxVal = Math.max(
    ...polyX.map(Math.abs),
    ...polyY.map(Math.abs),
    Math.abs(b1[0]), Math.abs(b1[1]),
    Math.abs(b2[0]), Math.abs(b2[1])
  );
  const pad = maxVal * 1.25;
  
  const plotTitle = viewMode === 'amp'
    ? `평탄 밴드 #${fb.band_index} BZ 특이점 맵 (|f(k)|)`
    : `평탄 밴드 #${fb.band_index} BZ 위상 소용돌이 맵 (arg(f))`;
    
  Plotly.newPlot(container, traces, {
    title: { text: plotTitle, font: { size: 12, weight: 'bold' } },
    xaxis: { title: 'k_x', range: [-pad, pad], scaleanchor: 'y', gridcolor: '#e2e8f0', zeroline: false },
    yaxis: { title: 'k_y', range: [-pad, pad], gridcolor: '#e2e8f0', zeroline: false },
    margin: { t: 40, l: 40, r: 10, b: 40 },
    plot_bgcolor:  '#f8fafc',
    paper_bgcolor: '#fff',
    showlegend: true,
    legend: { x: 0.95, xanchor: 'right', y: 0.95, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#e2e8f0', borderwidth: 1, font: { size: 9 } },
    dragmode: 'pan',
    annotations: annotations,
    font: { family: 'Segoe UI, sans-serif' }
  }, { responsive: true, scrollZoom: true, displayModeBar: true });
}

function initProjectorAnalysis(fb, card, orbLabels, cardIdx) {
  const quantitySelect = card.querySelector('.proj-quantity');
  const modeSelect = card.querySelector('.proj-plot-mode');
  const orb1Select = card.querySelector('.proj-orb-1');
  const orb2Select = card.querySelector('.proj-orb-2');
  const singSelectDiv = card.querySelector('.proj-singularity-select-div');
  const singSelect = card.querySelector('.proj-singularity-select');

  // Check if projector data exists in grid_data (robustness against old server runs)
  if (!fb.chern || !fb.chern.grid_data || !fb.chern.grid_data.P_real || !fb.chern.grid_data.P_imag) {
    const projSection = card.querySelector('.projector-analysis-section');
    if (projSection) {
      projSection.innerHTML = `
        <div style="padding: 1rem; background: #fffaf0; border: 1.5px solid #feebc8; border-radius: 8px; color: #c05621; font-size: 0.8rem; line-height: 1.6; margin-top: 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
          <strong>⚠️ 백엔드 사영자 데이터 누락</strong><br>
          해밀토니안 분석 서버가 사영자 계산 코드가 반영되기 이전 버전으로 백그라운드에서 계속 실행 중입니다.<br>
          <ol style="margin: 0.3rem 0; padding-left: 1.2rem;">
            <li>실행 중인 서버 터미널 창으로 이동합니다.</li>
            <li>터미널에서 <code>Ctrl+C</code>를 눌러 서버를 강제 종료합니다.</li>
            <li>프로젝트 디렉토리의 <code>start.bat</code>(또는 <code>start.sh</code>)을 다시 실행하여 서버를 재시작해 주세요.</li>
          </ol>
          그 후 브라우저를 새로고침하고 다시 <strong>[CLS 전체 분석]</strong> 버튼을 누르면 정상적으로 작동합니다.
        </div>
      `;
    }
    return;
  }

  // 1. Populate orbitals
  const Q = fb.chern.grid_data.P_real.length;
  orb1Select.innerHTML = '';
  orb2Select.innerHTML = '';
  for (let a = 0; a < Q; a++) {
    const label = orbLabels[a] || `Orbital ${a}`;
    const opt1 = document.createElement('option');
    opt1.value = a;
    opt1.textContent = label;
    const opt2 = document.createElement('option');
    opt2.value = a;
    opt2.textContent = label;
    orb1Select.appendChild(opt1);
    orb2Select.appendChild(opt2);
  }

  // Pre-select dominant orbital
  const domIdx = fb.chern.grid_data.dom_orbital || 0;
  orb1Select.value = domIdx;
  orb2Select.value = domIdx;

  // 2. Populate singularities
  let hasZeros = false;
  if (fb.chern.analytic && fb.chern.analytic.per_zero && fb.chern.analytic.per_zero.length > 0) {
    hasZeros = true;
    singSelect.innerHTML = '';
    fb.chern.analytic.per_zero.forEach((z, zi) => {
      const opt = document.createElement('option');
      opt.value = zi;
      const lbl = z.label ? `${z.label} ` : '';
      const ordStr = (z.order && z.order > 1) ? `, order=${z.order}` : '';
      opt.textContent = `${lbl}k0[${zi}] = (${z.k[0].toFixed(3)}, ${z.k[1].toFixed(3)}) (w = ${z.winding}${ordStr})`;
      singSelect.appendChild(opt);
    });
  } else {
    // Disable 1D cut if no singularities
    const cutOpt = modeSelect.querySelector('option[value="cut"]');
    if (cutOpt) {
      cutOpt.disabled = true;
      cutOpt.textContent += ' (특이점 없음)';
    }
  }

  // 3. Event listeners
  const onChange = () => {
    const quantity = quantitySelect.value;
    const mode = modeSelect.value;

    // Show/hide singularity select
    if (mode === 'cut' && hasZeros) {
      singSelectDiv.style.display = 'flex';
    } else {
      singSelectDiv.style.display = 'none';
    }

    // Disable orbitals for Berry Curvature
    if (quantity === 'berry') {
      orb1Select.disabled = true;
      orb2Select.disabled = true;
    } else {
      orb1Select.disabled = false;
      orb2Select.disabled = false;
    }

    updateProjectorPlot(fb, card, orbLabels, cardIdx);
  };

  quantitySelect.addEventListener('change', onChange);
  modeSelect.addEventListener('change', onChange);
  orb1Select.addEventListener('change', onChange);
  orb2Select.addEventListener('change', onChange);
  singSelect.addEventListener('change', onChange);

  // Initial render
  setTimeout(() => {
    onChange();
  }, 100);
}

function updateProjectorPlot(fb, card, orbLabels, cardIdx) {
  const container = card.querySelector('.proj-plot-container');
  if (!container || !fb || !fb.chern || !fb.chern.grid_data) return;

  const grid_data = fb.chern.grid_data;
  if (!grid_data.P_real || !grid_data.P_imag) return;
  const quantitySelect = card.querySelector('.proj-quantity');
  const modeSelect = card.querySelector('.proj-plot-mode');
  const orb1Select = card.querySelector('.proj-orb-1');
  const orb2Select = card.querySelector('.proj-orb-2');
  const singSelect = card.querySelector('.proj-singularity-select');
  const interpContent = card.querySelector('.proj-interpretation-content');

  const quantity = quantitySelect.value;
  const plotMode = modeSelect.value;
  const orb1 = parseInt(orb1Select.value);
  const orb2 = parseInt(orb2Select.value);

  const x_ticks = grid_data.x;
  const y_ticks = grid_data.y;
  const n_x = x_ticks.length;
  const n_y = y_ticks.length;

  // 1. Calculate target zData (n_y, n_x)
  let zData = [];
  let quantityLabel = '';
  let colorScale = 'Viridis';

  if (quantity === 'berry') {
    zData = grid_data.berry_curvature;
    quantityLabel = 'Berry Curvature F_xy(k)';
    colorScale = 'RdBu'; // Symmetric diverging
  } else {
    const P_real = grid_data.P_real[orb1][orb2];
    const P_imag = grid_data.P_imag[orb1][orb2];
    
    // Allocate 2D array
    zData = Array.from({ length: n_y }, () => new Array(n_x).fill(0));
    
    for (let r = 0; r < n_y; r++) {
      for (let c = 0; c < n_x; c++) {
        const re = P_real[r][c];
        const im = P_imag[r][c];
        if (quantity === 'mag') {
          zData[r][c] = Math.hypot(re, im);
        } else if (quantity === 'real') {
          zData[r][c] = re;
        } else if (quantity === 'imag') {
          zData[r][c] = im;
        } else if (quantity === 'phase') {
          zData[r][c] = Math.atan2(im, re);
        }
      }
    }

    const orb1Label = orbLabels[orb1] || `Orbital ${orb1}`;
    const orb2Label = orbLabels[orb2] || `Orbital ${orb2}`;
    
    if (quantity === 'mag') {
      quantityLabel = `|P_{${orb1Label}, ${orb2Label}}(k)|`;
    } else if (quantity === 'real') {
      quantityLabel = `Re P_{${orb1Label}, ${orb2Label}}(k)`;
      colorScale = 'RdBu';
    } else if (quantity === 'imag') {
      quantityLabel = `Im P_{${orb1Label}, ${orb2Label}}(k)`;
      colorScale = 'RdBu';
    } else if (quantity === 'phase') {
      quantityLabel = `arg P_{${orb1Label}, ${orb2Label}}(k) (rad)`;
      colorScale = 'HSV';
    }
  }

  // 2. Build Plotly data
  const traces = [];
  const bzData = fb.bz_plot || (fb.cls && fb.cls.bz_plot);

  if (plotMode === 'contour') {
    // 2D Contour
    traces.push({
      x: x_ticks,
      y: y_ticks,
      z: zData,
      type: 'contour',
      colorscale: colorScale,
      line: { width: 0 },
      contours: { coloring: 'heatmap' },
      showscale: true,
      colorbar: {
        title: { text: quantityLabel, font: { size: 10, weight: 'bold' } },
        thickness: 12,
        len: 0.8,
        y: 0.5,
        tickvals: quantity === 'phase' ? [-Math.PI, -Math.PI/2, 0, Math.PI/2, Math.PI] : undefined,
        ticktext: quantity === 'phase' ? ['-π', '-π/2', '0', 'π/2', 'π'] : undefined
      },
      hoverinfo: 'x+y+z',
      name: quantityLabel
    });

    // Overlay BZ Boundary
    if (bzData) {
      const polyX = bzData.vertices.map(v => v[0]);
      const polyY = bzData.vertices.map(v => v[1]);
      traces.push({
        x: polyX,
        y: polyY,
        mode: 'lines',
        type: 'scatter',
        line: { color: '#0f172a', width: 2 },
        name: '1st BZ 경계',
        hoverinfo: 'none',
        showlegend: true
      });
      
      // Reciprocal vectors
      const b1 = bzData.recip_vectors[0];
      const b2 = bzData.recip_vectors[1];
      traces.push({
        x: [0, b1[0], null, 0, b2[0]],
        y: [0, b1[1], null, 0, b2[1]],
        mode: 'lines',
        type: 'scatter',
        line: { color: '#475569', width: 1.5, dash: 'dash' },
        name: '역격자 벡터',
        hoverinfo: 'none',
        showlegend: true
      });
    }

    // Overlay singularities
    if (fb.chern.analytic && fb.chern.analytic.per_zero && fb.chern.analytic.per_zero.length > 0) {
      const zeros = fb.chern.analytic.per_zero;
      traces.push({
        x: zeros.map(z => z.k[0]),
        y: zeros.map(z => z.k[1]),
        mode: 'markers',
        type: 'scatter',
        marker: {
          size: 10,
          color: '#ffffff',
          symbol: 'circle-open-dot',
          line: { color: '#000000', width: 2 }
        },
        name: '특이점 k₀',
        hovertemplate: '특이점 k: (%{x:.3f}, %{y:.3f})<extra></extra>',
        showlegend: true
      });
    }

    const pad = Math.max(...x_ticks.map(Math.abs)) * 1.05;
    
    Plotly.newPlot(container, traces, {
      xaxis: { title: 'k_x', range: [-pad, pad], scaleanchor: 'y', gridcolor: '#e2e8f0', zeroline: false },
      yaxis: { title: 'k_y', range: [-pad, pad], gridcolor: '#e2e8f0', zeroline: false },
      margin: { t: 30, l: 40, r: 10, b: 40 },
      plot_bgcolor:  '#f8fafc',
      paper_bgcolor: '#fff',
      showlegend: true,
      legend: { x: 0.95, xanchor: 'right', y: 0.95, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#e2e8f0', borderwidth: 1, font: { size: 9 } },
      dragmode: 'pan',
      font: { family: 'Segoe UI, sans-serif' }
    }, { responsive: true, scrollZoom: true, displayModeBar: true });

  } else if (plotMode === 'surface') {
    // 3D Surface
    traces.push({
      x: x_ticks,
      y: y_ticks,
      z: zData,
      type: 'surface',
      colorscale: colorScale,
      showscale: true,
      colorbar: {
        title: { text: quantityLabel, font: { size: 10, weight: 'bold' } },
        thickness: 12,
        len: 0.7,
        y: 0.5,
        tickvals: quantity === 'phase' ? [-Math.PI, -Math.PI/2, 0, Math.PI/2, Math.PI] : undefined,
        ticktext: quantity === 'phase' ? ['-π', '-π/2', '0', 'π/2', 'π'] : undefined
      },
      name: quantityLabel
    });

    Plotly.newPlot(container, traces, {
      scene: {
        xaxis: { title: 'k_x', gridcolor: '#e2e8f0' },
        yaxis: { title: 'k_y', gridcolor: '#e2e8f0' },
        zaxis: { title: quantityLabel, gridcolor: '#e2e8f0' },
        camera: { eye: { x: 1.3, y: 1.3, z: 1.1 } }
      },
      margin: { t: 35, l: 10, r: 10, b: 15 },
      paper_bgcolor: '#fff',
      font: { family: 'Segoe UI, sans-serif' }
    }, { responsive: true, displayModeBar: true });

  } else if (plotMode === 'cut') {
    // 1D Cut
    const zIdx = parseInt(singSelect.value) || 0;
    const zInfo = fb.chern.analytic.per_zero[zIdx];
    if (!zInfo) return;

    const k0_x = zInfo.k[0];
    const k0_y = zInfo.k[1];

    let closest_x_idx = 0;
    let min_dx = Infinity;
    x_ticks.forEach((val, idx) => {
      const diff = Math.abs(val - k0_x);
      if (diff < min_dx) {
        min_dx = diff;
        closest_x_idx = idx;
      }
    });

    let closest_y_idx = 0;
    let min_dy = Infinity;
    y_ticks.forEach((val, idx) => {
      const diff = Math.abs(val - k0_y);
      if (diff < min_dy) {
        min_dy = diff;
        closest_y_idx = idx;
      }
    });

    const cut_x_values = x_ticks.map(v => v - k0_x);
    const cut_x_z = zData[closest_y_idx];

    const cut_y_values = y_ticks.map(v => v - k0_y);
    const cut_y_z = zData.map(row => row[closest_x_idx]);

    traces.push({
      x: cut_x_values,
      y: cut_x_z,
      mode: 'lines+markers',
      type: 'scatter',
      name: `k_x 방향 cut (k_y ≈ ${k0_y.toFixed(3)})`,
      line: { color: '#3b82f6', width: 2 },
      marker: { size: 4 }
    });

    traces.push({
      x: cut_y_values,
      y: cut_y_z,
      mode: 'lines+markers',
      type: 'scatter',
      name: `k_y 방향 cut (k_x ≈ ${k0_x.toFixed(3)})`,
      line: { color: '#ef4444', width: 2 },
      marker: { size: 4 }
    });

    traces.push({
      x: [0, 0],
      y: [Math.min(...cut_x_z, ...cut_y_z), Math.max(...cut_x_z, ...cut_y_z)],
      mode: 'lines',
      type: 'scatter',
      line: { color: '#64748b', width: 1, dash: 'dash' },
      name: '특이점 k₀ 위치 (q=0)',
      hoverinfo: 'none',
      showlegend: true
    });

    Plotly.newPlot(container, traces, {
      xaxis: { title: '특이점으로부터의 변위 q (k_x-k_{0x} 또는 k_y-k_{0y})', gridcolor: '#e2e8f0' },
      yaxis: { title: quantityLabel, gridcolor: '#e2e8f0' },
      margin: { t: 30, l: 50, r: 15, b: 45 },
      plot_bgcolor:  '#f8fafc',
      paper_bgcolor: '#fff',
      showlegend: true,
      legend: { x: 0.95, xanchor: 'right', y: 0.95, bgcolor: 'rgba(255,255,255,0.85)', bordercolor: '#e2e8f0', borderwidth: 1, font: { size: 9 } },
      font: { family: 'Segoe UI, sans-serif' }
    }, { responsive: true, displayModeBar: true });
  }

  // 3. Update interpretation guidelines
  let html = '';
  if (quantity === 'berry') {
    html = `
      <div style="font-weight: bold; margin-bottom: 0.3rem; color: #581c87; font-size: 0.8rem;">베리 곡률 F_xy(k) 해석</div>
      BZ 각 지점에서의 Berry Phase 곡률 밀도입니다. 이 값을 BZ에 대해 전역적으로 면적 적분(2D BZ)하고 $2\\pi$로 나누면 <strong>Chern 수(Chern Number)</strong>가 얻어집니다.
      <br><br>
      <span style="color:#7c3aed;"><strong>✓ 확인 포인트:</strong></span>
      <ul style="padding-left: 1.2rem; margin: 0.3rem 0;">
        <li>베리 곡률이 어디에 피크를 형성하고 집중되는지 확인하세요.</li>
        <li>보통 밴드 접촉이 해제되면서 갭이 열린 영역이나 특이점 $k_0$ 근처에서 곡률이 강하게 피크를 이룹니다.</li>
        <li>곡률의 부호(양/음)와 대칭성을 관찰해보세요.</li>
      </ul>
    `;
  } else {
    const isSingular = fb.singular === true;
    const orb1Label = orbLabels[orb1] || `Orbital ${orb1}`;
    const orb2Label = orbLabels[orb2] || `Orbital ${orb2}`;
    
    html = `
      <div style="font-weight: bold; margin-bottom: 0.3rem; color: #581c87; font-size: 0.8rem;">사영자 성분 P_{${orb1Label}, ${orb2Label}}(k) 해석</div>
      오비탈 ${orb1Label}와 ${orb2Label} 간의 평탄 밴드 사영자 행렬 성분입니다.
      <br><br>
    `;

    if (isSingular) {
      html += `
        이 평탄 밴드는 <strong>특이형 (Singular)</strong> 모델입니다. 즉, 공통 영점 $k_0$에서 Bloch 파동함수가 잘 정의되지 않고 불연속입니다.
        <br><br>
        <span style="color:#7c3aed;"><strong>✓ 투영자 불연속성 검증:</strong></span>
        <ul style="padding-left: 1.2rem; margin: 0.3rem 0;">
          <li><strong>3D Surface</strong> 모드에서 특이점 $k_0$ 주변을 회전시키며 관찰해보세요.</li>
          <li>만약 사영자 성분이 특이점에서 <strong>불연속(Rank A = 2)</strong>하다면, $k_0$ 근처에서 접근 방향에 따라 값이 급격히 변해 <strong>나선형 계단(Helicoid)</strong>이나 <strong>칼로 자른 듯한 절벽</strong> 형태를 보입니다.</li>
          <li><strong>1D Cut</strong> 모드로 전환하고, $k_x$ 방향 단면(파란색)과 $k_y$ 방향 단면(빨간색)을 관찰해보세요.</li>
          <li>두 곡선이 변위 $q=0$(특이점 위치)에서 **서로 만나지 않고 어긋난다면**, 이는 극한값이 방향에 의존하여 사영자가 불연속적임을 직접 증명합니다.</li>
        </ul>
      `;
    } else {
      html += `
        이 평탄 밴드는 <strong>비특이형 (Non-Singular)</strong> 모델입니다. 파동함수와 사영자 $P(k)$가 BZ 전체에서 완전히 매끄럽고 연속적으로 정의됩니다.
        <br><br>
        <span style="color:#7c3aed;"><strong>✓ 투영자 연속성 확인:</strong></span>
        <ul style="padding-left: 1.2rem; margin: 0.3rem 0;">
          <li><strong>3D Surface</strong> 모드에서 곡면이 뾰족하거나 찢어진 절벽 없이 <strong>완전하고 부드러운 돔 또는 분지(valley)</strong> 모양인지 확인하세요.</li>
          <li><strong>1D Cut</strong> 모드에서 $k_x$ 방향 단면(파란색)과 $k_y$ 방향 단면(빨간색)을 확인하면, 두 곡선이 변위 $q=0$에서 **정확히 동일한 값으로 부드럽게 수렴**합니다.</li>
          <li>이것은 공통 영점 유무와 무관하게 <strong>사영자 $P(k)$가 위상적으로 연속적이고 잘 정의됨</strong>을 의미하며, Chern 수 계산이 위상적으로 잘 성립하는 전제조건입니다.</li>
        </ul>
      `;
    }
  }
  interpContent.innerHTML = html;
}

const ORBITAL_SYMBOLS = ['circle', 'square', 'diamond', 'triangle-up', 'triangle-down', 'pentagon', 'hexagon', 'star', 'cross', 'x', 'hourglass', 'bowtie'];

function mapSvgShapeToPlotly(shape) {
  if (shape === 'triangle') return 'triangle-up';
  return shape || 'circle';
}

// Sublattice color palette — distinct pastel colors for each sublattice
const SUBLATTICE_COLORS = [
  { bg: '#93c5fd', border: '#3b82f6', name: 'blue' },     // sublattice 0
  { bg: '#fca5a5', border: '#ef4444', name: 'red' },      // sublattice 1
  { bg: '#86efac', border: '#22c55e', name: 'green' },    // sublattice 2
  { bg: '#fcd34d', border: '#f59e0b', name: 'amber' },    // sublattice 3
  { bg: '#c4b5fd', border: '#8b5cf6', name: 'violet' },   // sublattice 4
  { bg: '#fdba74', border: '#f97316', name: 'orange' },    // sublattice 5
  { bg: '#67e8f9', border: '#06b6d4', name: 'cyan' },     // sublattice 6
  { bg: '#f9a8d4', border: '#ec4899', name: 'pink' },     // sublattice 7
];

function getSublatticeColor(subIdx) {
  return SUBLATTICE_COLORS[subIdx % SUBLATTICE_COLORS.length];
}

function getOrbitalSymbol(orbitalInSublattice) {
  return ORBITAL_SYMBOLS[orbitalInSublattice % ORBITAL_SYMBOLS.length];
}

function getConvexHull(points) {
  if (points.length <= 1) return points;
  if (points.length === 2) return [points[0], points[1], points[0]];
  
  // Sort by x, then y
  const sorted = points.slice().sort((a, b) => a.x !== b.x ? a.x - b.x : a.y - b.y);
  
  const lower = [];
  for (let i = 0; i < sorted.length; i++) {
    while (lower.length >= 2 && crossProduct(lower[lower.length - 2], lower[lower.length - 1], sorted[i]) <= 0) {
      lower.pop();
    }
    lower.push(sorted[i]);
  }
  
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    while (upper.length >= 2 && crossProduct(upper[upper.length - 2], upper[upper.length - 1], sorted[i]) <= 0) {
      upper.pop();
    }
    upper.push(sorted[i]);
  }
  
  lower.pop();
  upper.pop();
  return lower.concat(upper).concat([lower[0]]);
}

function crossProduct(a, b, c) {
  return (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
}

function getPhaseColor(re, im) {
  const r = Math.hypot(re, im);
  if (r < 1e-9) return '#cbd5e1';
  const phi = Math.atan2(im, re); // -pi to pi
  let deg = phi * 180 / Math.PI;
  if (deg < 0) deg += 360;
  return `hsl(${deg.toFixed(1)}, 85%, 50%)`;
}

function getPhaseColorDark(re, im) {
  const r = Math.hypot(re, im);
  if (r < 1e-9) return '#475569';
  const phi = Math.atan2(im, re);
  let deg = phi * 180 / Math.PI;
  if (deg < 0) deg += 360;
  return `hsl(${deg.toFixed(1)}, 90%, 25%)`;
}

// Display-mode-aware colour: amplitude mode uses red/blue for ±, others use phase hue
function getDisplayColor(re, im, displayMode) {
  if (displayMode === 'amplitude') {
    if (Math.hypot(re, im) < 1e-9) return '#cbd5e1';
    return re >= 0 ? '#ef4444' : '#3b82f6';
  }
  return getPhaseColor(re, im);
}

// Patch base plot-data site amplitudes from a representation's serialised amps.
// Phase rotations never change which cells are non-zero, so this is always safe.
function applyReprAmplitudes(basePlotData, reprAmpData) {
  // Build lookup by orbital + integer cell.
  // repr amplitudes use integer cells (rounded by _ser_amp); sites use int_cell
  // (the integer unit-cell coordinate, stored since bridge.py fix).
  const lookup = {};
  for (const [qi, qdata] of Object.entries(reprAmpData)) {
    for (const a of qdata.amplitudes || []) {
      lookup[`${qi}_${a.cell.join(',')}`] = a;
    }
  }
  const newSites = basePlotData.sites.map(s => {
    // int_cell is the integer unit-cell coordinate; fall back to s.cell if absent
    const cellKey = (s.int_cell || s.cell).join(',');
    const key = `${s.orbital}_${cellKey}`;
    const amp = lookup[key];
    if (amp) return { ...s, is_cls: true, amplitude: amp.abs, amp_re: amp.re, amp_im: amp.im };
    if (s.is_cls) return { ...s, is_cls: false, amplitude: 0, amp_re: 0, amp_im: 0 };
    return s;
  });
  return { ...basePlotData, sites: newSites };
}

function modeLabel(m) {
  return m === 'amplitude' ? '진폭±' : m === 'phase' ? '위상' : '복소수';
}

function formatAmpLabel(s, displayMode = 'complex') {
  if (!s.is_cls) return '';
  const re = s.amp_re, im = s.amp_im;
  if (displayMode === 'amplitude') {
    if (Math.abs(im) < 1e-4) return re.toFixed(2);
    // Complex amplitude in amplitude mode: show magnitude with effective sign
    return (re >= 0 ? '+' : '-') + Math.hypot(re, im).toFixed(2);
  }
  if (Math.abs(im) < 1e-6) return re.toFixed(2);
  const abs = Math.hypot(re, im);
  const deg = (Math.atan2(im, re) * 180 / Math.PI + 360) % 360;
  return `${abs.toFixed(2)}∠${Math.round(deg)}°`;
}

function hexToRgba(hex, alpha) {
  if (!hex) return `rgba(100, 116, 139, ${alpha})`;
  hex = hex.replace('#', '');
  if (hex.length === 3) {
    hex = hex.split('').map(c => c + c).join('');
  }
  const r = parseInt(hex.substring(0, 2), 16);
  const g = parseInt(hex.substring(2, 4), 16);
  const b = parseInt(hex.substring(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// ─── Lattice + CLS Plot (Plotly) ───────────────────────────────────────────────
function renderLatticePlot(plotData, containerId, title, displayMode = 'complex') {
  const container = document.getElementById(containerId);
  if (!container || !plotData) return;

  const { sites, bonds, dimension: d, primitive_vectors } = plotData;
  if (!sites) return;

  const is3D = d === 3;
  const traces = [];

  // Ensure CLS sites have numeric amp fields (guard against null from original site dicts)
  sites.forEach(s => {
    if (s.is_cls) {
      s.amp_re = s.amp_re ?? 0;
      s.amp_im = s.amp_im ?? 0;
      s.amplitude = s.amplitude ?? Math.hypot(s.amp_re, s.amp_im);
    }
  });
  const maxAmp = Math.max(...sites.filter(s=>s.is_cls).map(s=>s.amplitude||0), 1e-9);

  if (is3D) {
    // 3D Sublattice links
    if (plotData.sublattice_links && plotData.sublattice_links.length > 0) {
      const linkX = [], linkY = [], linkZ = [];
      plotData.sublattice_links.forEach(l => {
        linkX.push(l.x0, l.x1, null);
        linkY.push(l.y0, l.y1, null);
        linkZ.push(l.z0, l.z1, null);
      });
      traces.push({
        type: 'scatter3d', mode: 'lines',
        x: linkX, y: linkY, z: linkZ,
        line: { color: '#cbd5e1', width: 1.5, dash: 'dash' },
        showlegend: false, hoverinfo: 'none'
      });
    }

    // 3D Sublattice centers
    if (plotData.sublattices && plotData.sublattices.length > 0) {
      const subX = plotData.sublattices.map(s => s.x);
      const subY = plotData.sublattices.map(s => s.y);
      const subZ = plotData.sublattices.map(s => s.z);
      traces.push({
        type: 'scatter3d', mode: 'markers',
        x: subX, y: subY, z: subZ,
        marker: { color: '#64748b', size: 3.5, symbol: 'circle' },
        name: '사이트 중심 (Sublattice)', showlegend: true, hoverinfo: 'none'
      });
    }

    // 3D bond lines (grouped into a single trace for high performance)
    const bondX = [], bondY = [], bondZ = [];
    bonds.forEach(b => {
      bondX.push(b.x0, b.x1, null);
      bondY.push(b.y0, b.y1, null);
      bondZ.push(b.z0, b.z1, null);
    });
    if (bonds.length > 0) {
      traces.push({
        type: 'scatter3d', mode: 'lines',
        x: bondX, y: bondY, z: bondZ,
        line: { color: '#cbd5e1', width: 2 }, showlegend: false,
        hoverinfo: 'none'
      });
    }

    // Background sites grouped by (sublattice, orbital)
    const bg = sites; // Draw ALL sites in background
    const hasSublatticeInfo3D = bg.length > 0 && bg[0].sublattice !== undefined;
    const isMultiOrbital3D = plotData.is_multi_orbital || false;

    if (hasSublatticeInfo3D) {
      const subGroups = {};
      bg.forEach(s => {
        const key = `${s.sublattice}_${s.orbital}`;
        if (!subGroups[key]) subGroups[key] = { sublattice: s.sublattice, orbital: s.orbital, sites: [] };
        subGroups[key].sites.push(s);
      });

      const sortedSubGroups3D = Object.values(subGroups).sort((a, b) => {
        const aOrb = a.sites[0].orbital_in_sublattice || 0;
        const bOrb = b.sites[0].orbital_in_sublattice || 0;
        return aOrb - bOrb;
      });

      sortedSubGroups3D.forEach(group => {
        const subIdx = group.sublattice;
        const subColor = getSublatticeColor(subIdx);
        const orbInSub = group.sites[0].orbital_in_sublattice || 0;
        const orbIndex = group.sites[0].orbital || 0;
        const symbol = mapSvgShapeToPlotly(latSt.orbShapes[orbIndex]);
        const label = group.sites[0].label;
        const subLabel = group.sites[0].sublattice_label || label;

        let legendName;
        if (isMultiOrbital3D) {
          legendName = `서브라티스 ${subLabel} · 오비탈 ${label}`;
        } else {
          legendName = `서브라티스 ${label}`;
        }

        const baseColor = latSt.orbColors[orbIndex] || subColor.border;
        const fillColor = orbInSub % 2 === 1 ? '#ffffff' : hexToRgba(baseColor, 0.25);

        traces.push({
          type: 'scatter3d', mode: 'markers',
          x: group.sites.map(s=>s.x), y: group.sites.map(s=>s.y), z: group.sites.map(s=>s.z),
          marker: { 
            symbol: symbol, 
            color: fillColor, 
            size: Math.max(7.5 - 2.0 * orbInSub, 3.5), 
            opacity: 0.65, 
            line: { color: baseColor, width: 1.0 } 
          },
          name: legendName, hoverinfo: 'none', showlegend: true
        });
      });
    } else {
      const bgGroups = {};
      bg.forEach(s => {
        if (!bgGroups[s.orbital]) bgGroups[s.orbital] = [];
        bgGroups[s.orbital].push(s);
      });
      Object.keys(bgGroups).forEach(qStr => {
        const q = parseInt(qStr);
        const group = bgGroups[qStr];
        const label = group[0].label;
        const symbol = mapSvgShapeToPlotly(latSt.orbShapes[q]);
        const baseColor = latSt.orbColors[q] || '#cbd5e1';
        const borderColor = latSt.orbColors[q] || '#94a3b8';

        traces.push({
          type: 'scatter3d', mode: 'markers',
          x: group.map(s=>s.x), y: group.map(s=>s.y), z: group.map(s=>s.z),
          marker: { 
            symbol: symbol, 
            color: latSt.orbColors[q] ? hexToRgba(baseColor, 0.25) : '#cbd5e1', 
            size: 4, 
            opacity: 0.65, 
            line: { color: borderColor, width: 1.0 } 
          },
          name: `오비탈 ${label} (배경)`, hoverinfo: 'none', showlegend: true
        });
      });
    }

    // CLS sites
    const cls = sites.filter(s => s.is_cls);
    if (cls.length > 0) {
      const clsX = cls.map(s => s.x);
      const clsY = cls.map(s => s.y);
      const clsZ = cls.map(s => s.z);
      const clsColors = cls.map(s => getDisplayColor(s.amp_re, s.amp_im, displayMode));
      const clsSizes = cls.map(s => 10 + 16 * (s.amplitude / maxAmp));
      const clsTexts = cls.map(s => formatAmpLabel(s, displayMode));
      const clsHoverTexts = cls.map(s => {
        const polar = formatAmpLabel(s, displayMode);
        const cartesian = Math.abs(s.amp_im) < 1e-6
          ? `${s.amp_re.toFixed(4)}`
          : `${s.amp_re.toFixed(4)} + ${s.amp_im.toFixed(4)}i`;
        const subInfo = s.sublattice !== undefined
          ? `<br>서브라티스: ${s.sublattice_label || s.sublattice}`
          : '';
        return `${s.label}(${s.cell.join(',')})${subInfo}<br>값: ${cartesian}<br>극좌표: ${polar}`;
      });

      traces.push({
        type: 'scatter3d', mode: 'markers+text',
        x: clsX, y: clsY, z: clsZ,
        marker: { 
          symbol: 'circle', // Always sphere for overlay bubble
          color: clsColors, 
          size: clsSizes, 
          opacity: 0.55, // Translucent sphere
          line: { color: '#0f172a', width: 1.5 } 
        },
        text: clsTexts,
        textposition: 'top center',
        textfont: { size: 9, color: '#0f172a', weight: 'bold' },
        name: 'CLS 성분',
        customdata: clsHoverTexts,
        hovertemplate: '%{customdata}<extra></extra>',
        showlegend: true
      });
    }

    Plotly.newPlot(container, traces, {
      title: { text: title, font: { size: 13 } },
      margin: { t: 40, l: 0, r: 0, b: 0 },
      scene: { aspectmode: 'data',
               xaxis: { title: 'x' }, yaxis: { title: 'y' }, zaxis: { title: 'z' } },
      paper_bgcolor: '#fff', showlegend: true,
      legend: { x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(255,255,255,0.8)' },
      dragmode: 'orbit',
      font: { family: 'Segoe UI, sans-serif' }
    }, { responsive: true, scrollZoom: true, displayModeBar: true });

  } else {
    // 2D grid lines representing unit cells
    if (d === 2 && primitive_vectors && primitive_vectors.length >= 2) {
      const gridX = [], gridY = [];
      const a1 = primitive_vectors[0];
      const a2 = primitive_vectors[1];
      
      const cells = sites.map(s => s.cell);
      const n1s = cells.map(c => c[0]);
      const n2s = cells.map(c => c[1]);
      const minN1 = Math.min(...n1s) - 1;
      const maxN1 = Math.max(...n1s) + 1;
      const minN2 = Math.min(...n2s) - 1;
      const maxN2 = Math.max(...n2s) + 1;

      // Lines along a1
      for (let n2 = minN2; n2 <= maxN2; n2++) {
        const x0 = minN1 * a1[0] + n2 * a2[0];
        const y0 = minN1 * a1[1] + n2 * a2[1];
        const x1 = maxN1 * a1[0] + n2 * a2[0];
        const y1 = maxN1 * a1[1] + n2 * a2[1];
        gridX.push(x0, x1, null);
        gridY.push(y0, y1, null);
      }
      // Lines along a2
      for (let n1 = minN1; n1 <= maxN1; n1++) {
        const x0 = n1 * a1[0] + minN2 * a2[0];
        const y0 = n1 * a1[1] + minN2 * a2[1];
        const x1 = n1 * a1[0] + maxN2 * a2[0];
        const y1 = n1 * a1[1] + maxN2 * a2[1];
        gridX.push(x0, x1, null);
        gridY.push(y0, y1, null);
      }
      
      traces.push({
        x: gridX, y: gridY,
        mode: 'lines', type: 'scatter',
        line: { color: '#e2e8f0', width: 0.9, dash: 'dash' },
        name: '단위셀 경계', showlegend: true, hoverinfo: 'none'
      });
    }

    // 2D Sublattice links
    if (plotData.sublattice_links && plotData.sublattice_links.length > 0) {
      const linkX = [], linkY = [];
      plotData.sublattice_links.forEach(l => {
        linkX.push(l.x0, l.x1, null);
        linkY.push(l.y0, l.y1, null);
      });
      traces.push({
        x: linkX, y: linkY,
        mode: 'lines', type: 'scatter',
        line: { color: '#cbd5e1', width: 1.2, dash: 'dash' },
        name: '서브라티스 연결', showlegend: false, hoverinfo: 'none'
      });
    }

    // 2D Sublattice centers
    if (plotData.sublattices && plotData.sublattices.length > 0) {
      const subX = plotData.sublattices.map(s => s.x);
      const subY = plotData.sublattices.map(s => s.y);
      traces.push({
        x: subX, y: subY,
        mode: 'markers', type: 'scatter',
        marker: { color: '#64748b', size: 4, symbol: 'circle' },
        name: '사이트 중심 (Sublattice)', showlegend: true, hoverinfo: 'none'
      });
    }

    // 2D bond lines
    const bondX = [], bondY = [];
    bonds.forEach(b => {
      bondX.push(b.x0, b.x1, null);
      bondY.push(b.y0, b.y1, null);
    });
    if (bonds.length > 0) {
      traces.push({
        x: bondX, y: bondY,
        mode: 'lines', type: 'scatter',
        line: { color: '#cbd5e1', width: 1.2 }, showlegend: false, hoverinfo: 'none'
      });
    }

    // Shaded convex hull around CLS sites (only for 2D)
    const clsSites = sites.filter(s => s.is_cls);
    if (d === 2 && clsSites.length >= 2) {
      const pts = clsSites.map(s => ({ x: s.x, y: s.y }));
      const hull = getConvexHull(pts);
      traces.push({
        x: hull.map(p => p.x),
        y: hull.map(p => p.y),
        mode: 'lines',
        type: 'scatter',
        fill: 'toself',
        fillcolor: 'rgba(124, 77, 255, 0.04)',
        line: { color: 'rgba(124, 77, 255, 0.25)', width: 1.5, dash: 'dash' },
        name: 'CLS 영역',
        showlegend: true,
        hoverinfo: 'none'
      });
    }

    // Background sites grouped by (sublattice, orbital)
    const bgSites = sites; // Draw ALL sites in background
    const hasSublatticeInfo = bgSites.length > 0 && bgSites[0].sublattice !== undefined;
    const isMultiOrbital = plotData.is_multi_orbital || false;

    if (hasSublatticeInfo) {
      // Group by sublattice first, then by orbital within sublattice
      const subGroups = {};
      bgSites.forEach(s => {
        const key = `${s.sublattice}_${s.orbital}`;
        if (!subGroups[key]) subGroups[key] = { sublattice: s.sublattice, orbital: s.orbital, sites: [] };
        subGroups[key].sites.push(s);
      });

      const sortedSubGroups = Object.values(subGroups).sort((a, b) => {
        const aOrb = a.sites[0].orbital_in_sublattice || 0;
        const bOrb = b.sites[0].orbital_in_sublattice || 0;
        return aOrb - bOrb;
      });

      sortedSubGroups.forEach(group => {
        const subIdx = group.sublattice;
        const subColor = getSublatticeColor(subIdx);
        const orbInSub = group.sites[0].orbital_in_sublattice || 0;
        const orbIndex = group.sites[0].orbital || 0;
        const symbol = mapSvgShapeToPlotly(latSt.orbShapes[orbIndex]);
        const label = group.sites[0].label;
        const subLabel = group.sites[0].sublattice_label || label;

        // Build legend name
        let legendName;
        if (isMultiOrbital) {
          legendName = `서브라티스 ${subLabel} · 오비탈 ${label}`;
        } else {
          legendName = `서브라티스 ${label}`;
        }

        const baseColor = latSt.orbColors[orbIndex] || subColor.border;
        const fillColor = orbInSub % 2 === 1 ? '#ffffff' : hexToRgba(baseColor, 0.25);

        traces.push({
          x: group.sites.map(s => s.x),
          y: group.sites.map(s => s.y),
          mode: 'markers',
          type: 'scatter',
          marker: {
            symbol: symbol,
            color: fillColor,
            size: Math.max(13 - 3.5 * orbInSub, 4.5),
            line: { color: baseColor, width: 1.2 },
            opacity: 0.65
          },
          name: legendName,
          text: group.sites.map(s => {
            const cellStr = s.cell.join(',');
            return isMultiOrbital
              ? `${label}[sub:${subLabel}](${cellStr})`
              : `${label}(${cellStr})`;
          }),
          hovertemplate: '%{text}<extra></extra>',
          showlegend: true
        });
      });
    } else {
      // Fallback: group by orbital (old behavior)
      const bgGroups = {};
      bgSites.forEach(s => {
        if (!bgGroups[s.orbital]) bgGroups[s.orbital] = [];
        bgGroups[s.orbital].push(s);
      });

      Object.keys(bgGroups).forEach(qStr => {
        const q = parseInt(qStr);
        const group = bgGroups[qStr];
        const label = group[0].label;
        const symbol = mapSvgShapeToPlotly(latSt.orbShapes[q]);
        const baseColor = latSt.orbColors[q] || '#cbd5e1';
        const borderColor = latSt.orbColors[q] || '#94a3b8';

        traces.push({
          x: group.map(s => s.x),
          y: group.map(s => s.y),
          mode: 'markers',
          type: 'scatter',
          marker: {
            symbol: symbol,
            color: latSt.orbColors[q] ? hexToRgba(baseColor, 0.25) : '#cbd5e1',
            size: 6,
            line: { color: borderColor, width: 1.0 },
            opacity: 0.65
          },
          name: `오비탈 ${label} (배경)`,
          text: group.map(s => `${label}(${s.cell.join(',')})`),
          hovertemplate: '%{text}<extra></extra>',
          showlegend: true
        });
      });
    }

    // CLS sites in a single trace
    const layoutAnnotations = [];
    if (clsSites.length > 0) {
      const clsX = clsSites.map(s => s.x);
      const clsY = clsSites.map(s => s.y);
      const clsColors = clsSites.map(s => getDisplayColor(s.amp_re, s.amp_im, displayMode));
      const clsSizes = clsSites.map(s => 18 + 24 * (s.amplitude / maxAmp));

      // On-plot text: format depends on display mode
      const clsTexts = clsSites.map(s => formatAmpLabel(s, displayMode));
      
      const clsTextPositions = clsSites.map(s => {
        const orbInSub = s.orbital_in_sublattice !== undefined ? s.orbital_in_sublattice : s.orbital;
        if (orbInSub === 0) return 'top center';
        if (orbInSub === 1) return 'bottom center';
        if (orbInSub === 2) return 'middle right';
        return 'middle left';
      });

      const clsHoverTexts = clsSites.map(s => {
        const polar = formatAmpLabel(s, displayMode);
        const cartesian = Math.abs(s.amp_im) < 1e-6
          ? `${s.amp_re.toFixed(4)}`
          : `${s.amp_re.toFixed(4)} + ${s.amp_im.toFixed(4)}i`;
        const subInfo = s.sublattice !== undefined
          ? `<br>서브라티스: ${s.sublattice_label || s.sublattice}`
          : '';
        return `${s.label}(${s.cell.join(',')})${subInfo}<br>값: ${cartesian}<br>표시: ${polar}`;
      });
      
      traces.push({
        x: clsX,
        y: clsY,
        mode: 'markers+text',
        type: 'scatter',
        marker: {
          symbol: 'circle', // Always circle overlay
          color: clsColors,
          size: clsSizes,
          opacity: 0.55, // Translucent overlay bubble
          line: { color: '#0f172a', width: 1.5 }
        },
        text: clsTexts,
        textposition: clsTextPositions,
        textfont: { size: 9, color: '#0f172a', weight: 'bold' },
        name: 'CLS 성분',
        customdata: clsHoverTexts,
        hovertemplate: '%{customdata}<extra></extra>',
        showlegend: true
      });

      // Phase arrows: only shown in phase/complex modes (meaningless for pure amplitude)
      if (displayMode !== 'amplitude') clsSites.forEach(s => {
        if (s.amp_re !== undefined && s.amp_im !== undefined) {
          const re = s.amp_re;
          const im = s.amp_im;
          const amp = s.amplitude || 0;
          if (amp > 1e-5) {
            const theta = Math.atan2(im, re);
            
            // Offset coordinates to start outside the marker to prevent overlapping
            const L_start = 0.08 + 0.06 * (amp / maxAmp);
            const L_end = 0.32 + 0.20 * (amp / maxAmp);
            
            const x_tail = s.x + Math.max(L_start, 0.02) * Math.cos(theta);
            const y_tail = s.y + Math.max(L_start, 0.02) * Math.sin(theta);
            const x_tip = s.x + L_end * Math.cos(theta);
            const y_tip = s.y + L_end * Math.sin(theta);
            
            const arrowCol = getPhaseColorDark(re, im);
            
            layoutAnnotations.push({
              x: x_tip,
              y: y_tip,
              ax: x_tail,
              ay: y_tail,
              xref: 'x',
              yref: 'y',
              axref: 'x',
              ayref: 'y',
              showarrow: true,
              arrowhead: 2,
              arrowsize: 1.1,
              arrowwidth: 2.0,
              arrowcolor: arrowCol,
              text: ''
            });
          }
        }
      });
    }

    Plotly.newPlot(container, traces, {
      title: { text: title, font: { size: 13 } },
      xaxis: { title: '', scaleanchor: 'y', gridcolor: '#f8fafc', zeroline: false },
      yaxis: { title: '', gridcolor: '#f8fafc', zeroline: false },
      margin: { t: 40, l: 40, r: 20, b: 40 },
      plot_bgcolor:  '#fafbff',
      paper_bgcolor: '#fff',
      showlegend: true,
      legend: { x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(255,255,255,0.8)' },
      dragmode: 'pan',
      annotations: layoutAnnotations,
      font: { family: 'Segoe UI, sans-serif' }
    }, { responsive: true, scrollZoom: true, displayModeBar: true });
  }
}

function renderBZPlot(bzData, containerId, title) {
  const container = document.getElementById(containerId);
  if (!container || !bzData) return;
  
  const { dimension: d } = bzData;
  const traces = [];
  
  if (d === 1) {
    const val = bzData.vertices[1][0];
    traces.push({
      x: [-val, val],
      y: [0, 0],
      mode: 'lines+markers',
      type: 'scatter',
      line: { color: '#4c7aff', width: 4 },
      marker: { size: 10, color: ['#ef4444', '#ef4444'] },
      name: '1D BZ Boundary'
    });
    
    traces.push({
      x: [0],
      y: [0],
      mode: 'markers+text',
      type: 'scatter',
      marker: { size: 12, color: '#10b981' },
      text: ['Γ'],
      textposition: 'top center',
      textfont: { size: 12, weight: 'bold' },
      showlegend: false
    });
    
    if (bzData.sym_points) {
      const uniqueSyms = {};
      bzData.sym_points.forEach(p => {
        const key = `${p.x.toFixed(3)}`;
        if (!uniqueSyms[key]) uniqueSyms[key] = p;
      });
      const uSymList = Object.values(uniqueSyms);
      traces.push({
        x: uSymList.map(p => p.x),
        y: uSymList.map(() => 0),
        mode: 'markers+text',
        type: 'scatter',
        marker: { size: 8, color: '#ef4444' },
        text: uSymList.map(p => p.label),
        textposition: 'bottom center',
        textfont: { size: 10, weight: 'bold' },
        name: '대칭점',
        showlegend: true
      });
    }
    
    Plotly.newPlot(container, traces, {
      title: { text: title, font: { size: 13 } },
      xaxis: { title: 'k_x', range: [-val * 1.5, val * 1.5], gridcolor: '#f8fafc' },
      yaxis: { visible: false },
      margin: { t: 40, l: 40, r: 20, b: 40 },
      plot_bgcolor:  '#fafbff',
      paper_bgcolor: '#fff',
      showlegend: false,
      dragmode: 'pan',
      font: { family: 'Segoe UI, sans-serif' }
    }, { responsive: true, scrollZoom: true, displayModeBar: true });
    
  } else if (d === 2) {
    const polyX = bzData.vertices.map(v => v[0]);
    const polyY = bzData.vertices.map(v => v[1]);
    
    traces.push({
      x: polyX,
      y: polyY,
      mode: 'lines',
      type: 'scatter',
      fill: 'toself',
      fillcolor: 'rgba(76, 122, 255, 0.03)',
      line: { color: '#4c7aff', width: 2 },
      name: '1st BZ 경계',
      hoverinfo: 'none',
      showlegend: true
    });
    
    const b1 = bzData.recip_vectors[0];
    const b2 = bzData.recip_vectors[1];
    
    traces.push({
      x: [0, b1[0], null, 0, b2[0]],
      y: [0, b1[1], null, 0, b2[1]],
      mode: 'lines',
      type: 'scatter',
      line: { color: '#64748b', width: 2 },
      name: '역격자 벡터',
      hoverinfo: 'none',
      showlegend: true
    });
    
    const annotations = [
      {
        x: b1[0], y: b1[1], text: 'b₁', showarrow: false,
        xanchor: 'left', yanchor: 'bottom', font: { size: 12, color: '#475569', weight: 'bold' }
      },
      {
        x: b2[0], y: b2[1], text: 'b₂', showarrow: false,
        xanchor: 'left', yanchor: 'bottom', font: { size: 12, color: '#475569', weight: 'bold' }
      }
    ];
    
    if (bzData.sym_points && bzData.sym_points.length > 0) {
      const pathX = bzData.sym_points.map(p => p.x);
      const pathY = bzData.sym_points.map(p => p.y);
      
      traces.push({
        x: pathX,
        y: pathY,
        mode: 'lines+markers',
        type: 'scatter',
        line: { color: '#f97316', width: 1.5, dash: 'dash' },
        marker: { color: '#f97316', size: 5 },
        name: 'k-경로',
        hoverinfo: 'none',
        showlegend: true
      });
      
      const uniqueSyms = {};
      bzData.sym_points.forEach(p => {
        const key = `${p.x.toFixed(3)},${p.y.toFixed(3)}`;
        if (!uniqueSyms[key]) uniqueSyms[key] = p;
      });
      const uSymList = Object.values(uniqueSyms);
      
      traces.push({
        x: uSymList.map(p => p.x),
        y: uSymList.map(p => p.y),
        mode: 'markers+text',
        type: 'scatter',
        marker: { size: 8, color: '#ef4444', line: { color: '#fff', width: 1 } },
        text: uSymList.map(p => p.label),
        textposition: 'top right',
        textfont: { size: 10, color: '#1e293b', weight: 'bold' },
        name: '대칭점',
        hovertemplate: '대칭점: %{text}<br>k: (%{x:.3f}, %{y:.3f})<extra></extra>',
        showlegend: true
      });
    }
    
    const maxVal = Math.max(
      ...polyX.map(Math.abs),
      ...polyY.map(Math.abs),
      Math.abs(b1[0]), Math.abs(b1[1]),
      Math.abs(b2[0]), Math.abs(b2[1])
    );
    const pad = maxVal * 1.3;
    
    Plotly.newPlot(container, traces, {
      title: { text: title, font: { size: 13 } },
      xaxis: { title: 'k_x', range: [-pad, pad], scaleanchor: 'y', gridcolor: '#f1f5f9', zeroline: false },
      yaxis: { title: 'k_y', range: [-pad, pad], gridcolor: '#f1f5f9', zeroline: false },
      margin: { t: 40, l: 40, r: 20, b: 40 },
      plot_bgcolor:  '#fafbff',
      paper_bgcolor: '#fff',
      showlegend: true,
      legend: { x: 1, xanchor: 'right', y: 1, bgcolor: 'rgba(255,255,255,0.8)' },
      dragmode: 'pan',
      annotations,
      font: { family: 'Segoe UI, sans-serif' }
    }, { responsive: true, scrollZoom: true, displayModeBar: true });
    
  } else if (d === 3) {
    if (bzData.faces) {
      bzData.faces.forEach((face, fi) => {
        const faceX = face.map(v => v[0]);
        const faceY = face.map(v => v[1]);
        const faceZ = face.map(v => v[2]);
        
        traces.push({
          type: 'scatter3d',
          mode: 'lines',
          x: faceX,
          y: faceY,
          z: faceZ,
          line: { color: '#4c7aff', width: 2 },
          name: fi === 0 ? 'BZ 경계' : '',
          showlegend: fi === 0,
          hoverinfo: 'none'
        });
      });
    }
    
    const b1 = bzData.recip_vectors[0];
    const b2 = bzData.recip_vectors[1];
    const b3 = bzData.recip_vectors[2];
    
    traces.push({
      type: 'scatter3d',
      mode: 'lines',
      x: [0, b1[0], null, 0, b2[0], null, 0, b3[0]],
      y: [0, b1[1], null, 0, b2[1], null, 0, b3[1]],
      z: [0, b1[2], null, 0, b2[2], null, 0, b3[2]],
      line: { color: '#475569', width: 3 },
      name: '역격자 벡터',
      hoverinfo: 'none',
      showlegend: true
    });
    
    if (bzData.sym_points && bzData.sym_points.length > 0) {
      const pathX = bzData.sym_points.map(p => p.x);
      const pathY = bzData.sym_points.map(p => p.y);
      const pathZ = bzData.sym_points.map(p => p.z);
      
      traces.push({
        type: 'scatter3d',
        mode: 'lines+markers',
        x: pathX,
        y: pathY,
        z: pathZ,
        line: { color: '#f97316', width: 2, dash: 'dash' },
        marker: { color: '#f97316', size: 4 },
        name: 'k-경로',
        hoverinfo: 'none',
        showlegend: true
      });
      
      const uniqueSyms = {};
      bzData.sym_points.forEach(p => {
        const key = `${p.x.toFixed(3)},${p.y.toFixed(3)},${p.z.toFixed(3)}`;
        if (!uniqueSyms[key]) uniqueSyms[key] = p;
      });
      const uSymList = Object.values(uniqueSyms);
      
      traces.push({
        type: 'scatter3d',
        mode: 'markers+text',
        x: uSymList.map(p => p.x),
        y: uSymList.map(p => p.y),
        z: uSymList.map(p => p.z),
        marker: { size: 6, color: '#ef4444' },
        text: uSymList.map(p => p.label),
        textposition: 'top center',
        textfont: { size: 10, color: '#1e293b', weight: 'bold' },
        name: '대칭점',
        showlegend: true
      });
    }
    
    Plotly.newPlot(container, traces, {
      title: { text: title, font: { size: 13 } },
      margin: { t: 40, l: 0, r: 0, b: 0 },
      scene: { aspectmode: 'data',
               xaxis: { title: 'k_x' }, yaxis: { title: 'k_y' }, zaxis: { title: 'k_z' } },
      paper_bgcolor: '#fff',
      showlegend: true,
      legend: { x: 1, xanchor: 'right', y: 1 },
      font: { family: 'Segoe UI, sans-serif' }
    }, { responsive: true, displayModeBar: true });
  }
}

// Keep legacy color/label functions for backward compatibility if needed
function clsColor(re, im) {
  return getPhaseColor(re, im);
}

function ampLabel(s) {
  return formatAmpLabel(s);
}

// ─── Tab / Panel Helpers ──────────────────────────────────────────────────────
function showPanel(name) {
  document.querySelectorAll('.result-panel').forEach(p => {
    p.classList.toggle('active', false);
    p.classList.toggle('hidden', true);
  });
  const target = document.getElementById(`panel-${name}`);
  if (target) { target.classList.remove('hidden'); target.classList.add('active'); }
  document.querySelectorAll('.rtab').forEach(t =>
    t.classList.toggle('active', t.dataset.panel === name));
  if (name === 'lattice') {
    requestAnimationFrame(() => { buildSiteStylePanel(); latBuildSVG(); });
  }
  if (name === 'nanoribbon') {
    rebuildRibbonBandSelector();
  }
  if (name === 'engineer') {
    requestAnimationFrame(() => engBuildLatticeSVG());
  }
}

function activateResultsTab(name) {
  const btn = document.querySelector(`.rtab[data-panel="${name}"]`);
  if (btn) btn.style.fontWeight = '700';
}

// ─── Lattice UI ──────────────────────────────────────────────────────────────
function setDimension(d) {
  state.dimension = d;
  // Resize primitive vectors
  const spatialDim = d; // we keep spatial == lattice dim for simplicity
  state.primitiveVectors = Array.from({length: d}, (_, i) => {
    const row = Array(d).fill(0.0);
    row[i] = 1.0;
    return row;
  });
  // Trim orbital positions
  state.orbitals = state.orbitals.map(o => ({
    label: o.label,
    position: Array.from({length: d}, (_, i) => o.position[i] || 0.0)
  }));

  // Set default kPathStr for new dimension
  state.kPathStr = getDefaultKPath(d, state.primitiveVectors);
  const kpathInput = document.getElementById('param-kpath');
  if (kpathInput) {
    kpathInput.value = state.kPathStr;
  }

  // Set default kPointsOverride for new dimension
  state.kPointsOverride = getDefaultKPointsMap(d, state.primitiveVectors);
  rebuildKPointsOverrideUI();

  // Rebuild
  rebuildLatticeUI();
  rebuildHamiltonianEditor();

  // Update lattice info bar
  updateLatticeInfoBar();
}

function updateLatticeInfoBar() {
  const bar = document.getElementById('lattice-info-bar');
  const content = document.getElementById('lattice-info-content');
  if (!bar || !content) return;

  const spec = buildSpecSafe();
  if (!spec || !spec.lattice) {
    bar.style.display = 'none';
    return;
  }

  try {
    const info = detectLatticeType(spec.lattice);
    content.textContent = info;
    bar.style.display = 'flex';
  } catch (e) {
    bar.style.display = 'none';
  }
}

function buildSpecSafe() {
  try { return buildSpec(); } catch (_) { return null; }
}

function rebuildLatticeUI() {
  const d = state.dimension;

  document.querySelectorAll('.dim-btn').forEach(b => {
    b.classList.toggle('active', parseInt(b.dataset.d) === d);
  });

  // Primitive vectors with math expression inputs
  const pvUI = document.getElementById('prim-vecs-ui');
  pvUI.innerHTML = '';
  state.primitiveVectors.forEach((vec, li) => {
    const row = document.createElement('div');
    row.className = 'math-input-row';

    const lbl = document.createElement('span');
    lbl.className = 'vec-label';
    lbl.innerHTML = `a<sub>${li+1}</sub>`;
    row.appendChild(lbl);

    const compWrap = document.createElement('div');
    compWrap.style.cssText = 'display:flex;gap:.3rem;flex:1';

    vec.forEach((val, ci) => {
      const wrap = document.createElement('div');
      wrap.className = 'math-expr-wrap';

      const inp = document.createElement('input');
      inp.className = 'math-expr-input';
      inp.type = 'text';
      inp.value = String(val);
      inp.placeholder = 'e.g. sqrt(3)/2';

      const preview = document.createElement('div');
      preview.className = 'math-preview';

      const update = () => {
        try {
          const v = evalMathExpr(inp.value);
          state.primitiveVectors[li][ci] = v;
          preview.textContent = '= ' + (Number.isInteger(v) ? v : parseFloat(v.toFixed(5)));
          preview.className = 'math-preview eval-ok';
          inp.classList.remove('parse-error');
        } catch (_) {
          preview.textContent = 'error';
          preview.className = 'math-preview eval-err';
          inp.classList.add('parse-error');
        }
      };

      inp.addEventListener('input', update);
      update();

      wrap.appendChild(inp);
      wrap.appendChild(preview);
      compWrap.appendChild(wrap);
    });

    row.appendChild(compWrap);
    pvUI.appendChild(row);
  });

  rebuildOrbitalsUI();
  rebuildHamiltonianEditor();
}

function rebuildOrbitalsUI() {
  const orbUI = document.getElementById('orbitals-ui');
  orbUI.innerHTML = '';
  const d = state.dimension;
  const coordNames = ['x', 'y', 'z'];

  state.orbitals.forEach((orb, qi) => {
    const row = document.createElement('div');
    row.className = 'orbital-row-math';

    const labelInp = document.createElement('input');
    labelInp.type = 'text';
    labelInp.className = 'orbital-label-input input-num';
    labelInp.value = orb.label;
    labelInp.maxLength = 4;
    labelInp.addEventListener('change', e => {
      state.orbitals[qi].label = e.target.value;
      rebuildHamiltonianEditor();
    });
    row.appendChild(labelInp);

    const posDiv = document.createElement('div');
    posDiv.className = 'orb-positions';

    orb.position.forEach((val, di) => {
      const posWrap = document.createElement('div');
      posWrap.className = 'orb-pos-wrap';

      const coordLbl = document.createElement('div');
      coordLbl.className = 'orb-coord-label';
      coordLbl.textContent = coordNames[di] || di;

      const inp = document.createElement('input');
      inp.className = 'math-expr-input';
      inp.type = 'text';
      inp.value = String(val);
      inp.placeholder = '0';

      const preview = document.createElement('div');
      preview.className = 'math-preview';

      const update = () => {
        try {
          const v = evalMathExpr(inp.value);
          state.orbitals[qi].position[di] = v;
          const s = Math.abs(v) < 1e-10 ? '0' : parseFloat(v.toFixed(4));
          preview.textContent = '= ' + s;
          preview.className = 'math-preview eval-ok';
          inp.classList.remove('parse-error');
        } catch (_) {
          preview.textContent = '?';
          preview.className = 'math-preview eval-err';
          inp.classList.add('parse-error');
        }
      };
      inp.addEventListener('input', update);
      update();

      posWrap.appendChild(coordLbl);
      posWrap.appendChild(inp);
      posWrap.appendChild(preview);
      posDiv.appendChild(posWrap);
    });

    row.appendChild(posDiv);

    const delBtn = document.createElement('button');
    delBtn.className = 'btn-remove';
    delBtn.textContent = '×';
    delBtn.disabled = state.orbitals.length <= 1;
    delBtn.onclick = () => {
      if (state.orbitals.length <= 1) return;
      state.orbitals.splice(qi, 1);
      rebuildOrbitalsUI();
      rebuildHamiltonianEditor();
    };
    row.appendChild(delBtn);
    orbUI.appendChild(row);
  });
}

// ─── Hamiltonian Editor ───────────────────────────────────────────────────────
function rebuildHamiltonianEditor() {
  if (state.hamiltonianMode === 'hopping') {
    rebuildHoppingTable();
  } else {
    rebuildSymMatrix();
  }
  // Auto-detect parameters from expressions
  rebuildParameterPanel();
  
  updateHamiltonianMatrixPreview();
}

function rebuildHoppingTable() {
  const ui = document.getElementById('hop-cards-ui');
  if (!ui) return;
  ui.innerHTML = '';

  if (state.hoppings.length === 0) {
    addDefaultHop();
  } else {
    state.hoppings.forEach(hop => addHopCard(hop));
  }
}

function addDefaultHop() {
  addHopCard({ i: 0, j: 0, R: Array(state.dimension).fill(0), t: 1.0 });
}

function addHopCard(hop) {
  const ui = document.getElementById('hop-cards-ui');
  if (!ui) return;
  const d = state.dimension;

  // Convert t value to expression string
  let tExprStr = '1.0';
  if (typeof hop.t === 'string') {
    tExprStr = hop.t;
  } else if (typeof hop.t === 'object' && hop.t !== null && 're' in hop.t) {
    const re = hop.t.re || 0;
    const im = hop.t.im || 0;
    if (Math.abs(im) < 1e-9) {
      tExprStr = String(re);
    } else if (Math.abs(re) < 1e-9) {
      tExprStr = `I*${im}`;
    } else {
      tExprStr = `${re} + I*${im}`;
    }
  } else if (typeof hop.t === 'number') {
    tExprStr = String(hop.t);
  }

  const card = document.createElement('div');
  card.className = 'hop-card';

  // Header row: hop number + delete
  const head = document.createElement('div');
  head.className = 'hop-card-head';
  const numSpan = document.createElement('span');
  numSpan.className = 'hop-card-num';
  numSpan.textContent = `호핑 #${ui.children.length + 1}`;
  const delBtn = document.createElement('button');
  delBtn.className = 'btn-remove';
  delBtn.textContent = '×';
  delBtn.onclick = () => { card.remove(); rebuildParameterPanel(); };
  head.appendChild(numSpan);
  head.appendChild(delBtn);

  // KaTeX formula display
  const formulaDiv = document.createElement('div');
  formulaDiv.className = 'hop-formula';

  // Fields grid
  const fields = document.createElement('div');
  fields.className = 'hop-fields';

  // orbital selectors
  const makeOrbitSel = (selectedIdx) => {
    const sel = document.createElement('select');
    state.orbitals.forEach((o, qi) => {
      const opt = document.createElement('option');
      opt.value = qi; opt.textContent = `${qi}: ${o.label}`;
      if (qi === selectedIdx) opt.selected = true;
      sel.appendChild(opt);
    });
    return sel;
  };

  const selI = makeOrbitSel(hop.i || 0);
  const selJ = makeOrbitSel(hop.j || 0);
  selI.className = 'hop-sel-i';
  selJ.className = 'hop-sel-j';

  // R vector
  const rRow = document.createElement('div');
  rRow.className = 'hop-r-row';
  const rInputs = [];
  for (let di = 0; di < d; di++) {
    const inp = document.createElement('input');
    inp.type = 'number'; inp.step = '1';
    inp.value = (hop.R && hop.R[di] != null) ? hop.R[di] : 0;
    inp.className = 'hop-r-inp';
    rInputs.push(inp);
    rRow.appendChild(inp);
  }

  // Expression input (replaces separate Re/Im fields)
  const exprWrap = document.createElement('div');
  exprWrap.style.cssText = 'display:flex;flex-direction:column;gap:.15rem;width:100%';

  const inpExpr = document.createElement('input');
  inpExpr.type = 'text';
  inpExpr.className = 'hop-expr-input';
  inpExpr.value = tExprStr;
  inpExpr.placeholder = '예: 1.0, -0.5, t1, sqrt(2)/2, I*0.3';

  const exprPreview = document.createElement('div');
  exprPreview.className = 'hop-expr-preview';

  exprWrap.appendChild(inpExpr);
  exprWrap.appendChild(exprPreview);

  // Update formula display
  const updateFormula = () => {
    const iIdx = parseInt(selI.value);
    const jIdx = parseInt(selJ.value);
    const iLbl = state.orbitals[iIdx]?.label ?? iIdx;
    const jLbl = state.orbitals[jIdx]?.label ?? jIdx;
    const R = rInputs.map(inp => parseInt(inp.value) || 0);
    const exprVal = inpExpr.value.trim() || '0';

    // Determine display value for formula
    let displayT;
    try {
      const numVal = evalMathExpr(exprVal);
      displayT = numVal;
    } catch (_) {
      displayT = exprVal;
    }

    renderKatex(formulaDiv, hopToLatex(String(iLbl), String(jLbl), R, displayT));

    // Preview: show evaluated value or parameter hint
    const params = extractParameters(exprVal);
    if (params.length > 0) {
      inpExpr.classList.add('has-param');
      exprPreview.textContent = `파라미터: ${params.join(', ')}`;
      exprPreview.className = 'hop-expr-preview eval-ok';
    } else {
      inpExpr.classList.remove('has-param');
      try {
        const v = evalMathExpr(exprVal);
        exprPreview.textContent = `= ${parseFloat(v.toFixed(6))}`;
        exprPreview.className = 'hop-expr-preview eval-ok';
      } catch (_) {
        // Might be valid symbolic (e.g. I*0.3)
        exprPreview.textContent = '수식';
        exprPreview.className = 'hop-expr-preview eval-ok';
      }
    }
  };

  const updateWithParamRebuild = () => {
    updateFormula();
    rebuildParameterPanel();
    debouncedBandPreview();
  };

  [selI, selJ].forEach(el => el.addEventListener('input', updateFormula));
  rInputs.forEach(inp => inp.addEventListener('input', updateFormula));
  inpExpr.addEventListener('input', updateWithParamRebuild);

  // Build fields layout
  const addRow = (labelTxt, el) => {
    const lbl = document.createElement('label');
    lbl.textContent = labelTxt;
    fields.appendChild(lbl);
    fields.appendChild(el);
  };
  addRow('To (i):', selI);
  addRow('From (j):', selJ);
  addRow('R 벡터:', rRow);
  addRow('t 진폭:', exprWrap);

  card.appendChild(head);
  card.appendChild(formulaDiv);
  card.appendChild(fields);
  ui.appendChild(card);
  updateFormula();
}


function rebuildSymMatrix() {
  const Q = state.orbitals.length;
  const ui = document.getElementById('sym-matrix-ui');
  if (!ui) return;

  while (state.symbolicMatrix.length < Q)
    state.symbolicMatrix.push(Array(Q).fill('0'));
  state.symbolicMatrix = state.symbolicMatrix.slice(0, Q).map(row => {
    while (row.length < Q) row.push('0');
    return row.slice(0, Q);
  });

  const grid = document.createElement('div');
  grid.className = 'hmat-grid';
  grid.style.gridTemplateColumns = `repeat(${Q}, 140px)`;

  for (let r = 0; r < Q; r++) {
    for (let c = 0; c < Q; c++) {
      const isDiag = r === c;
      const isLower = r > c;
      const hermAuto = document.getElementById('herm-auto')?.checked ?? true;
      const isReadOnly = isLower && hermAuto;

      const cell = document.createElement('div');
      cell.className = 'hmat-cell' + (isDiag ? ' hmat-diag' : '') + (isLower ? ' hmat-conj' : '');

      const lbl = document.createElement('div');
      lbl.className = 'hmat-label';
      const iLbl = state.orbitals[r]?.label || r;
      const jLbl = state.orbitals[c]?.label || c;
      lbl.innerHTML = `H<sub>${iLbl}${jLbl}</sub>(k)`;

      if (!isReadOnly) {
        const editBtn = document.createElement('span');
        editBtn.className = 'cell-edit-btn';
        editBtn.innerHTML = '✏️';
        editBtn.title = '수식 편집기 열기';
        editBtn.addEventListener('click', () => openExpressionModal(r, c));
        lbl.appendChild(editBtn);
      }

      const inp = document.createElement('input');
      inp.type = 'text';
      inp.id = `sym-${r}-${c}`;
      inp.className = 'hmat-inp';
      inp.value = state.symbolicMatrix[r][c] || '0';
      inp.placeholder = isDiag ? 'E  (예: 2)' : isReadOnly ? '(자동)' : '수식...';
      if (isReadOnly) inp.readOnly = true;
      inp.addEventListener('focus', () => { focusedSymCell = inp; });
      if (!isReadOnly) {
        inp.addEventListener('dblclick', () => openExpressionModal(r, c));
      }

      const katexDiv = document.createElement('div');
      katexDiv.className = 'hmat-katex';

      if (!isReadOnly) {
        const makeUpdate = (row, col, inpEl, kEl) => () => {
          const val = inpEl.value || '0';
          state.symbolicMatrix[row][col] = val;
          renderKatex(kEl, exprToLatex(val));
          if (document.getElementById('herm-auto')?.checked) applyHermitianFill();
        };
        const upd = makeUpdate(r, c, inp, katexDiv);
        inp.addEventListener('input', () => {
          upd();
          rebuildParameterPanel();
          debouncedBandPreview();
        });
        upd();
      } else {
        renderKatex(katexDiv, exprToLatex(inp.value));
      }

      cell.appendChild(lbl);
      cell.appendChild(inp);
      cell.appendChild(katexDiv);
      grid.appendChild(cell);
    }
  }

  ui.innerHTML = '';
  ui.appendChild(grid);

  if (document.getElementById('herm-auto')?.checked) {
    applyHermitianFill();
  } else {
    updateHamiltonianMatrixPreview();
  }
}

function applyHermitianFill() {
  const Q = state.symbolicMatrix.length;
  for (let r = 0; r < Q; r++) {
    for (let c = 0; c < Q; c++) {
      if (r > c) {
        const upper = state.symbolicMatrix[c][r] || '0';
        const conj = conjugateExpr(upper);
        state.symbolicMatrix[r][c] = conj;
        const inp = document.getElementById(`sym-${r}-${c}`);
        if (inp) {
          inp.value = conj;
          const kEl = inp.nextElementSibling;
          if (kEl) renderKatex(kEl, exprToLatex(conj));
        }
      }
    }
  }
  // Update status badge
  const statusEl = document.getElementById('herm-status');
  if (statusEl) {
    statusEl.textContent = 'Hermitian';
    statusEl.className = 'herm-status ok';
  }
  updateHamiltonianMatrixPreview();
}

// ─── Math Expression Utilities ───────────────────────────────────────────────

function evalMathExpr(expr) {
  if (expr === null || expr === undefined) return NaN;
  const s = String(expr).trim();
  if (s === '') return NaN;
  const n = Number(s);
  if (!isNaN(n)) return n;
  const safe = s
    .replace(/\bpi\b/gi, String(Math.PI))
    .replace(/\bsqrt\s*\(/gi, 'Math.sqrt(')
    .replace(/\babs\s*\(/gi, 'Math.abs(')
    .replace(/\bsin\s*\(/gi, 'Math.sin(')
    .replace(/\bcos\s*\(/gi, 'Math.cos(')
    .replace(/\btan\s*\(/gi, 'Math.tan(')
    .replace(/\bexp\s*\(/gi, 'Math.exp(')
    .replace(/\bln\s*\(/gi, 'Math.log(')
    .replace(/\^/g, '**');
  try {
    // eslint-disable-next-line no-new-func
    const val = Function('"use strict"; return (' + safe + ')')();
    if (typeof val !== 'number' || !isFinite(val)) throw new Error('non-finite');
    return val;
  } catch (_) { throw new Error('parse error'); }
}

// ─── Auto-sync & Formula Simplification helpers ───
// A hopping amplitude may be a string expression, a plain number, or a complex
// object {re, im} (e.g. from the Flat Band designer / IFT). Render it as a
// sympy-parseable scalar expression -- NOT String(obj) which yields the
// "[object Object]" that breaks both the H(k) preview and the band solver.
// Round to `sig` significant figures, returning a SHORT decimal string (no
// trailing 14-digit float noise like 0.0023164299306). Used for display.
function roundSig(x, sig = 5) {
  if (!isFinite(x) || x === 0) return 0;
  const d = Math.ceil(Math.log10(Math.abs(x)));
  const mag = Math.pow(10, sig - d);
  return Math.round(x * mag) / mag;
}

function hoppingAmplitudeToExpr(t, sig = 5) {
  if (t === null || t === undefined) return '0';
  if (typeof t === 'string') return t.trim();
  if (typeof t === 'number') return String(roundSig(t, sig));
  if (typeof t === 'object' && ('re' in t || 'im' in t)) {
    const re = roundSig(Number(t.re) || 0, sig);
    const im = roundSig(Number(t.im) || 0, sig);
    const EPS = 1e-12;
    if (Math.abs(im) < EPS) return String(re);
    if (Math.abs(re) < EPS) return `${im}*I`;
    return `${re}${im < 0 ? '-' : '+'}${Math.abs(im)}*I`;
  }
  return String(t);
}

function formatHoppingTerm(hop) {
  const R = hop.R;
  const t = hoppingAmplitudeToExpr(hop.t).trim();
  if (t === '0' || t === '') return '';

  const kNames = ['kx', 'ky', 'kz'];
  const nonZero = [];
  for (let d = 0; d < state.dimension; d++) {
    const r = R[d] || 0;
    if (r !== 0) {
      nonZero.push({ r, k: kNames[d] });
    }
  }

  let expStr = '';
  if (nonZero.length > 0) {
    const phaseTerms = nonZero.map(({ r, k }) => {
      if (r === 1) return k;
      if (r === -1) return `-${k}`;
      return `${r}*${k}`;
    });
    let phase = phaseTerms.join('+').replace(/\+-/g, '-');
    expStr = `exp(I*(${phase}))`;
  }

  if (!expStr) return t;

  const isOne = t === '1' || t === '1.0' || t === '1.';
  const isNegOne = t === '-1' || t === '-1.0' || t === '-1.';

  if (isOne) return expStr;
  if (isNegOne) return `-${expStr}`;

  const needsParens = t.includes('+') || t.includes('-');
  const tPart = needsParens ? `(${t})` : t;
  return `${tPart}*${expStr}`;
}

function syncHoppingsToSymbolicMatrix() {
  if (state.hamiltonianMode !== 'hopping') return;

  const Q = state.orbitals.length;
  const newMatrix = Array.from({ length: Q }, () => Array(Q).fill('0'));

  const grouped = Array.from({ length: Q }, () => Array.from({ length: Q }, () => []));
  for (const hop of state.hoppings) {
    const term = formatHoppingTerm(hop);
    if (term) {
      grouped[hop.i][hop.j].push(term);
    }
  }

  for (let r = 0; r < Q; r++) {
    for (let c = 0; c < Q; c++) {
      if (grouped[r][c].length > 0) {
        newMatrix[r][c] = grouped[r][c].join(' + ').replace(/\+\s*-/g, '- ');
      }
    }
  }

  state.symbolicMatrix = newMatrix;
}

function buildMatrixLatex(matrixData) {
  const Q = matrixData.length;
  let latex = 'H(k) = \\begin{pmatrix}\n';
  for (let r = 0; r < Q; r++) {
    const rowLatex = [];
    for (let c = 0; c < Q; c++) {
      const expr = matrixData[r][c] || '0';
      rowLatex.push(exprToLatex(expr));
    }
    latex += '  ' + rowLatex.join(' & ') + (r === Q - 1 ? '\n' : ' \\\\\n');
  }
  latex += '\\end{pmatrix}';
  return latex;
}

// Try to render LaTeX (display mode) with REAL error detection
// (throwOnError:true), returning whether it succeeded. throwOnError:false
// would silently paint a red error in-place and report success, which is
// exactly the "수식이 빨간 코드로 보임" symptom we want to avoid.
function tryKatexDisplay(el, latex) {
  try {
    katex.render(latex, el, { throwOnError: true, displayMode: true });
    el.classList.remove('render-err');
    return true;
  } catch (_) {
    return false;
  }
}

function renderKatexDisplay(el, latex) {
  if (!tryKatexDisplay(el, latex)) {
    el.textContent = latex;
    el.classList.add('render-err');
  }
}

// Render one matrix entry "label = expr" with graceful degradation:
//   1) multi-line wrapped (aligned)  ->  2) single line  ->  3) plain text.
// So a term KaTeX can't parse never blanks the entry or shows raw LaTeX code.
function renderEntrySafe(el, label, expr) {
  if (tryKatexDisplay(el, buildEntryAlignedLatex(label, expr))) return;
  let oneLine = null;
  try { oneLine = `${label} = ${exprToLatex(expr)}`; } catch (_) {}
  if (oneLine && tryKatexDisplay(el, oneLine)) return;
  el.classList.add('render-err');
  el.textContent = `${label.replace(/[{}\\,]/g, '')} = ${expr}`;
}

// Split an expression string into its TOP-LEVEL additive terms, keeping each
// term's leading sign. Depth-aware over () and {} so phases like
// exp(I*(kx+ky)) are not split, and scientific notation (1e-3) is preserved.
function splitTopLevelTerms(expr) {
  const s = String(expr == null ? '' : expr).trim();
  if (!s) return [];
  const terms = [];
  let depth = 0, brace = 0, start = 0;
  for (let i = 0; i < s.length; i++) {
    const ch = s[i];
    if (ch === '(') depth++;
    else if (ch === ')') depth--;
    else if (ch === '{') brace++;
    else if (ch === '}') brace--;
    else if ((ch === '+' || ch === '-') && depth === 0 && brace === 0 && i > start) {
      const prev = s[i - 1];
      // scientific notation: 1e-3 / 1E+5
      if ((prev === 'e' || prev === 'E') && /[0-9.]/.test(s[i + 1] || '')) continue;
      // sign glued to a preceding operator/open-paren => unary, not a split point
      if (prev === '*' || prev === '/' || prev === '+' || prev === '-' ||
          prev === '(' || prev === '^' || prev === ',') continue;
      terms.push(s.slice(start, i).trim());
      start = i; // keep the sign with the next term
    }
  }
  terms.push(s.slice(start).trim());
  return terms.filter(t => t.length > 0);
}

// Build a multi-line \begin{aligned} body for one matrix entry, wrapping long
// sums so they stay readable instead of overflowing horizontally.
function buildEntryAlignedLatex(label, expr) {
  const terms = splitTopLevelTerms(expr);
  if (terms.length === 0) return `${label} &= 0`;
  const PER_LINE = 3;
  let out = `${label} &= `;
  for (let i = 0; i < terms.length; i++) {
    let t = terms[i];
    let sign = '+';
    if (t[0] === '+') { t = t.slice(1).trim(); }
    else if (t[0] === '-') { sign = '-'; t = t.slice(1).trim(); }
    const lx = exprToLatex(t);
    if (i === 0) {
      out += (sign === '-' ? '-' : '') + lx;
    } else if (i % PER_LINE === 0) {
      out += ` \\\\\n  &\\quad ${sign}\\, ${lx}`;
    } else {
      out += ` ${sign}\\, ${lx}`;
    }
  }
  return `\\begin{aligned}\n  ${out}\n\\end{aligned}`;
}

// Big "H(k) view": a compact H_{ij} label matrix, followed by each nonzero
// entry written out separately with long expressions auto-wrapped. Rendering
// each entry on its own isolates KaTeX failures (one bad term no longer blanks
// the whole matrix).
function renderHMatrixBig(container, matrixData) {
  const Q = matrixData.length;
  container.innerHTML = '';
  container.style.display = 'block';
  container.style.width = '100%';
  container.style.margin = '0';

  const matWrap = document.createElement('div');
  matWrap.style.cssText = 'display:flex; justify-content:center; margin-bottom:1rem; overflow-x:auto;';
  let matLatex = 'H(k) = \\begin{pmatrix}\n';
  for (let r = 0; r < Q; r++) {
    const cells = [];
    for (let c = 0; c < Q; c++) {
      const expr = (matrixData[r] && matrixData[r][c] != null ? matrixData[r][c] : '0').toString().trim();
      cells.push((expr === '0' || expr === '') ? '0' : `H_{${r + 1},${c + 1}}`);
    }
    matLatex += '  ' + cells.join(' & ') + (r === Q - 1 ? '\n' : ' \\\\\n');
  }
  matLatex += '\\end{pmatrix}';
  renderKatexDisplay(matWrap, matLatex);
  container.appendChild(matWrap);

  const head = document.createElement('div');
  head.style.cssText = 'font-size:0.78rem; font-weight:700; color:#475569; margin:0.3rem 0 0.5rem; border-top:1px dashed #e2e8f0; padding-top:0.6rem;';
  head.textContent = '항별 전개 (각 성분 H_{ij}(k))';
  container.appendChild(head);

  const list = document.createElement('div');
  list.style.cssText = 'display:flex; flex-direction:column; gap:0.5rem; width:100%; max-width:840px; margin:0 auto;';
  let count = 0;
  for (let r = 0; r < Q; r++) {
    for (let c = 0; c < Q; c++) {
      const expr = (matrixData[r] && matrixData[r][c] != null ? matrixData[r][c] : '0').toString().trim();
      if (expr === '0' || expr === '') continue;
      count++;
      const item = document.createElement('div');
      item.style.cssText = 'padding:0.45rem 0.7rem; background:#fff; border:1px solid #e2e8f0; border-radius:6px; overflow-x:auto;';
      renderEntrySafe(item, `H_{${r + 1},${c + 1}}(k)`, expr);
      list.appendChild(item);
    }
  }
  if (count === 0) {
    const empty = document.createElement('div');
    empty.style.cssText = 'text-align:center; color:#94a3b8; font-size:0.85rem; padding:1rem;';
    empty.textContent = '0이 아닌 항이 없습니다.';
    list.appendChild(empty);
  }
  container.appendChild(list);
}

function renderMatrixPreview(matrixData) {
  const container = document.getElementById('h-matrix-preview-katex');
  if (!container) return;

  renderHMatrixBig(container, matrixData);

  const Q = state.orbitals.length;
  for (let r = 0; r < Q; r++) {
    for (let c = 0; c < Q; c++) {
      const cellKatex = document.querySelector(`#sym-${r}-${c}`)?.nextElementSibling;
      if (cellKatex) {
        const expr = (matrixData && matrixData[r] && matrixData[r][c] !== undefined)
                     ? matrixData[r][c]
                     : (state.symbolicMatrix[r] ? state.symbolicMatrix[r][c] : '0');
        renderKatex(cellKatex, exprToLatex(expr));
      }
    }
  }
}

function updateHamiltonianMatrixPreview() {
  const Q = state.orbitals.length;
  if (Q === 0) return;

  if (state.hamiltonianMode === 'hopping') {
    syncHoppingsToSymbolicMatrix();
  }

  const simplifyActive = document.getElementById('h-matrix-simplify-chk')?.checked && pyReady;
  
  if (simplifyActive) {
    const precisionVal = parseInt(document.getElementById('simplify-precision')?.value) || 5;
    const thresholdVal = parseFloat(document.getElementById('simplify-threshold')?.value) ?? 1e-4;

    clearTimeout(simplifyMatrixTimer);
    simplifyMatrixTimer = setTimeout(async () => {
      try {
        const matrixToSend = [];
        for (let r = 0; r < Q; r++) {
          const row = [];
          for (let c = 0; c < Q; c++) {
            row.push(state.symbolicMatrix[r][c] || '0');
          }
          matrixToSend.push(row);
        }
        
        const res = await apiFetch('simplify_matrix', { 
          matrix: matrixToSend,
          precision: precisionVal,
          threshold: thresholdVal
        });
        if (res && res.matrix) {
          simplifiedMatrix = res.matrix;
          renderMatrixPreview(res.matrix);
        }
      } catch (err) {
        console.error('Matrix simplification failed:', err);
        renderMatrixPreview(state.symbolicMatrix);
      }
    }, 250);
  } else {
    simplifiedMatrix = null;
    renderMatrixPreview(state.symbolicMatrix);
  }
}

function hopTermToLatex(R, tVal) {
  let tStr = '';
  if (typeof tVal === 'string') {
    tStr = exprToLatex(tVal);
  } else if (typeof tVal === 'object' && tVal !== null && 're' in tVal) {
    const re = tVal.re || 0;
    const im = tVal.im || 0;
    if (Math.abs(im) < 1e-9) {
      tStr = String(re);
    } else if (Math.abs(re) < 1e-9) {
      tStr = `${im}i`;
    } else {
      tStr = `(${re} + ${im}i)`;
    }
  } else if (typeof tVal === 'number') {
    tStr = String(tVal);
  } else {
    tStr = String(tVal ?? '0');
  }

  // Simplify simple numbers
  const isOne = tStr === '1' || tStr === '1.0' || tStr === '1.';
  const isNegOne = tStr === '-1' || tStr === '-1.0' || tStr === '-1.';
  const isZero = tStr === '0' || tStr === '0.0' || tStr === '0.';

  if (isZero) return '';

  // Phase factor e^{i k.R}
  const kNames = ['k_x', 'k_y', 'k_z'];
  const terms = [];
  for (let di = 0; di < R.length; di++) {
    const rVal = R[di];
    if (rVal !== 0) {
      const kName = kNames[di] || `k_{${di+1}}`;
      if (rVal === 1) {
        terms.push(kName);
      } else if (rVal === -1) {
        terms.push(`-${kName}`);
      } else {
        terms.push(`${rVal}${kName}`);
      }
    }
  }

  let expStr = '';
  if (terms.length > 0) {
    let phaseExpr = terms.join('+').replace(/\+-/g, '-');
    expStr = `e^{i(${phaseExpr})}`;
  }

  if (expStr === '') {
    return tStr;
  } else {
    if (isOne) return expStr;
    if (isNegOne) return `-${expStr}`;
    // If tStr has plus/minus, wrap in parentheses
    if ((tStr.includes('+') || tStr.includes('-')) && !tStr.startsWith('(')) {
      return `(${tStr})\\,${expStr}`;
    }
    return `${tStr}\\,${expStr}`;
  }
}

function getHamiltonianLatex() {
  // Sync state first to ensure we get current inputs
  try {
    syncUIToState();
  } catch (_) {}

  const Q = state.orbitals.length;

  const simplifyActive = document.getElementById('h-matrix-simplify-chk')?.checked && pyReady;
  if (simplifyActive && simplifiedMatrix) {
    return buildMatrixLatex(simplifiedMatrix);
  }

  let matrix = Array.from({length: Q}, () => Array.from({length: Q}, () => '0'));

  if (state.hamiltonianMode === 'hopping') {
    let tempTerms = Array.from({length: Q}, () => Array.from({length: Q}, () => []));
    for (const hop of state.hoppings) {
      const term = hopTermToLatex(hop.R, hop.t);
      if (term) {
        tempTerms[hop.i][hop.j].push(term);
      }
    }
    for (let r = 0; r < Q; r++) {
      for (let c = 0; c < Q; c++) {
        if (tempTerms[r][c].length > 0) {
          matrix[r][c] = tempTerms[r][c].join(' + ').replace(/\+\s*-/g, '- ');
        }
      }
    }
  } else {
    // Mode B: Symbolic Matrix
    for (let r = 0; r < Q; r++) {
      for (let c = 0; c < Q; c++) {
        const expr = state.symbolicMatrix[r][c] || '0';
        matrix[r][c] = exprToLatex(expr);
      }
    }
  }

  // Construct LaTeX output
  let latex = 'H(k) = \\begin{pmatrix}\n';
  for (let r = 0; r < Q; r++) {
    latex += '  ' + matrix[r].join(' & ') + (r === Q - 1 ? '\n' : ' \\\\\n');
  }
  latex += '\\end{pmatrix}';
  return latex;
}

function innerKx(s) {
  return s
    .replace(/\bkx\b/g, 'k_x').replace(/\bky\b/g, 'k_y').replace(/\bkz\b/g, 'k_z')
    .replace(/\bI\b/g, 'i').replace(/\s*\*\s*/g, '\\,').trim();
}

// ─── Helper: find matching closing paren ─────────────────────────────────────
function findMatchingParen(s, startIdx) {
  let depth = 0;
  for (let i = startIdx; i < s.length; i++) {
    if (s[i] === '(') depth++;
    else if (s[i] === ')') { depth--; if (depth === 0) return i; }
  }
  return -1; // no match found
}

// ─── Greek letter LaTeX map ──────────────────────────────────────────────────
const GREEK_LATEX = {
  alpha: '\\alpha', beta: '\\beta', gamma: '\\gamma', delta: '\\delta',
  epsilon: '\\varepsilon', mu: '\\mu', sigma: '\\sigma', lambda: '\\lambda',
  omega: '\\omega', theta: '\\theta', phi: '\\phi', psi: '\\psi',
  nu: '\\nu', tau: '\\tau', eta: '\\eta', zeta: '\\zeta',
  xi: '\\xi', rho: '\\rho', kappa: '\\kappa', pi: '\\pi'
};

function exprToLatex(expr) {
  if (!expr || expr === '0') return '0';
  let s = String(expr).trim();

  // Step 1: Handle function calls with bracket-aware inner parsing
  // Process exp(...) with depth-aware matching
  let result = '';
  let i = 0;
  while (i < s.length) {
    // Match exp(
    if (s.substring(i, i + 4) === 'exp(' && (i === 0 || !s[i-1]?.match(/[a-zA-Z_]/))) {
      const closeIdx = findMatchingParen(s, i + 3);
      if (closeIdx > 0) {
        const inner = s.substring(i + 4, closeIdx);
        const latexInner = exprToLatex(inner);
        result += `e^{${latexInner}}`;
        i = closeIdx + 1;
        continue;
      }
    }
    // Match cos(
    if (s.substring(i, i + 4) === 'cos(' && (i === 0 || !s[i-1]?.match(/[a-zA-Z_]/))) {
      const closeIdx = findMatchingParen(s, i + 3);
      if (closeIdx > 0) {
        const inner = s.substring(i + 4, closeIdx);
        const latexInner = exprToLatex(inner);
        // Simple single-var: \cos k_x   Complex: \cos(...)
        if (/^k_[xyz]$/.test(latexInner)) {
          result += `\\cos ${latexInner}`;
        } else {
          result += `\\cos\\left(${latexInner}\\right)`;
        }
        i = closeIdx + 1;
        continue;
      }
    }
    // Match sin(
    if (s.substring(i, i + 4) === 'sin(' && (i === 0 || !s[i-1]?.match(/[a-zA-Z_]/))) {
      const closeIdx = findMatchingParen(s, i + 3);
      if (closeIdx > 0) {
        const inner = s.substring(i + 4, closeIdx);
        const latexInner = exprToLatex(inner);
        if (/^k_[xyz]$/.test(latexInner)) {
          result += `\\sin ${latexInner}`;
        } else {
          result += `\\sin\\left(${latexInner}\\right)`;
        }
        i = closeIdx + 1;
        continue;
      }
    }
    // Match tan(
    if (s.substring(i, i + 4) === 'tan(' && (i === 0 || !s[i-1]?.match(/[a-zA-Z_]/))) {
      const closeIdx = findMatchingParen(s, i + 3);
      if (closeIdx > 0) {
        const inner = s.substring(i + 4, closeIdx);
        result += `\\tan\\left(${exprToLatex(inner)}\\right)`;
        i = closeIdx + 1;
        continue;
      }
    }
    // Match sqrt(
    if (s.substring(i, i + 5) === 'sqrt(' && (i === 0 || !s[i-1]?.match(/[a-zA-Z_]/))) {
      const closeIdx = findMatchingParen(s, i + 4);
      if (closeIdx > 0) {
        const inner = s.substring(i + 5, closeIdx);
        result += `\\sqrt{${exprToLatex(inner)}}`;
        i = closeIdx + 1;
        continue;
      }
    }
    // Match abs(
    if (s.substring(i, i + 4) === 'abs(' && (i === 0 || !s[i-1]?.match(/[a-zA-Z_]/))) {
      const closeIdx = findMatchingParen(s, i + 3);
      if (closeIdx > 0) {
        const inner = s.substring(i + 4, closeIdx);
        result += `\\left|${exprToLatex(inner)}\\right|`;
        i = closeIdx + 1;
        continue;
      }
    }
    // Match conj(
    if (s.substring(i, i + 5) === 'conj(' && (i === 0 || !s[i-1]?.match(/[a-zA-Z_]/))) {
      const closeIdx = findMatchingParen(s, i + 4);
      if (closeIdx > 0) {
        const inner = s.substring(i + 5, closeIdx);
        result += `\\overline{${exprToLatex(inner)}}`;
        i = closeIdx + 1;
        continue;
      }
    }
    result += s[i];
    i++;
  }
  s = result;

  // Step 2: Power notation  ** → ^
  s = s.replace(/\*\*/g, '^');
  // x^n → x^{n} for multi-digit or negative exponents
  s = s.replace(/\^(-?\d+(?:\.\d+)?)/g, (_, exp) => `^{${exp}}`);

  // Step 3: k-variables
  s = s.replace(/\bkx\b/g, 'k_x').replace(/\bky\b/g, 'k_y').replace(/\bkz\b/g, 'k_z');

  // Step 4: Imaginary unit
  s = s.replace(/\bI\b/g, 'i');

  // Step 5: Greek letters → LaTeX
  for (const [name, latex] of Object.entries(GREEK_LATEX)) {
    if (name === 'pi') continue; // pi handled separately
    const re = new RegExp('\\b' + name + '\\b', 'g');
    s = s.replace(re, latex + ' ');
  }
  s = s.replace(/\bpi\b/g, '\\pi');

  // Step 6: Parameter subscripts (t1→t_{1}, a12→a_{12})
  s = s.replace(/\b([a-zA-Z])(\d+)\b/g, (match, letter, digits) => {
    // Don't subscript things already LaTeX-processed or k_x etc
    if (match === 'k_x' || match === 'k_y' || match === 'k_z') return match;
    return `${letter}_{${digits}}`;
  });

  // Step 7: Multiplication
  s = s.replace(/\s*\*\s*/g, '\\,');

  // Step 8: Clean up spacing
  s = s.replace(/  +/g, ' ').trim();

  return s;
}

function conjugateExpr(expr) {
  if (!expr || expr === '0') return expr;
  let s = String(expr);

  // Step 1: Flip sign of I inside exp()
  // exp(I*...) → exp(-I*...)  and  exp(-I*...) → exp(I*...)
  s = s.replace(/exp\((-?)I\b/g, (_, sign) => {
    return sign === '-' ? 'exp(I' : 'exp(-I';
  });

  // Step 2: Replace standalone I with (-I), avoiding exp contexts (already handled)
  // Only replace I that is NOT immediately after 'exp(' or '-'
  s = s.replace(/(?<!exp\()(?<!exp\(-)(?<!\w)\bI\b/g, '(-I)');

  return s;
}

function renderKatex(el, latex) {
  try {
    katex.render(latex, el, { throwOnError: false, displayMode: false });
    el.classList.remove('render-err');
  } catch (_) {
    el.textContent = latex;
    el.classList.add('render-err');
  }
}

// Build LaTeX for a hopping term (supports expression strings)
function hopToLatex(iLbl, jLbl, R, tExpr) {
  let tStr = '';

  if (typeof tExpr === 'string') {
    // String expression for t
    tStr = exprToLatex(tExpr) + '\\,';
  } else if (typeof tExpr === 'object' && tExpr !== null && 're' in tExpr) {
    const tRe = tExpr.re || 0;
    const tIm = tExpr.im || 0;
    if (Math.abs(tIm) < 1e-9) {
      tStr = (tRe === 1) ? '' : (tRe === -1) ? '-' : String(tRe) + ' \\cdot ';
    } else if (Math.abs(tRe) < 1e-9) {
      tStr = `${tIm}i \\cdot `;
    } else {
      tStr = `(${tRe}+${tIm}i) \\cdot `;
    }
  } else {
    // Numeric
    const v = Number(tExpr) || 0;
    tStr = (v === 1) ? '' : (v === -1) ? '-' : String(v) + ' \\cdot ';
  }

  const kNames = ['k_x', 'k_y', 'k_z'];
  const nonZero = R.map((r, i) => ({ r, k: kNames[i] || `k_{${i+1}}` })).filter(x => x.r !== 0);
  let expStr = '';
  if (nonZero.length > 0) {
    const terms = nonZero.map(({ r, k }) => r === 1 ? k : r === -1 ? `-${k}` : `${r}${k}`)
      .join('+').replace(/\+-/g, '-');
    expStr = `e^{i(${terms})}`;
  }
  return `H_{${iLbl}\\leftarrow ${jLbl}} \\mathrel{+}= ${tStr}${expStr || '1'}`;
}

// ─── Utility: Laurent polynomial → readable HTML ─────────────────────────────
function laurentToHtml(repr) {
  if (!repr || repr === '0') return '0';
  let s = repr;
  // Match variables with exponents (integer or fractional)
  s = s.replace(/X(\d+)\^(-?\d+(?:\.\d+)?)/g, (_, n, p) => {
    const kName = ['k<sub>x</sub>', 'k<sub>y</sub>', 'k<sub>z</sub>'][parseInt(n) - 1] || `k<sub>${n}</sub>`;
    const pf = parseFloat(p);
    const pInt = Math.round(pf);
    let pStr;
    if (Math.abs(pf - pInt) < 0.0001) {
      pStr = pInt.toString();
    } else {
      // Try symbolic recognition for the absolute value, then restore sign
      const sym = toNiceNum(Math.abs(pf));
      pStr = sym !== null ? (pf < 0 ? `-${sym}` : sym) : pf.toPrecision(6).replace(/\.?0+$/, '');
    }
    if (pStr === '1') return `e<sup>i${kName}</sup>`;
    if (pStr === '-1') return `e<sup>-i${kName}</sup>`;
    return `e<sup>${pStr}·i${kName}</sup>`;
  });
  // Match variables without exponent (implicit ^1)
  s = s.replace(/X(\d+)/g, (_, n) => {
    const kName = ['k<sub>x</sub>', 'k<sub>y</sub>', 'k<sub>z</sub>'][parseInt(n) - 1] || `k<sub>${n}</sub>`;
    return `e<sup>i${kName}</sup>`;
  });
  // Replace decimal coefficients with symbolic equivalents (e.g. 0.866... → √3/2)
  s = s.replace(/(-?\d+(?:\.\d+)?(?:e[+-]?\d+)?)/g, (_, numStr) => {
    const v = parseFloat(numStr);
    if (!isFinite(v) || Number.isInteger(v)) return numStr;
    const sym = toNiceNum(Math.abs(v));
    if (sym === null) return numStr;
    return (v < 0 ? '-' : '') + sym;
  });
  // Clean up coefficient formatting
  s = s.replace(/\b1\.0+\b/g, '1');
  s = s.replace(/\b-1\.0+\b/g, '-1');
  s = s.replace(/(\.\d*[1-9])0+\b/g, '$1'); // trim trailing zeros
  // Clean up signs
  s = s.replace(/\+\s*-/g, '− ');
  s = s.replace(/\s+/g, ' ');
  return s;
}


// ─── Utility: HTML escape ──────────────────────────────────────────────────────
function escHtml(s) {
  if (typeof s !== 'string') s = String(s ?? '');
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ─── Input Synchronization and Auto-Save ───────────────────────────────────────
function syncUIToState() {
  // 1. Primitive vectors
  const pvInps = document.querySelectorAll('#prim-vecs-ui .math-expr-input');
  let idx = 0;
  const d = state.dimension;
  for (let li = 0; li < d; li++) {
    for (let ci = 0; ci < d; ci++) {
      const inp = pvInps[idx++];
      if (inp) {
        try {
          state.primitiveVectors[li][ci] = evalMathExpr(inp.value);
        } catch (_) {}
      }
    }
  }

  // 2. Orbitals
  const orbRows = document.querySelectorAll('#orbitals-ui .orbital-row-math');
  orbRows.forEach((row, qi) => {
    const labelInp = row.querySelector('.orbital-label-input');
    if (labelInp) state.orbitals[qi].label = labelInp.value;
    const posInps = row.querySelectorAll('.orb-pos-wrap .math-expr-input');
    posInps.forEach((inp, di) => {
      try {
        state.orbitals[qi].position[di] = evalMathExpr(inp.value);
      } catch (_) {}
    });
  });

  // 3. Hoppings or Matrix
  if (state.hamiltonianMode === 'hopping') {
    const cards = document.querySelectorAll('#hop-cards-ui .hop-card');
    const hops = [];
    cards.forEach(card => {
      const selI   = card.querySelector('.hop-sel-i');
      const selJ   = card.querySelector('.hop-sel-j');
      const rInps  = card.querySelectorAll('.hop-r-inp');
      const inpExpr = card.querySelector('.hop-expr-input');
      if (!selI || !selJ) return;
      const i   = parseInt(selI.value);
      const j   = parseInt(selJ.value);
      const R   = Array.from(rInps, inp => parseInt(inp.value) || 0);

      const exprVal = inpExpr ? inpExpr.value.trim() : '0';
      let t;
      if (exprVal === '' || exprVal === '0') {
        t = 0;
      } else if (hasSymbolicParams(exprVal)) {
        t = exprVal;
      } else {
        try {
          t = evalMathExpr(exprVal);
        } catch (_) {
          t = exprVal;
        }
      }
      hops.push({ i, j, R, t });
    });
    state.hoppings = hops;
  } else {
    const Q = state.orbitals.length;
    const mat = [];
    for (let r = 0; r < Q; r++) {
      const row = [];
      for (let c = 0; c < Q; c++) {
        const el  = document.getElementById(`sym-${r}-${c}`);
        row.push(el ? (el.value || '0') : '0');
      }
      mat.push(row);
    }
    state.symbolicMatrix = mat;
  }

  // 4. Custom k-path
  const kpathInput = document.getElementById('param-kpath');
  if (kpathInput) {
    state.kPathStr = kpathInput.value.trim();
  }

  const plotnInput = document.getElementById('param-plotn');
  if (plotnInput) {
    state.plotN = parseInt(plotnInput.value) || 60;
  }
  
  updateHamiltonianMatrixPreview();
}

function triggerAutoSave() {
  try {
    syncUIToState();
    const spec = buildSpec();
    const activeState = {
      spec: spec,
      dimension: state.dimension,
      kGrid: state.kGrid,
      flatTol: state.flatTol,
      plotN: state.plotN,
      hamiltonianMode: state.hamiltonianMode,
      parameters: { ...state.parameters },
      parameterRanges: { ...state.parameterRanges }
    };
    localStorage.setItem('cls_active_state', JSON.stringify(activeState));
  } catch (_) {
    // Ignore input errors while typing
  }
}

function loadActiveState() {
  try {
    const raw = localStorage.getItem('cls_active_state');
    if (!raw) return false;
    const activeState = JSON.parse(raw);
    if (!activeState || !activeState.spec) return false;

    console.log('[CLS] Restoring auto-saved active state');
    state.dimension = activeState.dimension || activeState.spec.lattice.dimension || 2;
    state.kGrid = activeState.kGrid || 30;
    state.flatTol = activeState.flatTol || 1e-5;
    state.plotN = activeState.plotN || 60;
    state.hamiltonianMode = activeState.hamiltonianMode || 'hopping';
    
    // Restore parameters
    if (activeState.parameters) state.parameters = { ...activeState.parameters };
    if (activeState.parameterRanges) {
      state.parameterRanges = {};
      for (const [k, v] of Object.entries(activeState.parameterRanges)) {
        state.parameterRanges[k] = { ...v };
      }
    }
    
    applySpecToState(activeState.spec);
    
    // Set UI values
    document.getElementById('param-kgrid').value = state.kGrid;
    document.getElementById('param-tol').value   = state.flatTol;
    document.getElementById('param-plotn').value = state.plotN || 60;
    
    const mode = state.hamiltonianMode;
    document.querySelectorAll('.tab-btn').forEach(b =>
      b.classList.toggle('active', b.dataset.tab === mode));
    document.getElementById('tab-hopping').classList.toggle('hidden', mode !== 'hopping');
    document.getElementById('tab-matrix').classList.toggle('hidden', mode !== 'matrix');
    
    rebuildLatticeUI();
    return true;
  } catch (e) {
    console.error('Failed to load auto-saved active state', e);
    return false;
  }
}

// ─── Export/Import Handlers ────────────────────────────────────────────────────
function exportModelToJson() {
  try {
    syncUIToState();
    const spec = buildSpec();
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(spec, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    const dateStr = new Date().toISOString().slice(0,10);
    downloadAnchor.setAttribute("download", `cls_model_${dateStr}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  } catch (err) {
    alert('모델을 내보내는 중 오류가 발생했습니다: ' + err.message);
  }
}

function importModelFromJson(e) {
  const file = e.target.files[0];
  if (!file) return;

  const reader = new FileReader();
  reader.onload = function(evt) {
    try {
      const spec = JSON.parse(evt.target.result);
      if (!spec || (!spec.lattice && !spec.primitive_vectors)) {
        throw new Error('유효한 CLS Finder 모델 규격이 아닙니다.');
      }
      
      applySpecToState(spec);
      rebuildLatticeUI();
      rebuildHamiltonianEditor();
      triggerAutoSave();
      
      // Reset preset dropdown selection
      document.getElementById('preset-select').value = "";
      document.getElementById('preset-desc').textContent = "가져온 사용자 모델";
      document.getElementById('delete-preset-btn').classList.add('hidden');
      
      alert('모델을 파일에서 정상적으로 가져왔습니다!');
    } catch (err) {
      alert('모델 가져오기 실패: ' + err.message);
    }
  };
  reader.readAsText(file);
}

// ─── Lattice Drawing ──────────────────────────────────────────────────────────

const LAT_PALETTE = [
  '#2563eb','#dc2626','#16a34a','#d97706','#7c3aed',
  '#0891b2','#db2777','#65a30d','#b45309','#4f46e5'
];

let latSt = {
  repX: 2, repY: 2,
  showCell: true, showVectors: true, showHops: true,
  showLabels: true, showGrid: false,
  zoom: 1.0, panX: 0, panY: 0,
  orbColors: [], orbShapes: [],
  annotations: [], nextAnnotId: 0,
  annotMode: false,
  isDragging: false,
  dragX0: 0, dragY0: 0, panX0: 0, panY0: 0,
};

function latEnsureOrbStyles() {
  const n = state.orbitals.length;
  
  // Detect sublattice groups for color assignment
  const sublattices = [];
  state.orbitals.forEach(orb => {
    const pos = orb.position;
    const match = sublattices.find(s =>
      s.pos.length === pos.length && s.pos.every((v, i) => Math.abs(v - pos[i]) < 1e-6)
    );
    if (match) {
      match.orbitals.push(orb);
    } else {
      sublattices.push({ pos: [...pos], orbitals: [orb] });
    }
  });

  // Assign colors: use sublattice-based colors from SUBLATTICE_COLORS
  while (latSt.orbColors.length < n) {
    const qi = latSt.orbColors.length;
    const orb = state.orbitals[qi];
    // Find which sublattice this orbital belongs to
    let subIdx = sublattices.findIndex(s =>
      s.pos.length === orb.position.length &&
      s.pos.every((v, i) => Math.abs(v - orb.position[i]) < 1e-6)
    );
    if (subIdx < 0) subIdx = qi;
    const subColor = getSublatticeColor(subIdx);
    latSt.orbColors.push(subColor.border);
  }

  // Assign shapes: use orbital_in_sublattice for differentiation
  while (latSt.orbShapes.length < n) {
    const qi = latSt.orbShapes.length;
    const orb = state.orbitals[qi];
    const samePosBefore = state.orbitals.slice(0, qi).filter(o =>
      o.position.length === orb.position.length &&
      o.position.every((v, i) => Math.abs(v - orb.position[i]) < 1e-6)
    ).length;
    const shapes = ['circle', 'square', 'diamond', 'triangle', 'triangle-down', 'pentagon', 'hexagon', 'star', 'cross', 'x', 'hourglass', 'bowtie'];
    latSt.orbShapes.push(shapes[samePosBefore % shapes.length]);
  }

  latSt.orbColors.length = n;
  latSt.orbShapes.length = n;
}

function latGetTransformGeneric(cfg, svgEl) {
  const dim = cfg.dimension, vecs = cfg.primitiveVectors, orbs = cfg.orbitals;
  const repX = cfg.repX, repY = dim >= 2 ? cfg.repY : 0;
  const rect = svgEl.getBoundingClientRect();
  const svgW = rect.width || 600, svgH = rect.height || 490;
  const allPts = [];
  for (let n = -repX; n <= repX; n++) {
    for (let m = -repY; m <= repY; m++) {
      orbs.forEach(orb => {
        const pos = orb.position || [];
        const f0 = (pos[0]||0) + n;
        const f1 = (dim>=2 ? (pos[1]||0) : 0) + m;
        let wx = 0, wy = 0;
        if (vecs[0]) {
          wx += f0 * vecs[0][0];
          wy += f0 * (vecs[0][1]||0);
        }
        if (dim >= 2 && vecs[1]) {
          wx += f1 * vecs[1][0];
          wy += f1 * (vecs[1][1]||0);
        }
        allPts.push([wx, wy]);
      });
    }
  }
  if (vecs[0]) allPts.push([vecs[0][0], vecs[0][1]||0]);
  if (dim >= 2 && vecs[1]) allPts.push([vecs[1][0], vecs[1][1]||0]);
  if (!allPts.length) allPts.push([0,0],[1,1]);
  const xvals = allPts.map(p=>p[0]), yvals = allPts.map(p=>p[1]);
  const xmin=Math.min(...xvals), xmax=Math.max(...xvals);
  const ymin=Math.min(...yvals), ymax=Math.max(...yvals);
  const dxw=Math.max(xmax-xmin,0.5), dyw=Math.max(ymax-ymin,0.5);
  const scale = Math.min((svgW*0.64)/dxw, (svgH*0.64)/dyw) * cfg.zoom;
  const wCx=(xmin+xmax)/2, wCy=(ymin+ymax)/2;
  const cx=svgW/2+cfg.panX, cy=svgH/2+cfg.panY;
  const w2s = (wx,wy) => [cx+(wx-wCx)*scale, cy-(wy-wCy)*scale];
  const s2w = (sx,sy) => [(sx-cx)/scale+wCx, (cy-sy)/scale+wCy];
  return { scale, w2s, s2w };
}

function latGetTransform(svgEl) {
  return latGetTransformGeneric({
    dimension: state.dimension,
    primitiveVectors: state.primitiveVectors,
    orbitals: state.orbitals,
    repX: latSt.repX,
    repY: latSt.repY,
    zoom: latSt.zoom,
    panX: latSt.panX,
    panY: latSt.panY,
  }, svgEl);
}

function svgE(tag, attrs={}) {
  const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [k,v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}

function latDarken(hex) {
  if (!hex || hex[0]!=='#' || hex.length<7) return '#000';
  return `rgb(${Math.floor(parseInt(hex.slice(1,3),16)*0.55)},${Math.floor(parseInt(hex.slice(3,5),16)*0.55)},${Math.floor(parseInt(hex.slice(5,7),16)*0.55)})`;
}

function latDrawSiteShape(layer, sx, sy, r, color, shape, opacityVal, isHollow = false) {
  const g = svgE('g', {opacity: opacityVal});
  const fillVal = isHollow ? '#ffffff' : color;
  const base = {fill:fillVal, stroke:latDarken(color), 'stroke-width':isHollow ? '2.0' : '1.8'};
  let el;
  if (shape==='square') {
    el = svgE('rect',{...base,x:sx-r,y:sy-r,width:2*r,height:2*r,rx:'2'});
  } else if (shape==='diamond') {
    el = svgE('polygon',{...base,points:`${sx},${sy-r} ${sx+r},${sy} ${sx},${sy+r} ${sx-r},${sy}`});
  } else if (shape==='triangle' || shape==='triangle-up') {
    const h=r*1.18;
    el = svgE('polygon',{...base,points:`${sx},${sy-h} ${sx+h*0.866},${sy+h*0.5} ${sx-h*0.866},${sy+h*0.5}`});
  } else if (shape==='triangle-down') {
    const h = r * 1.18;
    el = svgE('polygon',{...base,points:`${sx},${sy+h} ${sx+h*0.866},${sy-h*0.5} ${sx-h*0.866},${sy-h*0.5}`});
  } else if (shape==='pentagon') {
    const pts = [];
    for (let i = 0; i < 5; i++) {
      const angle = -Math.PI / 2 + i * (2 * Math.PI / 5);
      const px = sx + r * Math.cos(angle);
      const py = sy + r * Math.sin(angle);
      pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
    }
    el = svgE('polygon', {...base, points: pts.join(' ')});
  } else if (shape==='hexagon') {
    const pts = [];
    for (let i = 0; i < 6; i++) {
      const angle = i * (Math.PI / 3);
      const px = sx + r * Math.cos(angle);
      const py = sy + r * Math.sin(angle);
      pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
    }
    el = svgE('polygon', {...base, points: pts.join(' ')});
  } else if (shape==='star') {
    const pts = [];
    for (let i = 0; i < 10; i++) {
      const angle = -Math.PI / 2 + i * (Math.PI / 5);
      const R = (i % 2 === 0) ? r : r * 0.45;
      const px = sx + R * Math.cos(angle);
      const py = sy + R * Math.sin(angle);
      pts.push(`${px.toFixed(1)},${py.toFixed(1)}`);
    }
    el = svgE('polygon', {...base, points: pts.join(' ')});
  } else if (shape==='cross') {
    const w = r / 3.0;
    el = svgE('polygon', {
      ...base,
      points: `${sx-r},${sy-w} ${sx-w},${sy-w} ${sx-w},${sy-r} ${sx+w},${sy-r} ${sx+w},${sy-w} ${sx+r},${sy-w} ${sx+r},${sy+w} ${sx+w},${sy+w} ${sx+w},${sy+r} ${sx-w},${sy+r} ${sx-w},${sy+w} ${sx-r},${sy+w}`
    });
  } else if (shape==='x') {
    const w = r / 3.0;
    el = svgE('polygon', {
      ...base,
      points: `${sx-r},${sy-w} ${sx-w},${sy-w} ${sx-w},${sy-r} ${sx+w},${sy-r} ${sx+w},${sy-w} ${sx+r},${sy-w} ${sx+r},${sy+w} ${sx+w},${sy+w} ${sx+w},${sy+r} ${sx-w},${sy+r} ${sx-w},${sy+w} ${sx-r},${sy+w}`,
      transform: `rotate(45, ${sx}, ${sy})`
    });
  } else if (shape==='hourglass') {
    el = svgE('polygon', {
      ...base,
      points: `${sx-r},${sy-r} ${sx+r},${sy-r} ${sx+r*0.25},${sy} ${sx+r},${sy+r} ${sx-r},${sy+r} ${sx-r*0.25},${sy}`
    });
  } else if (shape==='bowtie') {
    el = svgE('polygon', {
      ...base,
      points: `${sx-r},${sy-r} ${sx-r},${sy+r} ${sx},${sy+r*0.25} ${sx+r},${sy+r} ${sx+r},${sy-r} ${sx},${sy-r*0.25}`
    });
  } else {
    el = svgE('circle',{...base,cx:sx,cy:sy,r});
  }
  g.appendChild(el); layer.appendChild(g);
}

function latBuildSVG() {
  const svg = document.getElementById('lat-svg');
  if (!svg) return;
  latEnsureOrbStyles();
  const dim=state.dimension, vecs=state.primitiveVectors, orbs=state.orbitals, hops=state.hoppings;
  const hint = document.getElementById('lat-empty-hint');
  if (hint) hint.style.display='none';
  const {w2s} = latGetTransform(svg);
  const repX=latSt.repX, repY=dim>=2?latSt.repY:0;
  svg.innerHTML = '';

  // Arrow markers
  const defs=svgE('defs');
  [['a1','#2563eb'],['a2','#dc2626'],['hop','#6b7280']].forEach(([id,col])=>{
    const m=svgE('marker',{id:`lat-arr-${id}`,markerWidth:'10',markerHeight:'7',refX:'8',refY:'3.5',orient:'auto'});
    m.appendChild(svgE('polygon',{points:'0 0, 10 3.5, 0 7',fill:col}));
    defs.appendChild(m);
  });
  svg.appendChild(defs);

  // Layers
  const L={};
  ['grid','cell','hops','sites','vectors','labels','annots'].forEach(n=>{L[n]=svgE('g');svg.appendChild(L[n]);});

  const sitePos=(q,n,m)=>{
    if(q>=orbs.length)return[0,0];
    const pos = orbs[q].position || [];
    const f0 = (pos[0]||0) + n;
    const f1 = (dim>=2 ? (pos[1]||0) : 0) + m;
    let wx = 0, wy = 0;
    if(vecs[0]){
      wx += f0 * vecs[0][0];
      wy += f0 * (vecs[0][1]||0);
    }
    if(dim>=2&&vecs[1]){
      wx += f1 * vecs[1][0];
      wy += f1 * (vecs[1][1]||0);
    }
    return [wx, wy];
  };

  // Background grid
  if(latSt.showGrid&&dim>=2&&vecs[0]&&vecs[1]){
    for(let n=-repX-1;n<=repX;n++)for(let m=-repY-1;m<=repY;m++){
      const ox=n*vecs[0][0]+m*vecs[1][0], oy=n*(vecs[0][1]||0)+m*(vecs[1][1]||0);
      [[ox,oy,ox+vecs[0][0],oy+(vecs[0][1]||0)],[ox,oy,ox+vecs[1][0],oy+(vecs[1][1]||0)]].forEach(([x1w,y1w,x2w,y2w])=>{
        const[x1,y1]=w2s(x1w,y1w),[x2,y2]=w2s(x2w,y2w);
        L.grid.appendChild(svgE('line',{x1,y1,x2,y2,stroke:'#e2e8f0','stroke-width':'0.8'}));
      });
    }
  }

  // Unit cell outlines
  if(latSt.showCell){
    if(dim>=2&&vecs[0]&&vecs[1]){
      for(let n=-repX;n<=repX;n++)for(let m=-repY;m<=repY;m++){
        const ox=n*vecs[0][0]+m*vecs[1][0], oy=n*(vecs[0][1]||0)+m*(vecs[1][1]||0);
        const pts=[[ox,oy],[ox+vecs[0][0],oy+(vecs[0][1]||0)],
          [ox+vecs[0][0]+vecs[1][0],oy+(vecs[0][1]||0)+(vecs[1][1]||0)],
          [ox+vecs[1][0],oy+(vecs[1][1]||0)]]
          .map(([wx,wy])=>w2s(wx,wy)).map(([x,y])=>`${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
        const isO=n===0&&m===0;
        L.cell.appendChild(svgE('polygon',{points:pts,
          fill:isO?'rgba(76,122,255,0.07)':'none',
          stroke:isO?'#4c7aff':'#c7d3ee',
          'stroke-width':isO?'1.5':'0.7',
          'stroke-dasharray':isO?'5,3':'3,2'}));
      }
    } else if(dim===1&&vecs[0]){
      for(let n=-repX;n<=repX;n++){
        const[sx0,sy0]=w2s(n*vecs[0][0],0),[sx1,sy1]=w2s((n+1)*vecs[0][0],0);
        const isO=n===0, bY=Math.max(sy0,sy1)+22;
        L.cell.appendChild(svgE('line',{x1:sx0,y1:bY,x2:sx1,y2:bY,
          stroke:isO?'#4c7aff':'#c7d3ee','stroke-width':isO?'2':'0.8','stroke-dasharray':isO?'':'3,2'}));
        if(isO)[sx0,sx1].forEach(bx=>L.cell.appendChild(svgE('line',{x1:bx,y1:bY-5,x2:bx,y2:bY+5,stroke:'#4c7aff','stroke-width':'1.5'})));
      }
    }
  }

  // Hopping bonds
  if(latSt.showHops&&hops.length>0){
    for(let n=-repX;n<=repX;n++)for(let m=-repY;m<=repY;m++){
      hops.forEach(hop=>{
        const{i,j,R}=hop;
        if(i>=orbs.length||j>=orbs.length)return;
        const rn=n+(R[0]||0), rm=m+(dim>=2?(R[1]||0):0);
        if(Math.abs(rn)>repX+0.5||(dim>=2&&Math.abs(rm)>repY+0.5))return;
        const[wx1,wy1]=sitePos(i,n,m),[wx2,wy2]=sitePos(j,rn,rm);
        const[sx1,sy1]=w2s(wx1,wy1),[sx2,sy2]=w2s(wx2,wy2);
        L.hops.appendChild(svgE('line',{x1:sx1,y1:sy1,x2:sx2,y2:sy2,stroke:'#94a3b8','stroke-width':'1.6','stroke-linecap':'round'}));
      });
    }
  }

  // Sites
  const siteR=12;
  const posCounts = {};
  const orbIndexInPos = [];
  orbs.forEach((orb, qi) => {
    const posKey = `${(orb.position[0]||0).toFixed(4)}_${(orb.position[1]||0).toFixed(4)}`;
    if (posCounts[posKey] === undefined) {
      posCounts[posKey] = 0;
    }
    orbIndexInPos[qi] = posCounts[posKey];
    posCounts[posKey]++;
  });

  for(let n=-repX;n<=repX;n++)for(let m=-repY;m<=repY;m++){
    const isOrigin=n===0&&m===0, opacity=isOrigin?1:0.35;
    orbs.forEach((orb,qi)=>{
      const[wx,wy]=sitePos(qi,n,m),[sx,sy]=w2s(wx,wy);
      const color=latSt.orbColors[qi]||LAT_PALETTE[qi%LAT_PALETTE.length];
      const r = Math.max(siteR - 3.2 * orbIndexInPos[qi], 4.5);
      const isHollow = orbIndexInPos[qi] % 2 === 1;
      latDrawSiteShape(L.sites,sx,sy,r,color,latSt.orbShapes[qi]||'circle',opacity,isHollow);
      if(latSt.showLabels&&isOrigin){
        const fo=svgE('foreignObject',{x:sx+r+2,y:sy-12,width:'80',height:'26'});
        const div=document.createElementNS('http://www.w3.org/1999/xhtml','div');
        div.className='lat-label lat-site-label';
        katex.render(/^[A-Za-z0-9]$/.test(orb.label)?orb.label:`\\text{${orb.label}}`,div,{throwOnError:false});
        fo.appendChild(div); L.labels.appendChild(fo);
      }
    });
  }

  // Origin dot
  const[ox0,oy0]=w2s(0,0);
  L.sites.appendChild(svgE('circle',{cx:ox0,cy:oy0,r:'2.5',fill:'#334155',opacity:'0.4'}));

  // Lattice vectors
  if(latSt.showVectors){
    const vDefs=[];
    if(vecs[0])vDefs.push({v:vecs[0],label:'\\mathbf{a}_1',col:'#2563eb',id:'a1'});
    if(dim>=2&&vecs[1])vDefs.push({v:vecs[1],label:'\\mathbf{a}_2',col:'#dc2626',id:'a2'});
    vDefs.forEach(({v,label,col,id})=>{
      const[x0,y0]=w2s(0,0),[x1,y1]=w2s(v[0],v[1]||0);
      const dx=x1-x0,dy=y1-y0,len=Math.sqrt(dx*dx+dy*dy);
      if(len<4)return;
      const sf=(len-11)/len;
      L.vectors.appendChild(svgE('line',{x1:x0,y1:y0,x2:(x0+dx*sf).toFixed(1),y2:(y0+dy*sf).toFixed(1),
        stroke:col,'stroke-width':'2.5','stroke-linecap':'round','marker-end':`url(#lat-arr-${id})`}));
      const mx=(x0+x1)/2,my=(y0+y1)/2,nx=-dy/len,ny=dx/len;
      const side=(ny<0||(Math.abs(ny)<0.1&&nx>0))?-1:1;
      const fo=svgE('foreignObject',{x:(mx+side*nx*20-15).toFixed(1),y:(my+side*ny*20-12).toFixed(1),width:'70',height:'26'});
      const div=document.createElementNS('http://www.w3.org/1999/xhtml','div');
      div.className='lat-label lat-vec-label'; div.style.color=col;
      katex.render(label,div,{throwOnError:false});
      fo.appendChild(div); L.vectors.appendChild(fo);
    });
  }

  // Annotations
  latSt.annotations.forEach(a=>{
    const[sx,sy]=w2s(a.wx,a.wy);
    const fo=svgE('foreignObject',{x:(sx-35).toFixed(1),y:(sy-14).toFixed(1),width:'120',height:'32'});
    const div=document.createElementNS('http://www.w3.org/1999/xhtml','div');
    div.className='lat-label lat-annot-label'; div.title='Click to remove';
    try{katex.render(a.latex,div,{throwOnError:false});}catch(_){div.textContent=a.latex;}
    div.onclick=()=>{latSt.annotations=latSt.annotations.filter(x=>x.id!==a.id);latBuildSVG();};
    fo.appendChild(div); L.annots.appendChild(fo);
  });
}

function buildSiteStylePanel() {
  const panel=document.getElementById('lat-site-styles');
  if(!panel)return;
  panel.innerHTML='';
  latEnsureOrbStyles();
  state.orbitals.forEach((orb,qi)=>{
    const div=document.createElement('div'); div.className='lat-orb-style';
    const lbl=document.createElement('span'); lbl.className='lat-orb-name'; lbl.textContent=orb.label;
    const colorInp=document.createElement('input');
    colorInp.type='color'; colorInp.className='lat-color-btn';
    colorInp.value=latSt.orbColors[qi]||LAT_PALETTE[qi%LAT_PALETTE.length];
    colorInp.title='색상';
    colorInp.addEventListener('input',e=>{latSt.orbColors[qi]=e.target.value;latBuildSVG();});
    const shapeSel=document.createElement('select'); shapeSel.className='lat-shape-sel';
    [
      ['circle','● 원 (Circle)'],
      ['square','■ 사각 (Square)'],
      ['diamond','◆ 마름모 (Diamond)'],
      ['triangle','▲ 삼각 (Triangle Up)'],
      ['triangle-down','▼ 역삼각 (Triangle Down)'],
      ['pentagon','⬠ 오각 (Pentagon)'],
      ['hexagon','⬡ 육각 (Hexagon)'],
      ['star','★ 별 (Star)'],
      ['cross','✚ 십자 (Cross)'],
      ['x','✖ X자 (X-Cross)'],
      ['hourglass','⧓ 모래시계 (Hourglass)'],
      ['bowtie','⋈ 나비넥타이 (Bowtie)']
    ].forEach(([v,t])=>{
      const opt=document.createElement('option'); opt.value=v; opt.textContent=t;
      if(v===(latSt.orbShapes[qi]||'circle'))opt.selected=true;
      shapeSel.appendChild(opt);
    });
    shapeSel.addEventListener('change',e=>{latSt.orbShapes[qi]=e.target.value;latBuildSVG();});
    div.append(lbl,colorInp,shapeSel); panel.appendChild(div);
  });
}

function latExportSVG() {
  const svg=document.getElementById('lat-svg'); if(!svg)return;
  const clone=svg.cloneNode(true);
  clone.setAttribute('xmlns','http://www.w3.org/2000/svg');
  clone.setAttribute('xmlns:xhtml','http://www.w3.org/1999/xhtml');
  const bg=document.createElementNS('http://www.w3.org/2000/svg','rect');
  bg.setAttribute('width','100%'); bg.setAttribute('height','100%'); bg.setAttribute('fill','white');
  clone.insertBefore(bg,clone.firstChild);
  const str='<?xml version="1.0" encoding="UTF-8"?>\n'+new XMLSerializer().serializeToString(clone);
  const url=URL.createObjectURL(new Blob([str],{type:'image/svg+xml'}));
  const a=document.createElement('a'); a.href=url; a.download='lattice.svg'; a.click();
  URL.revokeObjectURL(url);
}

function latExportPNG() {
  const svg=document.getElementById('lat-svg'); if(!svg)return;
  const{width:w,height:h}=svg.getBoundingClientRect();
  const sc=2, canvas=document.createElement('canvas');
  canvas.width=w*sc; canvas.height=h*sc;
  const ctx=canvas.getContext('2d');
  ctx.fillStyle='#ffffff'; ctx.fillRect(0,0,canvas.width,canvas.height);
  const clone=svg.cloneNode(true);
  clone.setAttribute('xmlns','http://www.w3.org/2000/svg');
  clone.querySelectorAll('foreignObject').forEach(fo=>{
    const div=fo.querySelector('div');
    const te=document.createElementNS('http://www.w3.org/2000/svg','text');
    te.setAttribute('x',parseFloat(fo.getAttribute('x')||0)+4);
    te.setAttribute('y',parseFloat(fo.getAttribute('y')||0)+14);
    te.setAttribute('font-size','12');
    te.setAttribute('font-family','STIX Two Math,Times New Roman,serif');
    te.setAttribute('fill','#1a1a3e');
    te.textContent=div?div.textContent:'';
    fo.replaceWith(te);
  });
  const url=URL.createObjectURL(new Blob([new XMLSerializer().serializeToString(clone)],{type:'image/svg+xml;charset=utf-8'}));
  const img=new Image();
  img.onload=()=>{
    ctx.scale(sc,sc); ctx.drawImage(img,0,0,w,h); URL.revokeObjectURL(url);
    const a=document.createElement('a'); a.download='lattice.png'; a.href=canvas.toDataURL('image/png'); a.click();
  };
  img.src=url;
}

function latSVGClick(e) {
  if(!latSt.annotMode)return;
  const latex=(document.getElementById('lat-annot-input')?.value||'').trim();
  if(!latex){alert('주석할 LaTeX 수식을 먼저 입력하세요.');return;}
  const svg=document.getElementById('lat-svg'); if(!svg)return;
  const rect=svg.getBoundingClientRect();
  const{s2w}=latGetTransform(svg);
  const[wx,wy]=s2w(e.clientX-rect.left,e.clientY-rect.top);
  latSt.annotations.push({wx,wy,latex,id:latSt.nextAnnotId++});
  latBuildSVG();
}

function initLatticeDrawing() {
  const wrap=document.getElementById('lat-canvas-wrap');
  const svg=document.getElementById('lat-svg');
  if(!wrap||!svg)return;

  document.getElementById('lat-update-btn')?.addEventListener('click',()=>{
    latSt.zoom=1.0; latSt.panX=0; latSt.panY=0;
    latEnsureOrbStyles(); buildSiteStylePanel(); latBuildSVG();
  });
  document.getElementById('lat-rep-x')?.addEventListener('change',e=>{latSt.repX=parseInt(e.target.value);latBuildSVG();});
  document.getElementById('lat-rep-y')?.addEventListener('change',e=>{latSt.repY=parseInt(e.target.value);latBuildSVG();});
  [['lat-show-cell','showCell'],['lat-show-vectors','showVectors'],
   ['lat-show-hops','showHops'],['lat-show-labels','showLabels'],['lat-show-grid','showGrid']
  ].forEach(([id,prop])=>document.getElementById(id)?.addEventListener('change',e=>{latSt[prop]=e.target.checked;latBuildSVG();}));

  document.getElementById('lat-export-svg')?.addEventListener('click',latExportSVG);
  document.getElementById('lat-export-png')?.addEventListener('click',latExportPNG);

  wrap.addEventListener('mousedown',e=>{
    if(latSt.annotMode)return;
    latSt.isDragging=true; latSt.dragX0=e.clientX; latSt.dragY0=e.clientY;
    latSt.panX0=latSt.panX; latSt.panY0=latSt.panY; e.preventDefault();
  });
  wrap.addEventListener('mousemove',e=>{
    if(!latSt.isDragging)return;
    latSt.panX=latSt.panX0+(e.clientX-latSt.dragX0); latSt.panY=latSt.panY0+(e.clientY-latSt.dragY0);
    latBuildSVG();
  });
  ['mouseup','mouseleave'].forEach(ev=>wrap.addEventListener(ev,()=>{latSt.isDragging=false;}));
  wrap.addEventListener('wheel',e=>{
    e.preventDefault();
    latSt.zoom=Math.max(0.12,Math.min(15,latSt.zoom*(e.deltaY>0?0.9:1.1)));
    latBuildSVG();
  },{passive:false});

  svg.addEventListener('click',latSVGClick);

  const annotBtn=document.getElementById('lat-annot-place-btn');
  annotBtn?.addEventListener('click',()=>{
    latSt.annotMode=!latSt.annotMode;
    wrap.classList.toggle('annot-mode',latSt.annotMode);
    if(annotBtn){annotBtn.textContent=latSt.annotMode?'✓ 배치 중...':'📍 클릭 배치'; annotBtn.style.borderStyle=latSt.annotMode?'solid':'';}
  });
  document.getElementById('lat-clear-annot')?.addEventListener('click',()=>{latSt.annotations=[];latBuildSVG();});
}

// ─── Event Listeners ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  // Try to load auto-saved state, otherwise load initial default UI
  if (!loadActiveState()) {
    state.kPointsOverride = getDefaultKPointsMap(state.dimension, state.primitiveVectors);
    rebuildLatticeUI();
    rebuildKPointsOverrideUI();
  }

  // Dimension buttons
  document.querySelectorAll('.dim-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      setDimension(parseInt(btn.dataset.d));
      triggerAutoSave();
    });
  });

  // Add orbital
  document.getElementById('add-orbital-btn').addEventListener('click', () => {
    const d   = state.dimension;
    const next = String.fromCharCode(65 + state.orbitals.length);
    state.orbitals.push({ label: next, position: Array(d).fill(0.0) });
    rebuildOrbitalsUI();
    rebuildHamiltonianEditor();
    triggerAutoSave();
  });

  // Add hopping row
  document.getElementById('add-hop-btn').addEventListener('click', () => {
    addDefaultHop();
    triggerAutoSave();
  });

  // Hamiltonian mode tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      const mode = btn.dataset.tab;
      state.hamiltonianMode = mode;
      document.querySelectorAll('.tab-btn').forEach(b =>
        b.classList.toggle('active', b.dataset.tab === mode));
      document.getElementById('tab-hopping').classList.toggle('hidden', mode !== 'hopping');
      document.getElementById('tab-matrix').classList.toggle('hidden', mode !== 'matrix');
      if (mode === 'matrix') rebuildSymMatrix();
      triggerAutoSave();
    });
  });

  // Symbol insert buttons
  document.querySelectorAll('.sym-insert').forEach(btn => {
    btn.addEventListener('click', () => {
      if (focusedSymCell) {
        const start = focusedSymCell.selectionStart;
        const end   = focusedSymCell.selectionEnd;
        const val   = focusedSymCell.value;
        focusedSymCell.value = val.slice(0, start) + btn.dataset.ins + val.slice(end);
        focusedSymCell.focus();
        const pos = start + btn.dataset.ins.length;
        focusedSymCell.setSelectionRange(pos, pos);
        // Trigger input event to update state + KaTeX preview
        focusedSymCell.dispatchEvent(new Event('input'));
      }
    });
  });

  // Hermitian auto-fill toggle
  document.getElementById('herm-auto')?.addEventListener('change', () => {
    rebuildSymMatrix();
    triggerAutoSave();
  });

  // Results tabs
  document.querySelectorAll('.rtab').forEach(btn => {
    btn.addEventListener('click', () => showPanel(btn.dataset.panel));
  });

  // Preset select description
  document.getElementById('preset-select').addEventListener('change', e => {
    const opt = e.target.selectedOptions[0];
    document.getElementById('preset-desc').textContent = opt?.dataset.desc || '';
    
    const delBtn = document.getElementById('delete-preset-btn');
    if (opt && opt.dataset.isUser === 'true') {
      delBtn.classList.remove('hidden');
    } else {
      delBtn.classList.add('hidden');
    }
  });

  // Load preset
  document.getElementById('load-preset-btn').addEventListener('click', () => {
    const id = document.getElementById('preset-select').value;
    if (id) {
      applyPreset(id);
      triggerAutoSave();
    }
  });

  // Delete user preset
  document.getElementById('delete-preset-btn')?.addEventListener('click', () => {
    const sel = document.getElementById('preset-select');
    const opt = sel.selectedOptions[0];
    if (!opt || opt.dataset.isUser !== 'true') return;
    
    if (confirm(`저장된 모델 "${opt.text}"을(를) 삭제하시겠습니까?`)) {
      const userModels = getUserModels().filter(m => m.id !== opt.value);
      saveUserModels(userModels);
      loadPresetList();
      document.getElementById('delete-preset-btn').classList.add('hidden');
      document.getElementById('preset-desc').textContent = '';
      triggerAutoSave();
    }
  });

  // Save current UI state as a preset
  document.getElementById('save-preset-btn')?.addEventListener('click', () => {
    const name = prompt('저장할 모델의 이름을 입력하세요:');
    if (!name || name.trim() === '') return;
    
    let spec;
    try {
      spec = buildSpec();
    } catch (err) {
      alert('설정 오류가 있어 저장할 수 없습니다: ' + err.message);
      return;
    }
    
    const id = 'user_model_' + Date.now();
    const Q = state.orbitals.length;
    const dim = state.dimension;
    const desc = `${dim}D 사용자 격자, Q=${Q}`;
    
    const newModel = {
      id,
      name: name.trim(),
      dim,
      Q,
      desc,
      spec
    };
    
    const userModels = getUserModels();
    userModels.push(newModel);
    saveUserModels(userModels);
    
    loadPresetList();
    
    // Select the newly saved model
    const sel = document.getElementById('preset-select');
    sel.value = id;
    sel.dispatchEvent(new Event('change'));
    
    alert(`모델 "${name}"이(가) 정상적으로 저장되었습니다!`);
    triggerAutoSave();
  });

  // Export JSON preset
  document.getElementById('export-preset-btn')?.addEventListener('click', exportModelToJson);

  // Import JSON preset
  const fileInp = document.getElementById('import-file-input');
  document.getElementById('import-preset-btn')?.addEventListener('click', () => {
    fileInp.click();
  });
  fileInp?.addEventListener('change', importModelFromJson);

  // Copy Hamiltonian LaTeX
  document.getElementById('copy-h-latex-btn')?.addEventListener('click', () => {
    try {
      const latex = getHamiltonianLatex();
      navigator.clipboard.writeText(latex).then(() => {
        const btn = document.getElementById('copy-h-latex-btn');
        const origText = btn.textContent;
        btn.textContent = '✅ 복사 완료!';
        btn.style.background = '#dcfce7';
        btn.style.color = '#166534';
        btn.style.borderColor = '#bbf7d0';
        setTimeout(() => {
          btn.textContent = origText;
          btn.style.background = '';
          btn.style.color = '';
          btn.style.borderColor = '';
        }, 1500);
      }).catch(err => {
        alert('복사 중 오류가 발생했습니다: ' + err.message);
      });
    } catch (err) {
      alert('LaTeX 생성 오류: ' + err.message);
    }
  });

  // Save Hamiltonian LaTeX
  document.getElementById('save-h-latex-btn')?.addEventListener('click', () => {
    try {
      const latex = getHamiltonianLatex();
      const blob = new Blob([latex], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", url);
      const dateStr = new Date().toISOString().slice(0,10);
      downloadAnchor.setAttribute("download", `hamiltonian_${dateStr}.tex`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert('LaTeX 저장 중 오류가 발생했습니다: ' + err.message);
    }
  });

  // Parameters
  document.getElementById('param-kgrid').addEventListener('change', e => {
    state.kGrid = parseInt(e.target.value) || 30;
    triggerAutoSave();
    debouncedBandPreview();
  });
  document.getElementById('param-tol').addEventListener('change', e => {
    state.flatTol = parseFloat(e.target.value) || 1e-5;
    triggerAutoSave();
  });
  document.getElementById('param-plotn').addEventListener('change', e => {
    state.plotN = parseInt(e.target.value) || 60;
    triggerAutoSave();
  });
  document.getElementById('param-kpath')?.addEventListener('change', e => {
    state.kPathStr = e.target.value.trim();
    triggerAutoSave();
    debouncedBandPreview();
  });
  document.getElementById('add-kpoint-override-btn')?.addEventListener('click', () => {
    const dim = state.dimension;
    let labelIndex = 0;
    let label = '';
    do {
      label = String.fromCharCode(65 + labelIndex);
      labelIndex++;
    } while (state.kPointsOverride[label]);
    
    state.kPointsOverride[label] = Array(dim).fill(0.0);
    rebuildKPointsOverrideUI();
    triggerAutoSave();
    debouncedBandPreview();
  });

  // Band preview button
  document.getElementById('band-preview-btn').addEventListener('click', () => {
    runBandPreview();
  });

  // Run button (CLS full analysis)
  document.getElementById('run-btn').addEventListener('click', () => {
    runAnalysis();
    triggerAutoSave();
  });

  document.querySelector('.input-panel')?.addEventListener('input', triggerAutoSave);
  document.querySelector('.input-panel')?.addEventListener('change', triggerAutoSave);
  
  // Hamiltonian matrix modal open/close
  document.getElementById('open-h-matrix-modal-btn')?.addEventListener('click', openHamiltonianMatrixModal);
  document.getElementById('h-matrix-modal-close-btn')?.addEventListener('click', closeHamiltonianMatrixModal);
  document.getElementById('h-matrix-modal-ok-btn')?.addEventListener('click', closeHamiltonianMatrixModal);
  document.getElementById('h-matrix-modal')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) {
      closeHamiltonianMatrixModal();
    }
  });

  // Hopping Structure Viewer
  document.getElementById('open-hopping-viewer-btn')?.addEventListener('click', openHoppingViewer);
  document.getElementById('hopping-viewer-close-btn')?.addEventListener('click', closeHoppingViewer);
  document.getElementById('hopping-viewer-ok-btn')?.addEventListener('click', closeHoppingViewer);
  document.getElementById('hopping-viewer-modal')?.addEventListener('click', (e) => {
    if (e.target === e.currentTarget) closeHoppingViewer();
  });
  ['hopview-rep', 'hopview-hc', 'hopview-onsite'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', () => renderHoppingViewer());
    document.getElementById(id)?.addEventListener('change', () => renderHoppingViewer());
  });

  // Modal LaTeX copy/save
  document.getElementById('modal-copy-h-latex-btn')?.addEventListener('click', () => {
    try {
      const latex = getHamiltonianLatex();
      navigator.clipboard.writeText(latex).then(() => {
        const btn = document.getElementById('modal-copy-h-latex-btn');
        const origText = btn.innerHTML;
        btn.innerHTML = '✅ 복사 완료!';
        btn.style.background = '#dcfce7';
        btn.style.color = '#166534';
        btn.style.borderColor = '#bbf7d0';
        setTimeout(() => {
          btn.innerHTML = origText;
          btn.style.background = '';
          btn.style.color = '';
          btn.style.borderColor = '';
        }, 1500);
      }).catch(err => {
        alert('복사 중 오류가 발생했습니다: ' + err.message);
      });
    } catch (err) {
      alert('LaTeX 생성 오류: ' + err.message);
    }
  });

  document.getElementById('modal-save-h-latex-btn')?.addEventListener('click', () => {
    try {
      const latex = getHamiltonianLatex();
      const blob = new Blob([latex], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", url);
      const dateStr = new Date().toISOString().slice(0,10);
      downloadAnchor.setAttribute("download", `hamiltonian_${dateStr}.tex`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert('LaTeX 저장 중 오류가 발생했습니다: ' + err.message);
    }
  });

  // Simplify checkbox listener
  document.getElementById('h-matrix-simplify-chk')?.addEventListener('change', () => {
    updateHamiltonianMatrixPreview();
  });
  document.getElementById('simplify-precision')?.addEventListener('change', () => {
    updateHamiltonianMatrixPreview();
  });
  document.getElementById('simplify-threshold')?.addEventListener('change', () => {
    updateHamiltonianMatrixPreview();
  });

  // Modal close buttons
  document.getElementById('modal-close-btn')?.addEventListener('click', closeExpressionModal);
  document.getElementById('modal-cancel-btn')?.addEventListener('click', closeExpressionModal);
  
  // Modal apply button
  document.getElementById('modal-apply-btn')?.addEventListener('click', applyExpressionModal);
  
  // Modal textarea live preview
  document.getElementById('modal-expr-input')?.addEventListener('input', updateModalPreview);
  
  // Helper insert buttons
  document.querySelectorAll('#expression-modal .helper-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      insertSymbolInModal(btn.dataset.insert);
    });
  });
  
  // Keyboard listeners for modal
  window.addEventListener('keydown', (e) => {
    const modal = document.getElementById('expression-modal');
    if (modal && !modal.classList.contains('hidden')) {
      if (e.key === 'Escape') {
        closeExpressionModal();
      } else if (e.key === 'Enter' && e.ctrlKey) {
        applyExpressionModal();
      }
    }
    const hModal = document.getElementById('h-matrix-modal');
    if (hModal && !hModal.classList.contains('hidden')) {
      if (e.key === 'Escape') {
        closeHamiltonianMatrixModal();
      }
    }
  });

  // Init native Python API server
  initApiClient();

  // Init lattice drawing panel
  initLatticeDrawing();

  // Nanoribbon Analysis run button listener
  document.getElementById('ribbon-run-btn')?.addEventListener('click', runNanoribbonAnalysis);
  document.getElementById('rbm-run-btn')?.addEventListener('click', runRBMAnalysis);
  document.getElementById('topo-run-btn')?.addEventListener('click', runGlobalTopologyAnalysis);
  document.getElementById('topo-fukane-btn')?.addEventListener('click', () => runFuKaneAnalysis());

  // Nanoribbon Periodic direction change listener
  document.getElementById('ribbon-pdir')?.addEventListener('change', e => {
    const pdir = e.target.value;
    const label = document.getElementById('ribbon-krange-label');
    if (label) {
      label.textContent = `k_${pdir} 범위`;
    }
    const fixedLabel = document.getElementById('ribbon-kfixed-label');
    if (fixedLabel) {
      fixedLabel.textContent = `고정된 k_${pdir === 'y' ? 'x' : 'y'}`;
    }
  });

  // Load saved ribbons from localStorage
  loadSavedRibbons();

  // Nanoribbon Save result listener
  document.getElementById('ribbon-save-btn')?.addEventListener('click', () => {
    if (!ribbonLastData) {
      alert('저장할 나노리본 계산 결과가 없습니다. 먼저 계산을 실행하세요.');
      return;
    }
    
    let name = document.getElementById('ribbon-save-name').value.trim();
    if (!name) {
      name = prompt('저장할 계산의 이름을 입력하세요:', '나노리본 계산');
      if (!name) return;
    }
    
    const spec = buildSpec();
    const Nx = parseInt(document.getElementById('ribbon-nx').value) || 40;
    const Nk = parseInt(document.getElementById('ribbon-nk').value) || 200;
    const periodicDir = document.getElementById('ribbon-pdir').value || 'y';
    const threshold = parseFloat(document.getElementById('ribbon-threshold').value) || 0.5;
    const kFixed = document.getElementById('ribbon-kfixed').value;
    
    const item = {
      id: 'ribbon_' + Date.now(),
      name: name,
      spec: spec,
      Nx: Nx,
      Nk: Nk,
      periodicDir: periodicDir,
      threshold: threshold,
      kFixed: kFixed,
      data: ribbonLastData
    };
    
    savedRibbons.push(item);
    saveSavedRibbons();
    
    // Select the newly saved run in dropdown
    const sel = document.getElementById('ribbon-saved-list');
    if (sel) {
      sel.value = savedRibbons.length - 1;
    }
    
    alert(`"${name}" 결과가 저장되었습니다.`);
  });

  // Nanoribbon Load result listener
  document.getElementById('ribbon-load-btn')?.addEventListener('click', () => {
    const selIdx = document.getElementById('ribbon-saved-list').value;
    if (selIdx === '') {
      alert('불러올 계산을 선택하세요.');
      return;
    }
    
    const item = savedRibbons[parseInt(selIdx)];
    if (!item) return;
    
    // Restore inputs in UI so the user knows what they are looking at
    document.getElementById('ribbon-nx').value = item.Nx;
    document.getElementById('ribbon-nk').value = item.Nk;
    document.getElementById('ribbon-pdir').value = item.periodicDir;
    document.getElementById('ribbon-threshold').value = item.threshold;
    if (item.kFixed !== undefined) {
      document.getElementById('ribbon-kfixed').value = item.kFixed;
    }
    
    // Fire periodicDir change event to update labels
    document.getElementById('ribbon-pdir').dispatchEvent(new Event('change'));
    
    // Render dispersion using the saved data, threshold, and periodicDir
    ribbonLastData = item.data;
    
    // We pass a unique ID to renderRibbonDispersion
    const dataToRender = { ...item.data, id: item.id };
    
    renderRibbonDispersion(dataToRender, item.threshold, item.periodicDir);
    
    // Save the specific spec and Nx for wavefunction queries on the plot container
    const container = document.getElementById('ribbon-dispersion-plot');
    if (container) {
      container.ribbonSpec = item.spec;
      container.ribbonNx = item.Nx;
    }
  });

  // Nanoribbon Delete result listener
  document.getElementById('ribbon-delete-btn')?.addEventListener('click', () => {
    const selIdx = document.getElementById('ribbon-saved-list').value;
    if (selIdx === '') {
      alert('삭제할 계산을 선택하세요.');
      return;
    }
    
    const idx = parseInt(selIdx);
    const item = savedRibbons[idx];
    if (!item) return;
    
    if (confirm(`저장된 계산 "${item.name}"을(를) 삭제하시겠습니까?`)) {
      savedRibbons.splice(idx, 1);
      saveSavedRibbons();
    }
  });

  // Nanoribbon Export result listener
  document.getElementById('ribbon-export-btn')?.addEventListener('click', () => {
    const selIdx = document.getElementById('ribbon-saved-list').value;
    if (selIdx === '') {
      alert('파일로 내보낼 계산을 선택하세요.');
      return;
    }
    
    const item = savedRibbons[parseInt(selIdx)];
    if (!item) return;
    
    try {
      const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(item, null, 2));
      const downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      const safeName = item.name.replace(/[^a-zA-Z0-9가-힣_ -]/g, '');
      downloadAnchor.setAttribute("download", `ribbon_${safeName}.json`);
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    } catch (err) {
      alert('파일 내보내기 중 오류가 발생했습니다: ' + err.message);
    }
  });

  // Nanoribbon Import result listener
  const ribbonImportFile = document.getElementById('ribbon-import-file');
  document.getElementById('ribbon-import-btn')?.addEventListener('click', () => {
    ribbonImportFile?.click();
  });
  
  ribbonImportFile?.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(evt) {
      try {
        const item = JSON.parse(evt.target.result);
        if (!item || !item.data || !item.spec) {
          throw new Error('유효한 나노리본 계산 저장 파일이 아닙니다.');
        }
        
        // Generate a new ID and update name to avoid duplicates
        item.id = 'ribbon_' + Date.now();
        item.name = item.name + ' (가져옴)';
        
        savedRibbons.push(item);
        saveSavedRibbons();
        
        // Select the newly imported run in dropdown
        const sel = document.getElementById('ribbon-saved-list');
        if (sel) {
          sel.value = savedRibbons.length - 1;
        }
        
        alert(`계산 "${item.name}"을(를) 성공적으로 가져왔습니다!`);
      } catch (err) {
        alert('파일 가져오기 실패: ' + err.message);
      }
    };
    reader.readAsText(file);
    e.target.value = ''; // Reset input to allow importing same file again
  });

  // Flat Band Engineering UI setup
  document.getElementById('eng-sub-count')?.addEventListener('input', rebuildEngSublattices);
  document.getElementById('eng-add-sing-btn')?.addEventListener('click', () => addEngSingularityRow());
  // The derived gauge zeta depends on the PRIMARY singularity (first row); keep
  // the read-only zeta fields in sync when the singularity list changes.
  ['input', 'change'].forEach(ev => {
    document.getElementById('eng-singularities-list')?.addEventListener(ev, engUpdateDerivedZetas);
  });
  document.getElementById('eng-run-btn')?.addEventListener('click', runFlatBandDesign);
  document.getElementById('eng-explore-btn')?.addEventListener('click', runFlatBandExplore);
  document.getElementById('eng-clear-log-btn')?.addEventListener('click', () => {
    const out = document.getElementById('eng-log-output');
    if (out) out.textContent = '';
  });
  document.getElementById('eng-load-model-btn')?.addEventListener('click', loadDesignedModel);

  // Real-space lattice preview controls
  ['eng-a1x', 'eng-a1y', 'eng-a2x', 'eng-a2y'].forEach(id => {
    document.getElementById(id)?.addEventListener('input', () => engBuildLatticeSVG());
  });
  document.getElementById('eng-repx')?.addEventListener('input', (e) => {
    engSt.repX = Math.max(0, parseInt(e.target.value) || 0);
    engBuildLatticeSVG();
  });
  document.getElementById('eng-repy')?.addEventListener('input', (e) => {
    engSt.repY = Math.max(0, parseInt(e.target.value) || 0);
    engBuildLatticeSVG();
  });
  document.getElementById('eng-zoom-in')?.addEventListener('click', () => {
    engSt.zoom = Math.min(5.0, engSt.zoom * 1.25);
    engBuildLatticeSVG();
  });
  document.getElementById('eng-zoom-out')?.addEventListener('click', () => {
    engSt.zoom = Math.max(0.2, engSt.zoom / 1.25);
    engBuildLatticeSVG();
  });

  // Auto / Manual mode toggle
  document.getElementById('eng-mode-auto')?.addEventListener('change', (e) => {
    if (e.target.checked) engSwitchMode('auto');
  });
  document.getElementById('eng-mode-manual')?.addEventListener('change', (e) => {
    if (e.target.checked) engSwitchMode('manual');
  });

  // Manual CLS placement controls
  document.getElementById('eng-manual-clear-btn')?.addEventListener('click', () => {
    engSt.clsSites = [];
    engBuildLatticeSVG();
    engRebuildSitesTable();
  });
  document.getElementById('eng-manual-run-btn')?.addEventListener('click', engRunManualAnalysis);

  // Initialize Flat Band Engineering Defaults
  rebuildEngSublattices();
  addEngSingularityRow('Gamma', '0.0', '0.0', '1');
  engUpdateDerivedZetas();
  engSwitchMode('auto');
});

// ─── Expression Editor Modal Logic ───────────────────────────────────────────
let currentEditRow = null;
let currentEditCol = null;

function openExpressionModal(row, col) {
  currentEditRow = row;
  currentEditCol = col;

  const modal = document.getElementById('expression-modal');
  const title = document.getElementById('modal-title');
  const textarea = document.getElementById('modal-expr-input');
  
  const iLbl = state.orbitals[row]?.label || row;
  const jLbl = state.orbitals[col]?.label || col;
  title.innerHTML = `H<sub>${iLbl}${jLbl}</sub>(k) 수식 편집기`;
  
  const val = state.symbolicMatrix[row][col] || '0';
  textarea.value = val === '0' ? '' : val;
  
  updateModalPreview();
  
  modal.classList.remove('hidden');
  textarea.focus();
}

function closeExpressionModal() {
  const modal = document.getElementById('expression-modal');
  modal.classList.add('hidden');
  currentEditRow = null;
  currentEditCol = null;
}

function openHamiltonianMatrixModal() {
  const modal = document.getElementById('h-matrix-modal');
  if (modal) {
    modal.classList.remove('hidden');
    updateHamiltonianMatrixPreview();
  }
}

function closeHamiltonianMatrixModal() {
  const modal = document.getElementById('h-matrix-modal');
  if (modal) {
    modal.classList.add('hidden');
  }
}

// ─── Hopping Structure Viewer ───────────────────────────────────────────────
// Shows, in real space, HOW the tight-binding hoppings connect orbitals across
// unit cells: a directed arrow from orbital i (central cell) to orbital j
// (cell R), an on-site loop for R=0/i=j, paired with a clean hopping table.
const HOPVIEW_PALETTE = [
  '#2563eb', '#dc2626', '#16a34a', '#d97706', '#7c3aed',
  '#0891b2', '#db2777', '#65a30d', '#9333ea', '#0d9488',
  '#ea580c', '#4f46e5', '#be123c', '#15803d', '#a16207',
];
let hopViewSt = { rep: 1, hc: true, onsite: true };

function openHoppingViewer() {
  const modal = document.getElementById('hopping-viewer-modal');
  if (!modal) return;
  try { syncUIToState(); } catch (_) {}
  modal.classList.remove('hidden');
  // Defer one frame so the SVG has a measured size before we compute the transform.
  requestAnimationFrame(() => renderHoppingViewer());
}

function closeHoppingViewer() {
  const modal = document.getElementById('hopping-viewer-modal');
  if (modal) modal.classList.add('hidden');
}

// Numeric magnitude/phase of a hopping amplitude (number, {re,im}, or a real
// scalar expression like "sqrt(3)/2"). Returns {re, im, mag, phase} or, for an
// un-evaluable symbolic amplitude, {expr}.
function hopAmpInfo(t) {
  let re = 0, im = 0;
  if (typeof t === 'number') {
    re = t;
  } else if (t && typeof t === 'object' && ('re' in t || 'im' in t)) {
    re = Number(t.re) || 0; im = Number(t.im) || 0;
  } else if (typeof t === 'string') {
    const s = t.trim();
    if (s === '' || s === '0') return { re: 0, im: 0, mag: 0, phase: 0 };
    try { re = evalMathExpr(s); }
    catch (_) { return { expr: s }; }   // symbolic (e.g. contains a free param)
  }
  return { re, im, mag: Math.hypot(re, im), phase: Math.atan2(im, re) };
}

// Compact phase string in units of pi (with a degree tooltip handled by caller).
function phaseToStr(phase) {
  if (Math.abs(phase) < 1e-6) return '0';
  const overPi = phase / Math.PI;
  const nice = [[1, 'π'], [-1, '-π'], [0.5, 'π/2'], [-0.5, '-π/2'],
                [0.25, 'π/4'], [-0.25, '-π/4'], [0.75, '3π/4'], [-0.75, '-3π/4']];
  for (const [v, s] of nice) if (Math.abs(overPi - v) < 1e-3) return s;
  return `${overPi.toFixed(2)}π`;
}

// World position of orbital q in cell (n, m).
function hopviewSitePos(orbs, vecs, dim, q, n, m) {
  if (q >= orbs.length) return [0, 0];
  const pos = orbs[q].position || [];
  const f0 = (pos[0] || 0) + n;
  const f1 = (dim >= 2 ? (pos[1] || 0) : 0) + m;
  let wx = 0, wy = 0;
  if (vecs[0]) { wx += f0 * vecs[0][0]; wy += f0 * (vecs[0][1] || 0); }
  if (dim >= 2 && vecs[1]) { wx += f1 * vecs[1][0]; wy += f1 * (vecs[1][1] || 0); }
  return [wx, wy];
}

// Build the displayed hopping list, optionally folding Hermitian conjugate
// partners (i→j,R) & (j→i,−R) into a single "+h.c." entry, and dropping
// on-site terms if requested.
function hopviewBuildList(hops, dim, foldHC, showOnsite) {
  const norm = hops.map((h, idx) => {
    const R = h.R || [];
    return { i: h.i, j: h.j, n: R[0] || 0, m: dim >= 2 ? (R[1] || 0) : 0,
             t: h.t, srcIdx: idx };
  }).filter(h => {
    const onsite = (h.i === h.j && h.n === 0 && h.m === 0);
    return showOnsite || !onsite;
  });

  if (!foldHC) return norm.map(h => ({ ...h, hc: false }));

  const seen = new Map();
  const out = [];
  for (const h of norm) {
    const a = [h.i, h.j, h.n, h.m];
    const b = [h.j, h.i, -h.n, -h.m];
    // canonical = lexicographically smaller of (i,j,n,m) and (j,i,-n,-m)
    let canon = a, cmp = 0;
    for (let k = 0; k < 4; k++) { if (a[k] !== b[k]) { cmp = a[k] < b[k] ? -1 : 1; break; } }
    if (cmp > 0) canon = b;
    const key = canon.join(',');
    const isOnsite = (h.i === h.j && h.n === 0 && h.m === 0);
    if (seen.has(key)) { seen.get(key).hc = !isOnsite; continue; }
    // keep the entry whose own tuple is the canonical one (so the drawn arrow
    // points in the canonical direction); otherwise still keep but flag hc.
    const entry = { ...h, hc: false };
    seen.set(key, entry);
    out.push(entry);
  }
  return out;
}

function renderHoppingViewer() {
  const svg = document.getElementById('hopview-svg');
  const tbody = document.querySelector('#hopview-table tbody');
  if (!svg || !tbody) return;

  hopViewSt.rep = Math.max(0, parseInt(document.getElementById('hopview-rep')?.value) || 0);
  hopViewSt.hc = !!document.getElementById('hopview-hc')?.checked;
  hopViewSt.onsite = !!document.getElementById('hopview-onsite')?.checked;

  const dim = state.dimension;
  const vecs = state.primitiveVectors;
  const orbs = state.orbitals;
  const hops = state.hoppings || [];
  svg.innerHTML = '';
  tbody.innerHTML = '';

  if (!orbs.length) {
    document.getElementById('hopview-count').textContent = '(궤도 없음)';
    return;
  }

  latEnsureOrbStyles();
  const list = hopviewBuildList(hops, dim, hopViewSt.hc, hopViewSt.onsite);
  const countEl = document.getElementById('hopview-count');
  if (hops.length === 0) {
    countEl.textContent = '(호핑 없음 — 심볼릭 행렬 모드이거나 hopping이 정의되지 않았습니다)';
  } else {
    countEl.textContent = `(${list.length}개${hopViewSt.hc ? ', h.c. 묶음' : ''})`;
  }

  // Draw rep must cover the farthest hopping target so every arrow's endpoint
  // is a visible site.
  let maxRange = 0;
  for (const h of list) maxRange = Math.max(maxRange, Math.abs(h.n), Math.abs(h.m));
  const drawRep = Math.max(hopViewSt.rep, maxRange);
  const repX = drawRep, repY = dim >= 2 ? drawRep : 0;

  const { w2s } = latGetTransformGeneric({
    dimension: dim, primitiveVectors: vecs, orbitals: orbs,
    repX, repY, zoom: 1.0, panX: 0, panY: 0,
  }, svg);

  const L = {};
  ['cells', 'bonds', 'sites', 'labels'].forEach(n => { L[n] = svgE('g'); svg.appendChild(L[n]); });

  // Faint unit-cell grid (helps read which cell R a target sits in).
  if (dim >= 2 && vecs[0] && vecs[1]) {
    for (let n = -repX; n <= repX; n++) for (let m = -repY; m <= repY; m++) {
      const ox = n * vecs[0][0] + m * vecs[1][0], oy = n * (vecs[0][1] || 0) + m * (vecs[1][1] || 0);
      const pts = [[ox, oy], [ox + vecs[0][0], oy + (vecs[0][1] || 0)],
        [ox + vecs[0][0] + vecs[1][0], oy + (vecs[0][1] || 0) + (vecs[1][1] || 0)],
        [ox + vecs[1][0], oy + (vecs[1][1] || 0)]]
        .map(([wx, wy]) => w2s(wx, wy)).map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
      const isO = n === 0 && m === 0;
      L.cells.appendChild(svgE('polygon', {
        points: pts, fill: isO ? 'rgba(37,99,235,0.06)' : 'none',
        stroke: isO ? '#4c7aff' : '#dbe3f0', 'stroke-width': isO ? '1.4' : '0.7',
        'stroke-dasharray': isO ? '5,3' : '3,3',
      }));
    }
  }

  const siteR = 12;
  // Orbital markers (origin cell solid, neighbours dimmed).
  for (let n = -repX; n <= repX; n++) for (let m = -repY; m <= repY; m++) {
    const isOrigin = n === 0 && m === 0;
    orbs.forEach((orb, qi) => {
      const [wx, wy] = hopviewSitePos(orbs, vecs, dim, qi, n, m);
      const [sx, sy] = w2s(wx, wy);
      const color = latSt.orbColors[qi] || LAT_PALETTE[qi % LAT_PALETTE.length];
      latDrawSiteShape(L.sites, sx, sy, isOrigin ? siteR : siteR * 0.7, color,
                       latSt.orbShapes[qi] || 'circle', isOrigin ? 1 : 0.28);
      if (isOrigin) {
        const fo = svgE('foreignObject', { x: sx + siteR + 1, y: sy - 11, width: '70', height: '24' });
        const div = document.createElementNS('http://www.w3.org/1999/xhtml', 'div');
        div.className = 'lat-label lat-site-label';
        try { katex.render(/^[A-Za-z0-9]$/.test(orb.label) ? orb.label : `\\text{${orb.label}}`, div, { throwOnError: false }); }
        catch (_) { div.textContent = orb.label; }
        fo.appendChild(div); L.labels.appendChild(fo);
      }
    });
  }

  // Bonds: one element per displayed hopping, tagged with data-hop for hover.
  const arrowEls = [];
  list.forEach((h, idx) => {
    const color = HOPVIEW_PALETTE[idx % HOPVIEW_PALETTE.length];
    const [wx1, wy1] = hopviewSitePos(orbs, vecs, dim, h.i, 0, 0);
    const [wx2, wy2] = hopviewSitePos(orbs, vecs, dim, h.j, h.n, h.m);
    const [x1, y1] = w2s(wx1, wy1);
    const [x2, y2] = w2s(wx2, wy2);
    const g = svgE('g', { 'data-hop': idx, opacity: '0.5', style: 'cursor:pointer;' });
    const dx = x2 - x1, dy = y2 - y1, len = Math.hypot(dx, dy);
    if (len < 1e-3) {
      // on-site (or same screen point): a small loop above the site
      const r = siteR + 6;
      g.appendChild(svgE('circle', { cx: x1, cy: y1 - r, r: 7, fill: 'none', stroke: color, 'stroke-width': '2.4' }));
    } else {
      const ux = dx / len, uy = dy / len;
      const x1s = x1 + ux * (siteR + 1), y1s = y1 + uy * (siteR + 1);
      const x2s = x2 - ux * (siteR + 4), y2s = y2 - uy * (siteR + 4);
      g.appendChild(svgE('line', { x1: x1s, y1: y1s, x2: x2s, y2: y2s, stroke: color, 'stroke-width': '2.4', 'stroke-linecap': 'round' }));
      // arrowhead
      const ah = 7, aw = 4;
      const bx = x2s - ux * ah, by = y2s - uy * ah;
      const px = -uy, py = ux;
      g.appendChild(svgE('polygon', {
        points: `${x2s.toFixed(1)},${y2s.toFixed(1)} ${(bx + px * aw).toFixed(1)},${(by + py * aw).toFixed(1)} ${(bx - px * aw).toFixed(1)},${(by - py * aw).toFixed(1)}`,
        fill: color,
      }));
    }
    L.bonds.appendChild(g);
    arrowEls[idx] = g;
  });

  // Table rows.
  list.forEach((h, idx) => {
    const color = HOPVIEW_PALETTE[idx % HOPVIEW_PALETTE.length];
    const info = hopAmpInfo(h.t);
    const li = orbs[h.i], lj = orbs[h.j];
    const ci = latSt.orbColors[h.i] || LAT_PALETTE[h.i % LAT_PALETTE.length];
    const cj = latSt.orbColors[h.j] || LAT_PALETTE[h.j % LAT_PALETTE.length];
    const onsite = (h.i === h.j && h.n === 0 && h.m === 0);

    const tr = document.createElement('tr');
    tr.dataset.hop = idx;
    tr.style.cssText = 'border-bottom:1px solid #f1f5f9; cursor:pointer;';

    const dot = (c) => `<span style="display:inline-block;width:9px;height:9px;border-radius:50%;background:${c};border:1px solid ${latDarken(c)};vertical-align:middle;margin:0 2px;"></span>`;
    const swatch = `<span style="display:inline-block;width:11px;height:11px;border-radius:3px;background:${color};vertical-align:middle;"></span>`;

    let ampMag, ampPhase;
    if (info.expr !== undefined) { ampMag = info.expr; ampPhase = '—'; }
    else {
      ampMag = (typeof roundSig === 'function' ? roundSig(info.mag, 4) : info.mag.toFixed(4));
      ampPhase = onsite ? '—' : phaseToStr(info.phase);
    }
    const bondLabel = onsite
      ? `${dot(ci)}${escapeHtmlSafe(li?.label ?? h.i)} <span style="color:#94a3b8;">(온사이트)</span>`
      : `${dot(ci)}${escapeHtmlSafe(li?.label ?? h.i)} <span style="color:#64748b;">→</span> ${dot(cj)}${escapeHtmlSafe(lj?.label ?? h.j)}`;
    const hcBadge = h.hc ? ` <span style="font-size:0.66rem;color:#2563eb;background:#eff6ff;border-radius:3px;padding:0 3px;">+h.c.</span>` : '';

    tr.innerHTML =
      `<td style="padding:4px 6px;white-space:nowrap;">${swatch} <span style="color:#94a3b8;">${idx + 1}</span></td>` +
      `<td style="padding:4px 6px;">${bondLabel}${hcBadge}</td>` +
      `<td style="padding:4px 6px;white-space:nowrap;color:#475569;">(${h.n}, ${h.m})</td>` +
      `<td style="padding:4px 6px;white-space:nowrap;">${ampMag}</td>` +
      `<td style="padding:4px 6px;white-space:nowrap;color:#475569;" title="${info.expr !== undefined ? '' : (info.phase * 180 / Math.PI).toFixed(1) + '°'}">${ampPhase}</td>`;

    const hi = () => hopviewHighlight(idx);
    const un = () => hopviewHighlight(null);
    tr.addEventListener('mouseenter', hi);
    tr.addEventListener('mouseleave', un);
    if (arrowEls[idx]) {
      arrowEls[idx].addEventListener('mouseenter', hi);
      arrowEls[idx].addEventListener('mouseleave', un);
    }
    tbody.appendChild(tr);
  });
}

function hopviewHighlight(idx) {
  const svg = document.getElementById('hopview-svg');
  const tbody = document.querySelector('#hopview-table tbody');
  if (!svg) return;
  svg.querySelectorAll('[data-hop]').forEach(g => {
    const i = parseInt(g.dataset.hop);
    if (idx === null) { g.setAttribute('opacity', '0.5'); g.style.filter = ''; }
    else if (i === idx) { g.setAttribute('opacity', '1'); g.style.filter = 'drop-shadow(0 0 2px rgba(0,0,0,0.35))'; }
    else { g.setAttribute('opacity', '0.1'); g.style.filter = ''; }
  });
  if (tbody) tbody.querySelectorAll('tr').forEach(tr => {
    tr.style.background = (idx !== null && parseInt(tr.dataset.hop) === idx) ? '#eff6ff' : '';
  });
}

function escapeHtmlSafe(s) {
  return String(s).replace(/[&<>"']/g, c => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

function applyExpressionModal() {
  if (currentEditRow === null || currentEditCol === null) return;
  
  const textarea = document.getElementById('modal-expr-input');
  const val = textarea.value.trim() || '0';
  
  // Update state and input field
  state.symbolicMatrix[currentEditRow][currentEditCol] = val;
  const inp = document.getElementById(`sym-${currentEditRow}-${currentEditCol}`);
  if (inp) {
    inp.value = val;
    // Trigger input event to update KaTeX and Hermitian conjugations
    inp.dispatchEvent(new Event('input'));
  }
  
  closeExpressionModal();
  triggerAutoSave();
}

function updateModalPreview() {
  const textarea = document.getElementById('modal-expr-input');
  const preview = document.getElementById('modal-math-preview');
  const val = textarea.value.trim() || '0';
  renderKatex(preview, exprToLatex(val));
}

function insertSymbolInModal(symbol) {
  const textarea = document.getElementById('modal-expr-input');
  if (!textarea) return;
  
  const start = textarea.selectionStart;
  const end = textarea.selectionEnd;
  const val = textarea.value;
  
  textarea.value = val.slice(0, start) + symbol + val.slice(end);
  textarea.focus();
  const pos = start + symbol.length;
  textarea.setSelectionRange(pos, pos);
  
  updateModalPreview();
}

// ─── Nanoribbon Analysis ─────────────────────────────────────────────────────
let ribbonLastData = null;
let lastRibbonQ = null;

function rebuildRibbonBandSelector() {
  const container = document.getElementById('ribbon-band-checkboxes');
  if (!container) return;
  
  const Q = state.orbitals.length;
  if (Q === lastRibbonQ) {
    return;
  }
  
  container.innerHTML = '';
  lastRibbonQ = Q;
  
  for (let i = 0; i < Q; i++) {
    const label = document.createElement('label');
    label.style.display = 'flex';
    label.style.alignItems = 'center';
    label.style.gap = '4px';
    label.style.fontSize = '0.8rem';
    label.style.margin = '0';
    label.style.cursor = 'pointer';
    label.style.fontWeight = '500';
    label.style.color = '#334155';
    
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.value = i;
    cb.className = 'ribbon-band-cb';
    cb.checked = true; // default checked
    
    label.appendChild(cb);
    label.appendChild(document.createTextNode(`밴드 ${i}`));
    container.appendChild(label);
  }
}

async function runNanoribbonAnalysis() {
  const spec = buildSpec();
  const dim = spec.lattice ? parseInt(spec.lattice.dimension) : parseInt(state.dimension);
  if (dim !== 2) {
    alert("나노리본 분석은 2차원(2D) 격자 모델만 지원합니다.");
    return;
  }

  const periodicDir = document.getElementById('ribbon-pdir').value || 'y';
  const Nx = parseInt(document.getElementById('ribbon-nx').value) || 40;
  const Nk = parseInt(document.getElementById('ribbon-nk').value) || 200;
  const threshold = parseFloat(document.getElementById('ribbon-threshold').value) || 0.5;

  let kyMin, kyMax;
  try {
    kyMin = evalMathExpr(document.getElementById('ribbon-kmin').value);
  } catch (_) {
    alert(`k_${periodicDir} 범위의 시작 값이 올바르지 않습니다. (예: -pi, -3.14)`);
    return;
  }
  try {
    kyMax = evalMathExpr(document.getElementById('ribbon-kmax').value);
  } catch (_) {
    alert(`k_${periodicDir} 범위의 끝 값이 올바르지 않습니다. (예: pi, 3.14)`);
    return;
  }

  if (kyMin >= kyMax) {
    alert(`k_${periodicDir} 시작 범위가 끝 범위보다 크거나 같을 수 없습니다.`);
    return;
  }

  let kFixed;
  try {
    kFixed = evalMathExpr(document.getElementById('ribbon-kfixed').value);
  } catch (_) {
    alert(`고정된 k_${periodicDir === 'y' ? 'x' : 'y'} 값이 올바르지 않습니다. (예: 0, pi/2)`);
    return;
  }

  const checkboxes = document.querySelectorAll('.ribbon-band-cb');
  const selectedBands = [];
  checkboxes.forEach(cb => {
    if (cb.checked) {
      selectedBands.push(parseInt(cb.value));
    }
  });

  const loading = document.getElementById('ribbon-loading');

  if (selectedBands.length === 0) {
    alert("최소 하나의 벌크 밴드를 선택해야 합니다.");
    if (loading) loading.classList.add('hidden');
    return;
  }

  if (loading) loading.classList.remove('hidden');

  try {
    const res = await fetch('/api/nanoribbon_data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ 
        spec, 
        Nx, 
        Nk, 
        selected_bands: selectedBands,
        ky_min: kyMin,
        ky_max: kyMax,
        periodic_dir: periodicDir,
        k_fixed: kFixed
      })
    });
    const data = await res.json();
    if (loading) loading.classList.add('hidden');

    if (data.error) {
      alert(`에러: ${data.error}`);
      return;
    }

    ribbonLastData = data;
    const container = document.getElementById('ribbon-dispersion-plot');
    if (container) {
      container.ribbonSpec = spec;
      container.ribbonNx = Nx;
    }
    const selPreset = document.getElementById('preset-select');
    const modelName = (selPreset && selPreset.value) ? selPreset.selectedOptions[0].text : '사용자 모델';
    const nameInp = document.getElementById('ribbon-save-name');
    if (nameInp) {
      nameInp.value = `${modelName} (N=${Nx}, k_${periodicDir})`;
    }
    const dataToRender = { ...data, id: 'ribbon_' + Date.now() };
    renderRibbonDispersion(dataToRender, threshold, periodicDir);
  } catch (err) {
    if (loading) loading.classList.add('hidden');
    console.error(err);
    alert(`계산 오류: ${err.message}`);
  }
}

// ─── Robust Boundary Mode (RBM) ───────────────────────────────────────────────
async function runRBMAnalysis() {
  const spec = buildSpec();
  const dim = spec.lattice ? parseInt(spec.lattice.dimension) : parseInt(state.dimension);
  if (dim !== 2) {
    alert("경계 모드(RBM) 시각화는 현재 2차원(2D) 격자만 지원합니다.");
    return;
  }
  const Nx = parseInt(document.getElementById('rbm-nx').value) || 16;
  const Ny = parseInt(document.getElementById('rbm-ny').value) || Nx;
  const bandStr = (document.getElementById('rbm-band').value || '').trim();
  const band_index = bandStr === '' ? null : parseInt(bandStr);
  const defStr = (document.getElementById('rbm-defect').value || '').trim();
  let defect_cell = null;
  if (defStr) {
    defect_cell = defStr.split(/[,\s]+/).map(s => parseInt(s)).filter(n => !isNaN(n));
    if (defect_cell.length !== 2) {
      alert("결함 셀은 'nx,ny' 형식의 정수 두 개여야 합니다 (예: 8,8).");
      return;
    }
  }

  const loading = document.getElementById('rbm-loading');
  if (loading) loading.classList.remove('hidden');
  try {
    const res = await fetch('/api/rbm_data', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spec, Nx, Ny, band_index, defect_cell })
    });
    const data = await res.json();
    if (loading) loading.classList.add('hidden');
    if (!res.ok || data.detail) {
      const msg = data.detail ? JSON.stringify(data.detail) : `HTTP ${res.status}`;
      alert(`요청 오류: ${msg}`);
      return;
    }
    if (data.error) { alert(`에러: ${data.error}`); return; }
    rbmLastData = data;
    renderRBMSummary(data);
    renderRBMMap(data);
    renderRBMSkin(data);
  } catch (err) {
    if (loading) loading.classList.add('hidden');
    console.error(err);
    alert(`계산 오류: ${err.message}`);
  }
}

let rbmLastData = null;

function renderRBMSummary(data) {
  const box = document.getElementById('rbm-summary');
  if (!box) return;
  box.style.display = 'flex';
  const sing = data.singular;
  const singTxt = sing === null ? '판정 불가' : (sing ? '특이형 (Singular)' : '비특이형 (Nonsingular)');
  const singColor = sing ? '#0d9488' : '#b45309';
  const v = data.validation || {};
  const passed = v.passed;
  document.getElementById('rbm-stat-singular').textContent = `위상: ${singTxt}`;
  document.getElementById('rbm-stat-singular').style.color = singColor;
  const bulkEl = document.getElementById('rbm-stat-bulk');
  bulkEl.textContent = `Bulk 소멸: ${passed ? '✓ 통과' : '✗ 실패'} (max|ψ|_bulk = ${(v.max_bulk_amp ?? 0).toExponential(2)})`;
  bulkEl.style.color = passed ? '#16a34a' : '#dc2626';
  document.getElementById('rbm-stat-radius').textContent = `CLS 반경 = ${data.support_radius}`;
  document.getElementById('rbm-stat-sites').textContent = `경계 사이트 = ${data.n_nonzero_sites}  (E=${(data.energy ?? 0).toFixed(3)}, M=${data.M})`;
  const defEl = document.getElementById('rbm-stat-defect');
  if (data.defect) {
    const dv = data.defect.validation || {};
    defEl.textContent = `결함 @[${data.defect.cell}] → Bulk 소멸 ${dv.passed ? '유지' : '국소 붕괴'} (max|ψ|_bulk=${(dv.max_bulk_amp ?? 0).toExponential(2)})`;
  } else {
    defEl.textContent = '';
  }
}

function renderRBMMap(data) {
  const container = document.getElementById('rbm-map-plot');
  if (!container) return;
  const sites = (data.defect ? data.defect.sites : data.sites) || [];
  const nz = sites.filter(s => s.nonzero);

  // background faint grid: all lattice sites if provided via sites? we only have
  // nonzero; draw a light bounding frame using nonzero extents + system size.
  const traces = [];

  // marker color: real -> red/blue by sign; complex -> phase colorscale
  const anyComplex = nz.some(s => Math.abs(s.amp_im) > 1e-9);
  const maxAbs = nz.reduce((m, s) => Math.max(m, s.abs), 0) || 1;

  if (anyComplex) {
    traces.push({
      type: 'scatter', mode: 'markers',
      x: nz.map(s => s.x), y: nz.map(s => s.y),
      marker: {
        size: nz.map(s => 8 + 26 * (s.abs / maxAbs)),
        color: nz.map(s => s.phase),
        colorscale: 'HSV', cmin: -Math.PI, cmax: Math.PI,
        colorbar: { title: 'arg(ψ)', thickness: 12 },
        line: { color: '#333', width: 0.5 }
      },
      text: nz.map(s => `cell ${JSON.stringify(s.cell)} · ${s.label}<br>|ψ|=${s.abs.toFixed(3)} · arg=${s.phase.toFixed(2)}`),
      hoverinfo: 'text', name: 'ψ_∂'
    });
  } else {
    const pos = nz.filter(s => s.amp_re >= 0), neg = nz.filter(s => s.amp_re < 0);
    const mk = (arr, color, nm) => ({
      type: 'scatter', mode: 'markers',
      x: arr.map(s => s.x), y: arr.map(s => s.y),
      marker: { size: arr.map(s => 8 + 26 * (s.abs / maxAbs)), color,
                line: { color: '#333', width: 0.5 } },
      text: arr.map(s => `cell ${JSON.stringify(s.cell)} · ${s.label}<br>ψ=${s.amp_re.toFixed(3)}`),
      hoverinfo: 'text', name: nm
    });
    traces.push(mk(pos, '#d62728', 'ψ > 0'));
    traces.push(mk(neg, '#1f77b4', 'ψ < 0'));
  }

  const passed = data.validation && data.validation.passed;
  const layout = {
    title: { text: `실공간 경계 모드 ${passed ? '(Bulk 소멸 ✓)' : '(Bulk 잔존 ✗)'}`, font: { size: 14 } },
    xaxis: { title: 'x', scaleanchor: 'y', scaleratio: 1, zeroline: false },
    yaxis: { title: 'y', zeroline: false },
    margin: { l: 50, r: 20, t: 40, b: 45 },
    showlegend: !anyComplex, plot_bgcolor: '#fafafa'
  };
  Plotly.newPlot(container, traces, layout, { responsive: true, displaylogo: false });
}

function renderRBMSkin(data) {
  const container = document.getElementById('rbm-skin-plot');
  if (!container) return;
  const sd = data.skin_depth || {};
  const traces = [];
  const palette = ['#7c3aed', '#0d9488', '#ea580c'];
  Object.keys(sd).forEach((ax, i) => {
    const p = sd[ax];
    traces.push({
      type: 'scatter', mode: 'lines+markers',
      x: p.distance, y: p.max_amp,
      line: { shape: 'hvh', color: palette[i % palette.length], width: 2 },
      marker: { size: 5 },
      name: `axis ${ax}`
    });
  });
  const layout = {
    title: { text: 'Skin Depth (Step-function 기대)', font: { size: 14 } },
    xaxis: { title: '경계로부터의 층 index' },
    yaxis: { title: 'max |ψ_∂|', rangemode: 'tozero' },
    margin: { l: 55, r: 20, t: 40, b: 45 }, showlegend: true
  };
  Plotly.newPlot(container, traces, layout, { responsive: true, displaylogo: false });
}

function renderRibbonDispersion(data, threshold, periodicDir = 'y') {
  const container = document.getElementById('ribbon-dispersion-plot');
  if (!container) return;

  window.fetchNanoribbonState = fetchNanoribbonState;

  const openNewWindow = document.getElementById('ribbon-new-window')?.checked;
  let targetContainer = container;
  let targetWindow = window;
  let plotWin = null;

  if (openNewWindow) {
    const winName = "NanoribbonPlotWindow_" + (data.id || Date.now());
    plotWin = window.open("plot.html", winName, "width=1200,height=800,resizable=yes,scrollbars=yes");
    if (!plotWin) {
      alert("팝업이 차단되었습니다! 주소창 우측에서 팝업 허용을 설정해 주세요.");
      return;
    }
    
    Plotly.purge(container);
    container.innerHTML = '<div style="display:flex; justify-content:center; align-items:center; height:100%; color:#64748b; font-size:0.85rem; font-weight:500; border: 1.5px dashed #cbd5e1; border-radius:6px; margin: 10px 0;">분산 그래프가 새 창으로 열려 있습니다.</div>';
  } else {
    container.innerHTML = '';
  }

  const ky_space = data.ky_space;
  const energies = data.energies;
  const edge_weights = data.edge_weights;
  const iprs = data.iprs;
  const lo = data.lo !== undefined ? data.lo : 0;

  const n_bands = energies.length; // num_selected
  const Nk = ky_space.length;

  const flatKy = [];
  const flatE = [];
  const flatWeight = [];
  const flatIpr = [];
  const flatBandIdx = [];
  const hoverText = [];

  for (let n = 0; n < n_bands; n++) {
    for (let ik = 0; ik < Nk; ik++) {
      const ky = ky_space[ik];
      const E = energies[n][ik];
      const w = edge_weights[n][ik];
      const ipr = iprs[n][ik];
      
      flatKy.push(ky);
      flatE.push(E);
      flatWeight.push(w);
      flatIpr.push(ipr);
      flatBandIdx.push(lo + n);

      const isEdge = w >= threshold ? "경계 상태 (Edge)" : "벌크 상태 (Bulk)";
      hoverText.push(
        `k_${periodicDir}: ${ky.toFixed(4)}<br>` +
        `에너지: ${E.toFixed(4)}<br>` +
        `에지 가중치: ${(w*100).toFixed(1)}%<br>` +
        `IPR: ${ipr.toFixed(3)}<br>` +
        `밴드 인덱스: ${lo + n}<br>` +
        `구분: ${isEdge}`
      );
    }
  }

  const traces = [];

  // Add bulk band traces first, so they are drawn behind the ribbon states
  if (data.bulk_energies) {
    const n_bulk_bands = data.bulk_energies.length;
    for (let j = 0; j < n_bulk_bands; j++) {
      traces.push({
        x: ky_space,
        y: data.bulk_energies[j],
        mode: 'lines',
        type: 'scattergl',
        line: {
          color: '#94a3b8', // slate-400
          width: 1.5,
          dash: 'dash'
        },
        name: '벌크 밴드',
        hoverinfo: 'skip',
        showlegend: j === 0,
        legendgroup: 'bulk'
      });
    }
  }

  const trace = {
    x: flatKy,
    y: flatE,
    mode: 'markers',
    type: 'scattergl',
    name: '리본 상태',
    marker: {
      size: 4,
      color: flatWeight,
      colorscale: [
        [0.0, '#334155'],
        [0.3, '#64748b'],
        [threshold, '#f97316'],
        [1.0, '#ef4444']
      ],
      showscale: true,
      colorbar: {
        title: 'Edge Weight',
        titleside: 'right',
        thickness: 15,
        len: 0.8
      }
    },
    text: hoverText,
    hoverinfo: 'text'
  };
  traces.push(trace);

  const layout = {
    xaxis: { 
      title: `k_${periodicDir} (Momentum)`, 
      gridcolor: '#f1f5f9', 
      zeroline: false,
      autorange: true
    },
    yaxis: { 
      title: 'Energy', 
      gridcolor: '#f1f5f9', 
      zeroline: false,
      autorange: true
    },
    margin: { t: 40, l: 40, r: 10, b: 40 }, // increased top margin to accommodate legend
    hovermode: 'closest',
    plot_bgcolor: '#fafbff',
    paper_bgcolor: '#fff',
    font: { family: 'Segoe UI, sans-serif' },
    showlegend: !!data.bulk_energies,
    legend: {
      orientation: 'h',
      x: 0,
      y: 1.1,
      xanchor: 'left',
      yanchor: 'bottom'
    }
  };

  function drawPlot(tgtWindow, tgtContainer) {
    tgtWindow.Plotly.newPlot(tgtContainer, traces, layout, { responsive: true });

    // Store spec and Nx inside the target container
    const mainContainer = document.getElementById('ribbon-dispersion-plot');
    tgtContainer.ribbonSpec = mainContainer?.ribbonSpec || buildSpec();
    tgtContainer.ribbonNx = mainContainer?.ribbonNx || parseInt(document.getElementById('ribbon-nx').value) || 40;

    if (typeof tgtContainer.off === 'function') {
      tgtContainer.off('plotly_click');
    }
    tgtContainer.on('plotly_click', function(clickData) {
      if (clickData.points.length > 0) {
        const pt = clickData.points[0];
        // Only process click if the clicked curve is the ribbon scatter trace (the last one)
        if (pt.curveNumber === traces.length - 1) {
          const pointIdx = pt.pointNumber;
          const n = flatBandIdx[pointIdx];
          const ky = flatKy[pointIdx];
          
          const specToUse = tgtContainer.ribbonSpec;
          const NxToUse = tgtContainer.ribbonNx;
          
          if (typeof fetchNanoribbonState === 'function') {
            fetchNanoribbonState(ky, n, periodicDir, specToUse, NxToUse);
          } else if (window.fetchNanoribbonState) {
            window.fetchNanoribbonState(ky, n, periodicDir, specToUse, NxToUse);
          } else if (window.opener && window.opener.fetchNanoribbonState) {
            window.opener.fetchNanoribbonState(ky, n, periodicDir, specToUse, NxToUse);
          }
        }
      }
    });
  }

  if (openNewWindow) {
    // Poll until the plot window has loaded plot.html and Plotly is defined
    const checkInterval = setInterval(() => {
      if (plotWin.closed) {
        clearInterval(checkInterval);
        return;
      }
      try {
        if (plotWin.Plotly && plotWin.document.getElementById('plot')) {
          clearInterval(checkInterval);
          const tgtContainer = plotWin.document.getElementById('plot');
          drawPlot(plotWin, tgtContainer);
        }
      } catch (e) {
        // Ignore cross-origin error that may transiently occur during initial load
      }
    }, 20);
  } else {
    drawPlot(window, container);
  }

  const selectedInfo = document.getElementById('ribbon-selected-info');
  if (selectedInfo) selectedInfo.textContent = "분산 플롯에서 분석하고 싶은 상태(점)를 클릭하세요.";
  Plotly.purge('ribbon-wf-plot');
  Plotly.purge('ribbon-lattice-plot');
}

async function fetchNanoribbonState(ky, bandIdx, periodicDir = 'y', specOverride = null, NxOverride = null) {
  const spec = specOverride || buildSpec();
  const Nx = NxOverride || parseInt(document.getElementById('ribbon-nx').value) || 40;
  const infoEl = document.getElementById('ribbon-selected-info');
  if (infoEl) infoEl.textContent = `선택된 상태 불러오는 중... (k_${periodicDir}: ${ky.toFixed(3)}, Band: ${bandIdx})`;

  try {
    const res = await fetch('/api/nanoribbon_state', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ spec, Nx, ky, band_idx: bandIdx, periodic_dir: periodicDir })
    });
    const data = await res.json();

    if (data.error) {
      if (infoEl) infoEl.textContent = `오류: ${data.error}`;
      return;
    }

    if (infoEl) {
      infoEl.innerHTML = `선택된 상태: <span style="color:#ef4444; font-weight:700;">k_${periodicDir} = ${ky.toFixed(4)}</span>, ` +
                         `에너지 = <span style="color:#2563eb; font-weight:700;">${data.energy.toFixed(5)}</span>, ` +
                         `IPR = <span style="font-weight:700;">${data.ipr.toFixed(3)}</span>, ` +
                         `에지 가중치 = <span style="font-weight:700;">${(data.edge_weight*100).toFixed(1)}%</span>`;
    }

    renderStateWavefunction(data);
    renderStateLatticeConfinement(data);
  } catch (err) {
    if (infoEl) infoEl.textContent = `불러오기 실패: ${err.message}`;
    console.error(err);
  }
}

function renderStateWavefunction(data) {
  const container = document.getElementById('ribbon-wf-plot');
  if (!container) return;

  const Nx = data.layer_prob.length;
  const layers = Array.from({length: Nx}, (_, i) => i + 1);

  const trace = {
    x: layers,
    y: data.layer_prob,
    type: 'bar',
    marker: {
      color: '#7c3aed',
      line: { color: '#6d28d9', width: 1 }
    },
    hovertemplate: '층: %{x}<br>확률밀도: %{y:.4f}<extra></extra>'
  };

  const layout = {
    xaxis: { title: 'Layer Index (x)', dtick: Math.max(1, Math.floor(Nx / 10)), gridcolor: '#f1f5f9' },
    yaxis: { title: 'Probability Density', min: 0, gridcolor: '#f1f5f9' },
    margin: { t: 10, l: 50, r: 15, b: 35 },
    plot_bgcolor: '#fafbff',
    paper_bgcolor: '#fff',
    font: { family: 'Segoe UI, sans-serif' }
  };

  Plotly.newPlot(container, [trace], layout, { responsive: true });
}

function renderStateLatticeConfinement(data) {
  const container = document.getElementById('ribbon-lattice-plot');
  if (!container) return;

  const sites = data.sites;
  const xs = sites.map(s => s.x);
  const ys = sites.map(s => s.y);
  const probs = sites.map(s => s.prob);

  const maxProb = Math.max(...probs, 1e-9);

  const hoverText = sites.map(s => 
    `층: ${s.layer + 1}<br>` +
    `오비탈: ${s.label}<br>` +
    `확률밀도: ${s.prob.toFixed(5)}<br>` +
    `복소진폭: ${s.re.toFixed(4)} + ${s.im.toFixed(4)}i`
  );

  const trace = {
    x: xs,
    y: ys,
    mode: 'markers',
    type: 'scattergl',
    marker: {
      size: probs.map(p => 6 + 18 * (p / maxProb)),
      color: probs,
      colorscale: 'Viridis',
      showscale: false,
      line: { color: '#0f172a', width: 1 }
    },
    text: hoverText,
    hoverinfo: 'text'
  };

  const layout = {
    xaxis: { title: '', showgrid: false, zeroline: false, showticklabels: false },
    yaxis: { title: '', showgrid: false, zeroline: false, showticklabels: false, scaleanchor: 'x' },
    margin: { t: 5, l: 10, r: 10, b: 5 },
    plot_bgcolor: '#f8fafc',
    paper_bgcolor: '#fff',
    font: { family: 'Segoe UI, sans-serif' }
  };

  Plotly.newPlot(container, [trace], layout, { responsive: true });
}

let savedRibbons = [];

function loadSavedRibbons() {
  try {
    const raw = localStorage.getItem('cls_saved_ribbons');
    if (raw) {
      savedRibbons = JSON.parse(raw);
    } else {
      savedRibbons = [];
    }
  } catch (e) {
    console.error('Failed to load saved ribbons', e);
    savedRibbons = [];
  }
  updateSavedRibbonsDropdown();
}

function saveSavedRibbons() {
  try {
    localStorage.setItem('cls_saved_ribbons', JSON.stringify(savedRibbons));
  } catch (e) {
    console.error('Failed to save saved ribbons to localStorage', e);
    alert('브라우저 저장 공간이 가득 차 일부 데이터가 저장되지 않았을 수 있습니다. 파일 내보내기 기능을 이용해 안전하게 백업하세요.');
  }
  updateSavedRibbonsDropdown();
}

function updateSavedRibbonsDropdown() {
  const sel = document.getElementById('ribbon-saved-list');
  if (!sel) return;
  
  sel.innerHTML = '';
  if (savedRibbons.length === 0) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = '— 저장된 계산 없음 —';
    sel.appendChild(opt);
    return;
  }
  
  const optPlaceholder = document.createElement('option');
  optPlaceholder.value = '';
  optPlaceholder.textContent = '— 선택하세요 —';
  sel.appendChild(optPlaceholder);
  
  savedRibbons.forEach((item, index) => {
    const opt = document.createElement('option');
    opt.value = index;
    opt.textContent = item.name || `계산 #${index + 1}`;
    sel.appendChild(opt);
  });
}


// ─── Flat Band Engineering Panel Logic ───────────────────────────────────────
let engLastResult = null;
let engSt = {
  repX: 2, repY: 2,
  zoom: 1.0, panX: 0, panY: 0,
  mode: 'auto',
  clsSites: [],          // [{alpha, n, m, A, theta}]
  candidates: [],        // progress.candidate entries, indexed by candidate.index
  selectedCandidate: -1,
  exploreTarget: null,
  explorePrimVecs: null,
  exploreOrbitals: null,
  exploreRanked: [],
};

const ENG_SHAPES = ['circle', 'square', 'diamond', 'triangle', 'triangle-down', 'pentagon', 'hexagon', 'star', 'cross', 'x', 'hourglass', 'bowtie'];

function engGetLatticeSpec() {
  const a1x = evalMathExpr(document.getElementById('eng-a1x').value);
  const a1y = evalMathExpr(document.getElementById('eng-a1y').value);
  const a2x = evalMathExpr(document.getElementById('eng-a2x').value);
  const a2y = evalMathExpr(document.getElementById('eng-a2y').value);

  const sublattices = [];
  document.querySelectorAll('#eng-zetas-container .eng-sub-row').forEach(row => {
    const idx = parseInt(row.dataset.idx);
    const label = String.fromCharCode(65 + idx);
    const zeta = evalMathExpr(row.querySelector('.eng-zeta-input').value) || 0.0;
    const px = evalMathExpr(row.querySelector('.eng-pos-x').value) || 0.0;
    const py = evalMathExpr(row.querySelector('.eng-pos-y').value) || 0.0;
    sublattices.push({ label, zeta, position: [px, py] });
  });

  return {
    primitive_vectors: [[a1x, a1y], [a2x, a2y]],
    sublattices: sublattices,
  };
}

// The primary singularity's fractional reciprocal coordinates (f1, f2) drive
// the derived gauge zeta. Defaults to Gamma=(0,0) if no singularity exists yet.
function engFirstSingularityKFrac() {
  const row = document.querySelector('#eng-singularities-list .eng-sing-row');
  if (!row) return [0.0, 0.0];
  try {
    const f1 = evalMathExpr(row.querySelector('.eng-sing-f1').value);
    const f2 = evalMathExpr(row.querySelector('.eng-sing-f2').value);
    return [Number.isFinite(f1) ? f1 : 0.0, Number.isFinite(f2) ? f2 : 0.0];
  } catch (_) {
    return [0.0, 0.0];
  }
}

// zeta^(alpha) is NOT a free input: it is the physically-determined site-shift
// gauge (Note A), fixed once the orbital position r_alpha and the primary
// singularity k0 are fixed. Since b_l . a_m = 2*pi*delta_lm, the Cartesian
// dot product collapses to zeta = k0 . r_alpha = 2*pi*(f1*x + f2*y) (mod 2pi),
// independent of the actual a1/a2 vectors. We mirror the engine's
// with_derived_zetas() so the read-only field shows what will be used.
function engUpdateDerivedZetas() {
  const [f1, f2] = engFirstSingularityKFrac();
  const TWO_PI = 2 * Math.PI;
  document.querySelectorAll('#eng-zetas-container .eng-sub-row').forEach(row => {
    const zin = row.querySelector('.eng-zeta-input');
    if (!zin) return;
    let px = 0.0, py = 0.0;
    try { px = evalMathExpr(row.querySelector('.eng-pos-x').value) || 0.0; } catch (_) {}
    try { py = evalMathExpr(row.querySelector('.eng-pos-y').value) || 0.0; } catch (_) {}
    let zeta = (TWO_PI * (f1 * px + f2 * py)) % TWO_PI;
    if (zeta < 0) zeta += TWO_PI;
    zin.value = zeta.toFixed(4);
  });
}

function rebuildEngSublattices() {
  const container = document.getElementById('eng-zetas-container');
  if (!container) return;
  const N = parseInt(document.getElementById('eng-sub-count').value) || 2;

  // Save current values if any
  const oldValues = {};
  container.querySelectorAll('.eng-sub-row').forEach(row => {
    oldValues[row.dataset.idx] = {
      zeta: row.querySelector('.eng-zeta-input').value,
      px: row.querySelector('.eng-pos-x').value,
      py: row.querySelector('.eng-pos-y').value,
    };
  });

  container.innerHTML = '';
  for (let i = 0; i < N; i++) {
    const label = String.fromCharCode(65 + i); // A, B, C...
    const old = oldValues[i] || { zeta: '0.0', px: '0.0', py: '0.0' };

    const row = document.createElement('div');
    row.className = 'eng-sub-row';
    row.dataset.idx = i;
    row.style.cssText = 'display: grid; grid-template-columns: 36px 1fr 1fr 1fr; gap: 6px; align-items: center;';

    const lbl = document.createElement('span');
    lbl.style.cssText = 'font-size:0.8rem; font-weight:bold; color:#475569; text-align:center;';
    lbl.textContent = label;
    row.appendChild(lbl);

    const mk = (cls, val, opts = {}) => {
      const inp = document.createElement('input');
      inp.type = 'text';
      inp.className = `lat-select ${cls}`;
      inp.dataset.idx = i;
      inp.value = val;
      inp.style.cssText = 'width: 100%; box-sizing: border-box; text-align: center; font-size: 0.78rem; padding: 3px;';
      if (opts.readonly) {
        inp.readOnly = true;
        inp.tabIndex = -1;
        inp.style.background = '#f1f5f9';
        inp.style.color = '#64748b';
        inp.title = 'ζ는 입력값이 아니라 orbital 위치와 주 특이점 k₀로부터 자동 도출됩니다:\nζ = k₀·r = 2π(f₁·x + f₂·y)  (mod 2π).\n게이지 불변이므로 Chern 수에는 영향이 없고, 최종 hopping의 상대 위상만 결정합니다.';
      } else {
        inp.addEventListener('input', () => { engBuildLatticeSVG(); engUpdateDerivedZetas(); });
      }
      row.appendChild(inp);
      return inp;
    };
    mk('eng-zeta-input', old.zeta, { readonly: true });
    mk('eng-pos-x', old.px);
    mk('eng-pos-y', old.py);

    container.appendChild(row);
  }
  engUpdateDerivedZetas();
  engBuildLatticeSVG();
}

function engGetTransform(svgEl) {
  const spec = engGetLatticeSpec();
  const orbitals = spec.sublattices.map(s => ({ label: s.label, position: s.position }));
  return latGetTransformGeneric({
    dimension: 2,
    primitiveVectors: spec.primitive_vectors,
    orbitals: orbitals,
    repX: engSt.repX, repY: engSt.repY,
    zoom: engSt.zoom, panX: engSt.panX, panY: engSt.panY,
  }, svgEl);
}

function engBuildLatticeSVG() {
  const svg = document.getElementById('eng-lat-svg');
  if (!svg) return;
  const spec = engGetLatticeSpec();
  const vecs = spec.primitive_vectors, subs = spec.sublattices;
  if (!vecs[0] || !vecs[1] || subs.length === 0) { svg.innerHTML = ''; return; }
  const { w2s } = engGetTransform(svg);
  const repX = engSt.repX, repY = engSt.repY;

  svg.innerHTML = '';
  const defs = svgE('defs');
  [['a1','#2563eb'],['a2','#dc2626']].forEach(([id,col])=>{
    const mk=svgE('marker',{id:`eng-arr-${id}`,markerWidth:'10',markerHeight:'7',refX:'8',refY:'3.5',orient:'auto'});
    mk.appendChild(svgE('polygon',{points:'0 0, 10 3.5, 0 7',fill:col}));
    defs.appendChild(mk);
  });
  svg.appendChild(defs);

  // z-order bottom -> top: cell, vectors, shapes, labels, hits. Hit-targets
  // are always topmost so manual-mode click areas are never shadowed by
  // markers/labels/cell fills from this or neighboring sites.
  const L = {};
  ['cell','vectors','shapes','labels','hits'].forEach(n=>{L[n]=svgE('g'); svg.appendChild(L[n]);});

  const cellOrigin = (n,m) => [
    n*vecs[0][0] + m*vecs[1][0],
    n*(vecs[0][1]||0) + m*(vecs[1][1]||0),
  ];

  // Unit cell outlines (central cell highlighted)
  for (let n=-repX; n<=repX; n++) for (let m=-repY; m<=repY; m++) {
    const [ox,oy] = cellOrigin(n,m);
    const pts=[[ox,oy],[ox+vecs[0][0],oy+(vecs[0][1]||0)],
      [ox+vecs[0][0]+vecs[1][0],oy+(vecs[0][1]||0)+(vecs[1][1]||0)],
      [ox+vecs[1][0],oy+(vecs[1][1]||0)]]
      .map(([wx,wy])=>w2s(wx,wy)).map(([x,y])=>`${x.toFixed(1)},${y.toFixed(1)}`).join(' ');
    const isO = n===0 && m===0;
    L.cell.appendChild(svgE('polygon',{points:pts,
      fill: isO ? 'rgba(76,122,255,0.07)' : 'none',
      stroke: isO ? '#4c7aff' : '#c7d3ee',
      'stroke-width': isO ? '1.5' : '0.7',
      'stroke-dasharray': isO ? '5,3' : '3,2'}));
  }

  // Lattice vectors a1, a2 from origin
  [{v:vecs[0],label:'\\mathbf{a}_1',col:'#2563eb',id:'a1'},
   {v:vecs[1],label:'\\mathbf{a}_2',col:'#dc2626',id:'a2'}].forEach(({v,label,col,id})=>{
    const [x0,y0]=w2s(0,0), [x1,y1]=w2s(v[0],v[1]||0);
    const dx=x1-x0, dy=y1-y0, len=Math.sqrt(dx*dx+dy*dy);
    if (len<4) return;
    const sf=(len-11)/len;
    L.vectors.appendChild(svgE('line',{x1:x0,y1:y0,x2:(x0+dx*sf).toFixed(1),y2:(y0+dy*sf).toFixed(1),
      stroke:col,'stroke-width':'2.5','stroke-linecap':'round','marker-end':`url(#eng-arr-${id})`}));
    const mx=(x0+x1)/2, my=(y0+y1)/2, nx=-dy/len, ny=dx/len;
    const side=(ny<0||(Math.abs(ny)<0.1&&nx>0))?-1:1;
    const fo=svgE('foreignObject',{x:(mx+side*nx*20-15).toFixed(1),y:(my+side*ny*20-12).toFixed(1),width:'70',height:'26'});
    const div=document.createElementNS('http://www.w3.org/1999/xhtml','div');
    div.className='lat-label lat-vec-label'; div.style.color=col;
    katex.render(label,div,{throwOnError:false});
    fo.appendChild(div); L.vectors.appendChild(fo);
  });

  // Sites: central cell + neighbor repeats.
  // Pass 1: compute every (alpha,n,m)'s screen position first, so the
  // hit/highlight radius below can be sized from the closest actual pair --
  // this guarantees neighboring sites' click-targets never overlap and
  // "cover" each other (the previous fixed r=siteR+5 could overlap at high
  // repX/repY or for closely-spaced sublattices).
  const siteR = 13;
  const sites = [];
  for (let n=-repX; n<=repX; n++) for (let m=-repY; m<=repY; m++) {
    const [ox,oy] = cellOrigin(n,m);
    subs.forEach((sub, alpha) => {
      const pos = sub.position || [0,0];
      const wx = ox + (pos[0]||0)*vecs[0][0] + (pos[1]||0)*vecs[1][0];
      const wy = oy + (pos[0]||0)*(vecs[0][1]||0) + (pos[1]||0)*(vecs[1][1]||0);
      const [sx,sy] = w2s(wx,wy);
      sites.push({alpha, n, m, sx, sy, isOrigin: n===0 && m===0});
    });
  }
  let minDist = Infinity;
  for (let i=0; i<sites.length; i++) for (let j=i+1; j<sites.length; j++) {
    const dx = sites[i].sx-sites[j].sx, dy = sites[i].sy-sites[j].sy;
    const d = Math.sqrt(dx*dx+dy*dy);
    if (d > 0.5 && d < minDist) minDist = d;
  }
  if (!isFinite(minDist)) minDist = (siteR+5)*2 + 2;
  const hitR = Math.max(6, Math.min(siteR+5, minDist/2 - 1));

  // Draw non-origin repeats first, central cell (0,0) last, so its markers
  // and labels render on top of the (rare) screen overlap with a repeat.
  const order = sites.slice().sort((a,b)=>(a.isOrigin?1:0)-(b.isOrigin?1:0));
  order.forEach(({alpha, n, m, sx, sy, isOrigin}) => {
    const sub = subs[alpha];
    const opacity = isOrigin ? 1 : 0.4;
    const color = getSublatticeColor(alpha).border;
    const shape = ENG_SHAPES[alpha % ENG_SHAPES.length];
    const cls = engSt.clsSites.find(s=>s.alpha===alpha && s.n===n && s.m===m);

    latDrawSiteShape(L.shapes, sx, sy, siteR, color, shape, opacity, false);
    if (cls) {
      L.shapes.appendChild(svgE('circle',{cx:sx,cy:sy,r:siteR+4,fill:'none',stroke:'#f59e0b','stroke-width':'2.5',opacity}));
    }

    if (isOrigin || cls) {
      const fo=svgE('foreignObject',{x:(sx+siteR+2).toFixed(1),y:(sy-10).toFixed(1),width:'110',height:'24'});
      const div=document.createElementNS('http://www.w3.org/1999/xhtml','div');
      div.className='lat-label lat-site-label';
      let txt = /^[A-Za-z0-9]$/.test(sub.label) ? sub.label : `\\text{${sub.label}}`;
      if (cls) txt += `\\,(${cls.A.toFixed(2)}\\angle${cls.theta.toFixed(2)})`;
      katex.render(txt, div, {throwOnError:false});
      fo.appendChild(div); L.labels.appendChild(fo);
    }

    if (engSt.mode === 'manual') {
      // Visible dashed "slot" doubling as the click target -- always shown
      // (not just on hover) so the whole clickable grid is visible up front,
      // and brightens on hover so the user can see exactly which site a
      // click will toggle before clicking.
      const hit = svgE('circle',{cx:sx,cy:sy,r:hitR,
        fill:'rgba(76,122,255,0.05)', stroke:'#b8c8f0',
        'stroke-width':'1', 'stroke-dasharray':'2,2'});
      hit.style.cursor = 'pointer';
      hit.addEventListener('mouseenter', () => {
        hit.setAttribute('fill','rgba(76,122,255,0.22)');
        hit.setAttribute('stroke','#4c7aff');
        hit.setAttribute('stroke-width','1.5');
      });
      hit.addEventListener('mouseleave', () => {
        hit.setAttribute('fill','rgba(76,122,255,0.05)');
        hit.setAttribute('stroke','#b8c8f0');
        hit.setAttribute('stroke-width','1');
      });
      hit.addEventListener('click', () => engToggleSite(alpha, n, m));
      L.hits.appendChild(hit);
    }
  });
}

function engToggleSite(alpha, n, m) {
  const idx = engSt.clsSites.findIndex(s => s.alpha===alpha && s.n===n && s.m===m);
  if (idx >= 0) {
    engSt.clsSites.splice(idx, 1);
  } else {
    engSt.clsSites.push({ alpha, n, m, A: 1.0, theta: 0.0 });
  }
  engBuildLatticeSVG();
  engRebuildSitesTable();
}

function engRebuildSitesTable() {
  const tbody = document.querySelector('#eng-cls-sites-table tbody');
  const empty = document.getElementById('eng-cls-sites-empty');
  if (!tbody) return;
  tbody.innerHTML = '';

  if (engSt.clsSites.length === 0) {
    if (empty) empty.style.display = 'block';
    return;
  }
  if (empty) empty.style.display = 'none';

  const spec = engGetLatticeSpec();
  engSt.clsSites.forEach(site => {
    const tr = document.createElement('tr');
    tr.style.cssText = 'border-top:1px solid #e2e8f0; text-align:center;';

    const sub = spec.sublattices[site.alpha];
    const tdLabel = document.createElement('td');
    tdLabel.style.padding = '4px';
    tdLabel.textContent = sub ? sub.label : String.fromCharCode(65+site.alpha);
    tr.appendChild(tdLabel);

    const tdCell = document.createElement('td');
    tdCell.style.padding = '4px';
    tdCell.textContent = `(${site.n}, ${site.m})`;
    tr.appendChild(tdCell);

    const mkNumInput = (val, onChange) => {
      const td = document.createElement('td');
      td.style.padding = '4px';
      const inp = document.createElement('input');
      inp.type = 'number'; inp.step = '0.1'; inp.value = val;
      inp.className = 'lat-select';
      inp.style.cssText = 'width:60px; text-align:center; font-size:0.76rem; padding:2px;';
      inp.addEventListener('input', () => { onChange(parseFloat(inp.value) || 0.0); engBuildLatticeSVG(); });
      td.appendChild(inp);
      tr.appendChild(td);
    };
    mkNumInput(site.A, v => site.A = v);
    mkNumInput(site.theta, v => site.theta = v);

    const tdDel = document.createElement('td');
    tdDel.style.padding = '4px';
    const delBtn = document.createElement('button');
    delBtn.className = 'btn-remove';
    delBtn.textContent = '×';
    delBtn.style.cssText = 'width: 22px; height: 22px; line-height: 18px; padding: 0; font-size: 14px;';
    delBtn.onclick = () => engToggleSite(site.alpha, site.n, site.m);
    tdDel.appendChild(delBtn);
    tr.appendChild(tdDel);

    tbody.appendChild(tr);
  });
}

function engSwitchMode(mode) {
  engSt.mode = mode;
  const autoPanel = document.getElementById('eng-auto-panel');
  const manualPanel = document.getElementById('eng-manual-panel');
  if (autoPanel) autoPanel.classList.toggle('hidden', mode !== 'auto');
  if (manualPanel) manualPanel.classList.toggle('hidden', mode !== 'manual');
  const candidatesPanel = document.getElementById('eng-candidates-panel');
  if (candidatesPanel) candidatesPanel.classList.add('hidden');
  const metricsPanel = document.getElementById('eng-metrics-panel');
  if (metricsPanel) metricsPanel.style.display = 'none';
  engBuildLatticeSVG();
}

function addEngSingularityRow(name = '', f1 = '0.0', f2 = '0.0', w = '1') {
  const container = document.getElementById('eng-singularities-list');
  if (!container) return;
  
  const row = document.createElement('div');
  row.className = 'eng-sing-row';
  row.style.cssText = 'display: flex; gap: 6px; align-items: center; background: #fff; border: 1.5px solid #cbd5e1; padding: 6px; border-radius: 6px; margin-top: 4px; box-sizing: border-box;';
  
  // Selection box for named presets
  const sel = document.createElement('select');
  sel.className = 'lat-select eng-sing-preset';
  sel.style.cssText = 'width: 90px; padding: 2px; font-size: 0.8rem;';
  
  const presets = [
    { value: 'Gamma', text: 'Γ (0, 0)', f1: '0.0', f2: '0.0' },
    { value: 'M', text: 'M (0.5, 0.5)', f1: '0.5', f2: '0.5' },
    { value: 'X', text: 'X (0.5, 0)', f1: '0.5', f2: '0.0' },
    { value: 'Y', text: 'Y (0, 0.5)', f1: '0.0', f2: '0.5' },
    { value: 'K', text: 'K (1/3, 1/3)', f1: '0.33333333', f2: '0.33333333' },
    { value: 'custom', text: '직접 입력', f1: '', f2: '' }
  ];
  
  presets.forEach(p => {
    const opt = document.createElement('option');
    opt.value = p.value;
    opt.textContent = p.text;
    sel.appendChild(opt);
  });
  
  // Inputs for coordinates
  const f1Inp = document.createElement('input');
  f1Inp.type = 'text';
  f1Inp.className = 'lat-select eng-sing-f1';
  f1Inp.value = f1;
  f1Inp.style.cssText = 'width: 60px; text-align: center; padding: 2px; font-size: 0.8rem;';
  
  const f2Inp = document.createElement('input');
  f2Inp.type = 'text';
  f2Inp.className = 'lat-select eng-sing-f2';
  f2Inp.value = f2;
  f2Inp.style.cssText = 'width: 60px; text-align: center; padding: 2px; font-size: 0.8rem;';
  
  // Auto update coordinates when selecting a preset
  sel.addEventListener('change', () => {
    const p = presets.find(x => x.value === sel.value);
    if (p && p.value !== 'custom') {
      f1Inp.value = p.f1;
      f2Inp.value = p.f2;
      f1Inp.disabled = true;
      f2Inp.disabled = true;
    } else {
      f1Inp.disabled = false;
      f2Inp.disabled = false;
    }
  });
  
  // Set default selection based on values
  let matchedPreset = 'custom';
  if (name) {
    const p = presets.find(x => x.value.toLowerCase() === name.toLowerCase());
    if (p) matchedPreset = p.value;
  } else {
    // try to match coordinates
    const p = presets.find(x => Math.abs(parseFloat(x.f1) - parseFloat(f1)) < 1e-4 && Math.abs(parseFloat(x.f2) - parseFloat(f2)) < 1e-4);
    if (p) matchedPreset = p.value;
  }
  sel.value = matchedPreset;
  if (matchedPreset !== 'custom') {
    f1Inp.disabled = true;
    f2Inp.disabled = true;
  }
  
  // Winding selection
  const wSel = document.createElement('select');
  wSel.className = 'lat-select eng-sing-w';
  wSel.style.cssText = 'width: 55px; padding: 2px; font-size: 0.8rem;';
  
  const optPos = document.createElement('option');
  optPos.value = '1';
  optPos.textContent = '+1';
  if (w === '1' || w === 1) optPos.selected = true;
  
  const optNeg = document.createElement('option');
  optNeg.value = '-1';
  optNeg.textContent = '-1';
  if (w === '-1' || w === -1) optNeg.selected = true;
  
  wSel.appendChild(optPos);
  wSel.appendChild(optNeg);
  
  // Delete button
  const delBtn = document.createElement('button');
  delBtn.className = 'btn-remove';
  delBtn.textContent = '×';
  delBtn.style.cssText = 'margin-left: auto; width: 22px; height: 22px; line-height: 18px; padding: 0; font-size: 14px;';
  delBtn.onclick = () => {
    row.remove();
  };
  
  row.appendChild(sel);
  row.appendChild(document.createTextNode(' frac: '));
  row.appendChild(f1Inp);
  row.appendChild(f2Inp);
  row.appendChild(document.createTextNode(' w: '));
  row.appendChild(wSel);
  row.appendChild(delBtn);
  
  container.appendChild(row);
}

async function runFlatBandDesign() {
  const loading = document.getElementById('eng-loading');
  const logOutput = document.getElementById('eng-log-output');
  const metricsPanel = document.getElementById('eng-metrics-panel');
  const candidatesPanel = document.getElementById('eng-candidates-panel');
  if (candidatesPanel) candidatesPanel.classList.add('hidden');
  engSt.selectedCandidate = -1;
  if (loading) loading.classList.remove('hidden');

  try {
    // 1. Lattice vectors + sublattices (label, gauge zeta, fractional position)
    const lattice_spec = engGetLatticeSpec();

    // 3. DesignTarget
    const target = engGatherTarget();

    // 4. Advanced parameters
    const payload = {
      lattice_spec: lattice_spec,
      target: target,
      E0: parseFloat(document.getElementById('eng-e0').value) || 0.0,
      t: parseFloat(document.getElementById('eng-disp-t').value) || 0.3,
      delta: parseFloat(document.getElementById('eng-disp-delta').value) || 0.5,
      n_grid_ift: parseInt(document.getElementById('eng-ift-grid').value) || 24,
      max_retries: parseInt(document.getElementById('eng-max-retries').value) || 8,
      r_max: engGetRMax(),
      cls_size: engGetClsSize(),
    };

    logOutput.textContent = '설계 실행 중... API 호출 중입니다.\n';
    
    const res = await apiFetch('design_flat_band', payload);
    
    if (!res.success) {
      logOutput.textContent += `[오류] 설계 실패:\n${res.error}\n\n${res.traceback || ''}`;
      if (metricsPanel) metricsPanel.style.display = 'none';
      return;
    }
    
    // Success!
    engLastResult = res;

    // Render logs
    logOutput.textContent = res.log.join('\n');
    logOutput.scrollTop = logOutput.scrollHeight; // Scroll to bottom

    renderEngMetrics(res, target);

  } catch (err) {
    logOutput.textContent += `[오류] ${err.message}`;
  } finally {
    if (loading) loading.classList.add('hidden');
  }
}

// Max hopping range cap for the exact local model: integer, or null = no cap
// (= the natural, exactly-flat range determined by the CLS geometry).
function engGetRMax() {
  const v = document.getElementById('eng-rmax')?.value;
  if (v === undefined || v === null || v === '') return null;
  const n = parseInt(v);
  return Number.isFinite(n) ? n : null;
}

// Target CLS size (spatial extent): positive integer, or null = auto/minimal.
function engGetClsSize() {
  const v = document.getElementById('eng-cls-size')?.value;
  if (v === undefined || v === null || v === '') return null;
  const n = parseInt(v);
  return Number.isFinite(n) ? n : null;
}

function engGatherTarget() {
  const targetC = parseInt(document.getElementById('eng-target-c').value);
  const singularities = [];
  let sumW = 0;

  const singRows = document.querySelectorAll('#eng-singularities-list .eng-sing-row');
  if (singRows.length === 0) {
    throw new Error('최소 하나의 특이점(Singularity)을 지정해야 합니다.');
  }

  singRows.forEach((row, idx) => {
    const presetSel = row.querySelector('.eng-sing-preset');
    const f1 = evalMathExpr(row.querySelector('.eng-sing-f1').value);
    const f2 = evalMathExpr(row.querySelector('.eng-sing-f2').value);
    const w = parseInt(row.querySelector('.eng-sing-w').value);

    let name = presetSel.value;
    if (name === 'custom') name = `k_${idx+1}`;

    singularities.push({ name, k_frac: [f1, f2], w });
    sumW += w;
  });

  if (sumW !== targetC) {
    throw new Error(`특이점 w의 합은 목표 Chern 수 C와 일치해야 합니다. (w의 합: ${sumW}, 목표 C: ${targetC})`);
  }

  return { C: targetC, singularities };
}

function renderEngMetrics(result, target) {
  const metricsPanel = document.getElementById('eng-metrics-panel');
  const targetBlock = document.getElementById('eng-target-metrics');
  const manualBlock = document.getElementById('eng-manual-report');
  const badge = document.getElementById('eng-success-badge');
  if (!metricsPanel) return;
  metricsPanel.style.display = 'block';

  if (target) {
    // Auto-design / explorer-candidate path: target-driven verification
    if (targetBlock) targetBlock.classList.remove('hidden');
    if (manualBlock) manualBlock.classList.add('hidden');

    const v = result.verification;
    document.getElementById('eng-val-target-c').textContent = v.target_C;
    document.getElementById('eng-val-analytic-c').textContent = v.analytic_C;
    document.getElementById('eng-val-full-c').textContent = v.full_numerical_C;
    document.getElementById('eng-val-trunc-c').textContent = v.trunc_numerical_C;

    // Hopping range: natural (exactly-flat) range, plus the applied cap if any.
    const rangeEl = document.getElementById('eng-val-range');
    if (rangeEl) {
      const nat = v.natural_hopping_range;
      const order = (r) => (r === 1 ? 'NN(최근접)' : r === 2 ? 'NNN(차근접)' : r === 3 ? '3rd' : `${r}th`);
      let s = `${v.max_hopping_range}셀 (${order(v.max_hopping_range)}), ${v.n_hopping_terms}개 항`;
      if (v.r_max_applied) {
        s += ` — R_max=${v.r_max}로 제한 (자연 범위 ${nat}셀, ${(v.truncation_ratio * 100).toFixed(2)}% 손실 → 근사)`;
      } else if (nat !== undefined) {
        s += ` — CLS 기하로 자동 결정된 정확 범위`;
      }
      rangeEl.textContent = s;
    }

    // CLS information: requested size (cls_size) and actual extent & site count
    const clsExtentEl = document.getElementById('eng-val-cls-extent');
    if (clsExtentEl) {
      clsExtentEl.textContent = v.cls_size !== undefined && v.cls_size !== null ? `${v.cls_size} (실제 ${v.cls_extent}셀)` : `자동 (실제 ${v.cls_extent}셀)`;
    }
    const clsSitesEl = document.getElementById('eng-val-cls-sites');
    if (clsSitesEl) {
      clsSitesEl.textContent = v.n_cls_sites !== undefined ? `${v.n_cls_sites}개` : '-';
    }
    // Flatness: exact (~0) or the truncation-induced deviation.
    const flatEl = document.getElementById('eng-val-flatness');
    if (flatEl) {
      const fd = v.flat_band_max_dev;
      if (typeof fd === 'number' && fd >= 0) {
        flatEl.textContent = v.exact_flat
          ? `${fd.toExponential(2)} (정확 평탄 — 재로드해도 유지됨)`
          : `${fd.toExponential(2)} (범위 제한으로 평탄성 깨짐)`;
        flatEl.style.color = v.exact_flat ? '#16a34a' : '#d97706';
      } else {
        flatEl.textContent = '-';
      }
    }

    let contText = '';
    for (const [name, info] of Object.entries(v.continuity)) {
      const ok = info.loop.projector_continuous && info.loop.winding === target.singularities.find(s=>s.name === name)?.w;
      contText += `${name}: ${ok ? '연속' : '불연속/실패'} (winding: ${info.loop.winding}, rank ratio: ${info.loop.rank_ratio.toExponential(2)}) `;
    }
    // The flat band of a nonzero-Chern CLS construction TOUCHES the dispersive
    // sector at its singularity (Note A Sec.8) -- report this as the expected,
    // correct outcome, not a defect.
    if (v.trunc_singular) {
      contText += '| 평탄밴드는 특이점에서 분산밴드와 닿아 있음 (비고립) — 비자명 위상에서 정상/의도된 동작.';
    } else {
      contText += '| 절단 밴드 고립됨 (FHS 수치 Chern으로 추가 확인됨).';
    }
    if (typeof v.flat_band_max_dev === 'number' && v.flat_band_max_dev >= 0) {
      contText += ` 평탄도 max‖Hψ−E₀ψ‖=${v.flat_band_max_dev.toExponential(2)}.`;
    }
    document.getElementById('eng-val-continuity').textContent = contText;

    if (badge) {
      // Success is the realised topology (analytic ground truth) + exact
      // flatness. A range-capped (approximate) model is its own status, not a
      // failure. A touching band is EXPECTED, not a defect.
      if (v.analytic_match === false) {
        badge.textContent = '설계 불완전 (위상 불일치)';
        badge.className = 'status-badge warning';
        badge.style.backgroundColor = '#f59e0b';
      } else if (v.r_max_applied && !v.exact_flat) {
        badge.textContent = `근사 모델 (R_max=${v.r_max} 제한)`;
        badge.className = 'status-badge warning';
        badge.style.backgroundColor = '#d97706';
      } else if (v.phase5_success) {
        badge.textContent = v.trunc_singular ? '설계 성공 (특이 평탄밴드)' : '설계 성공';
        badge.className = 'status-badge active';
        badge.style.backgroundColor = '#10b981';
      } else {
        badge.textContent = '설계 불완전';
        badge.className = 'status-badge warning';
        badge.style.backgroundColor = '#f59e0b';
      }
    }
  } else {
    // Manual-mode path: no target -- report the auto-discovered topology
    if (targetBlock) targetBlock.classList.add('hidden');
    if (manualBlock) manualBlock.classList.remove('hidden');

    const cr = result.chern_report || {};
    document.getElementById('eng-val-manual-c').textContent =
      cr.chern_number !== undefined ? cr.chern_number : '-';
    document.getElementById('eng-val-manual-welldef').textContent =
      cr.well_defined === undefined ? '-' : (cr.well_defined ? '예' : '아니오');
    document.getElementById('eng-val-manual-zeros').textContent =
      result.zeros ? result.zeros.length : '0';

    const v = result.verification || {};
    document.getElementById('eng-val-manual-trunc-ratio').textContent =
      (v.truncation_ratio !== undefined && v.truncation_ratio !== null) ? `${(v.truncation_ratio * 100).toFixed(4)}%` : '-';
    document.getElementById('eng-val-manual-rcut').textContent =
      v.R_cut !== undefined ? v.R_cut : '-';
    document.getElementById('eng-val-manual-trunc').textContent =
      v.trunc_isolated === undefined ? '-' : (v.trunc_isolated ? '예' : '아니오');

    const warnBox = document.getElementById('eng-manual-warnings');
    if (warnBox) {
      warnBox.innerHTML = '';
      const warnings = [...(result.warnings || [])];
      if (result.trivial) {
        warnings.unshift('이 CLS는 모든 사이트가 (0,0) 셀에 있어 f(k)가 k에 무관합니다 (자명한 분자궤도 평탄밴드, C=0). 이웃 셀에도 사이트를 배치하면 비자명한 위상을 얻을 수 있습니다.');
      }
      warnings.forEach(w => {
        const d = document.createElement('div');
        d.style.cssText = 'background:#fffbeb; border:1px solid #fde68a; color:#92400e; padding:6px 8px; border-radius:4px; font-size:0.78rem;';
        d.textContent = `⚠ ${w}`;
        warnBox.appendChild(d);
      });
    }

    const summaryEl = document.getElementById('eng-manual-summary');
    if (summaryEl) summaryEl.textContent = cr.summary || '';

    if (badge) {
      if (cr.well_defined) {
        badge.textContent = '밴드 고립됨';
        badge.className = 'status-badge active';
        badge.style.backgroundColor = '#10b981';
      } else {
        badge.textContent = '밴드 비고립/특이';
        badge.className = 'status-badge warning';
        badge.style.backgroundColor = '#f59e0b';
      }
    }
  }
}

async function engRunManualAnalysis() {
  const loading = document.getElementById('eng-loading');
  const logOutput = document.getElementById('eng-log-output');
  const metricsPanel = document.getElementById('eng-metrics-panel');
  const candidatesPanel = document.getElementById('eng-candidates-panel');
  if (candidatesPanel) candidatesPanel.classList.add('hidden');
  if (loading) loading.classList.remove('hidden');

  try {
    if (engSt.clsSites.length === 0) {
      throw new Error('최소 하나의 CLS 사이트를 배치해야 합니다. 위 격자 미리보기에서 사이트를 클릭하세요.');
    }

    const lattice_spec = engGetLatticeSpec();
    const cls_sites = engSt.clsSites.map(s => ({ alpha: s.alpha, n: s.n, m: s.m, A: s.A, theta: s.theta }));

    const payload = {
      lattice_spec,
      cls_sites,
      E0: parseFloat(document.getElementById('eng-e0').value) || 0.0,
      t: parseFloat(document.getElementById('eng-disp-t').value) || 0.3,
      delta: parseFloat(document.getElementById('eng-disp-delta').value) || 0.5,
      n_grid_ift: parseInt(document.getElementById('eng-ift-grid').value) || 24,
      R_cut: engGetRMax() || 3,
      max_rcut_retries: parseInt(document.getElementById('eng-max-retries').value) || 4,
    };

    logOutput.textContent = 'CLS 분석 실행 중... API 호출 중입니다.\n';

    const res = await apiFetch('analyze_manual_cls', payload);

    if (!res.success) {
      logOutput.textContent += `[오류] 분석 실패:\n${res.error}\n\n${res.traceback || ''}`;
      if (metricsPanel) metricsPanel.style.display = 'none';
      return;
    }

    if (!res.valid) {
      logOutput.textContent = (res.log || []).join('\n') +
        `\n[중단] ${res.reason || '이 CLS 배치는 유효한 f(k)를 만들지 못합니다 (f(k) ≡ 0).'}`;
      if (metricsPanel) metricsPanel.style.display = 'none';
      return;
    }

    engLastResult = res;
    logOutput.textContent = (res.log || []).join('\n');
    logOutput.scrollTop = logOutput.scrollHeight;

    renderEngMetrics(res, null);

  } catch (err) {
    logOutput.textContent += `[오류] ${err.message}`;
  } finally {
    if (loading) loading.classList.add('hidden');
  }
}

async function runFlatBandExplore() {
  const loading = document.getElementById('eng-loading');
  const logOutput = document.getElementById('eng-log-output');
  const metricsPanel = document.getElementById('eng-metrics-panel');
  const candidatesPanel = document.getElementById('eng-candidates-panel');
  const progressEl = document.getElementById('eng-explore-progress');
  const tbody = document.querySelector('#eng-candidates-table tbody');
  if (loading) loading.classList.remove('hidden');
  if (metricsPanel) metricsPanel.style.display = 'none';

  try {
    const lattice_spec = engGetLatticeSpec();
    const target = engGatherTarget();

    const offsets = Array.from(document.querySelectorAll('.eng-explore-offset:checked')).map(cb => parseInt(cb.value));
    if (offsets.length === 0) {
      throw new Error('최소 하나의 shell offset을 선택해야 합니다.');
    }

    const mk_variants = document.getElementById('eng-explore-mk').value
      .split('\n').map(l => l.trim()).filter(l => l)
      .map(l => l.split(',').map(x => parseFloat(x.trim())))
      .filter(pair => pair.length === 2 && !isNaN(pair[0]) && !isNaN(pair[1]));
    if (mk_variants.length === 0) {
      throw new Error('최소 하나의 (t, δ) 후보가 필요합니다.');
    }

    // R_max candidates: each value caps the hopping range; 0 (or empty) = no
    // cap (exact). Empty input -> a single exact candidate.
    let rcut_variants = document.getElementById('eng-explore-rcuts').value
      .split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x) && x > 0);
    if (rcut_variants.length === 0) rcut_variants = [0];   // 0 = exact (no cap)

    // CLS sizes: positive integers (1, 2, 3...) indicating requested CLS radius.
    // 0 or empty/NaN translates to None (automatic/minimal shells).
    let cls_sizes = document.getElementById('eng-explore-cls-sizes').value
      .split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x) && x >= 0);
    if (cls_sizes.length === 0) cls_sizes = [0]; // 0 = auto/default

    const payload = {
      lattice_spec,
      target,
      E0: parseFloat(document.getElementById('eng-e0').value) || 0.0,
      mk_variants,
      offsets,
      rcut_variants,
      cls_sizes,
      n_grid_ift: parseInt(document.getElementById('eng-ift-grid').value) || 24,
      max_retries: parseInt(document.getElementById('eng-explore-max-retries').value) || 2,
      max_candidates: parseInt(document.getElementById('eng-explore-max').value) || 24,
    };

    engSt.candidates = [];
    engSt.selectedCandidate = -1;
    engSt.exploreTarget = target;
    engSt.explorePrimVecs = null;
    engSt.exploreOrbitals = null;
    engSt.exploreRanked = [];
    if (tbody) tbody.innerHTML = '';
    if (candidatesPanel) candidatesPanel.classList.remove('hidden');
    if (progressEl) progressEl.textContent = '탐색을 시작합니다...';
    logOutput.textContent = '다중 후보 탐색 실행 중...\n';

    const response = await fetch('/api/design_flat_band_explore_stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const txt = await response.text().catch(() => response.statusText);
      throw new Error(`API ${response.status}: ${txt}`);
    }

    const handleLine = (line) => {
      if (!line.trim()) return;
      let data;
      try { data = JSON.parse(line); }
      catch (e) { console.error('Failed to parse line:', line, e); return; }

      if (data.type === 'progress') {
        const cand = data.candidate;
        engSt.candidates[cand.index] = cand;
        addEngCandidateRow(cand);
        if (progressEl) progressEl.textContent = `진행: ${data.index + 1} / ${data.total}`;
        const status = cand.error ? `오류: ${cand.error}` : `score=${cand.score.toFixed(2)}`;
        logOutput.textContent += `[${data.index+1}/${data.total}] offset=${cand.offset} t=${cand.t} δ=${cand.delta} R_cut0=${cand.R_cut0} -> ${status}\n`;
        logOutput.scrollTop = logOutput.scrollHeight;
      } else if (data.type === 'done') {
        engSt.explorePrimVecs = data.primitive_vectors;
        engSt.exploreOrbitals = data.orbitals;
        engSt.exploreRanked = data.ranked;
        if (progressEl) progressEl.textContent = `탐색 완료: ${data.count}개 시도 중 ${data.ranked.length}개 고유 후보 (점수순)`;
        markRankedCandidates(data.ranked);
        if (data.ranked.length > 0) engSelectCandidate(data.ranked[0]);
      } else if (data.type === 'error') {
        logOutput.textContent += `[오류] ${data.error}\n${data.traceback || ''}`;
      }
    };

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) handleLine(line);
    }
    if (buffer.trim()) handleLine(buffer);

  } catch (err) {
    logOutput.textContent += `[오류] ${err.message}`;
  } finally {
    if (loading) loading.classList.add('hidden');
  }
}

function addEngCandidateRow(cand) {
  const tbody = document.querySelector('#eng-candidates-table tbody');
  if (!tbody) return;
  let tr = tbody.querySelector(`tr[data-cand-idx="${cand.index}"]`);
  if (!tr) {
    tr = document.createElement('tr');
    tr.dataset.candIdx = cand.index;
    tr.style.cssText = 'border-top:1px solid #e2e8f0; text-align:center;';
    tbody.appendChild(tr);
  }
  tr.innerHTML = '';

  const v = cand.verification;
  const fdev = (v && typeof v.flat_band_max_dev === 'number' && v.flat_band_max_dev >= 0)
    ? (v.flat_band_max_dev < 1e-9 ? '0 (정확)' : v.flat_band_max_dev.toExponential(1)) : '-';
  const status = cand.error ? '오류'
    : (!v ? '-' : (v.analytic_match === false ? '불완전'
      : (v.exact_flat ? '정확' : '근사')));
  const cls_info = v
    ? `${v.cls_size ?? '자동'} (${v.cls_extent}셀, ${v.n_cls_sites}점)`
    : (cand.cls_size ?? '자동');
  const cells = [
    cand.index + 1,
    cand.offset,
    (!cand.R_cut0 ? '정확' : cand.R_cut0),
    cls_info,
    v ? (v.max_hopping_range ?? '-') : '-',
    v ? (v.n_hopping_terms ?? '-') : '-',
    fdev,
    v ? v.trunc_numerical_C : '-',
    cand.error ? '-' : cand.score.toFixed(2),
    status,
  ];
  cells.forEach(c => {
    const td = document.createElement('td');
    td.style.padding = '4px';
    td.textContent = c;
    tr.appendChild(td);
  });

  const tdBtn = document.createElement('td');
  tdBtn.style.padding = '4px';
  if (!cand.error && v) {
    const btn = document.createElement('button');
    btn.className = 'btn btn-ghost btn-sm';
    btn.style.cssText = 'padding:2px 8px; font-size:0.74rem;';
    btn.textContent = '선택';
    btn.onclick = () => engSelectCandidate(cand.index);
    tdBtn.appendChild(btn);
  }
  tr.appendChild(tdBtn);

  if (cand.error) {
    tr.title = cand.error;
    tr.style.color = '#94a3b8';
  } else {
    tr.style.cursor = 'pointer';
    tr.onclick = (ev) => { if (ev.target.tagName !== 'BUTTON') engSelectCandidate(cand.index); };
  }

  if (engSt.selectedCandidate === cand.index) {
    tr.style.background = '#eff6ff';
  }
}

function markRankedCandidates(ranked) {
  const tbody = document.querySelector('#eng-candidates-table tbody');
  if (!tbody) return;
  ranked.forEach((idx, rank) => {
    const tr = tbody.querySelector(`tr[data-cand-idx="${idx}"]`);
    if (!tr) return;
    if (rank === 0) {
      tr.style.fontWeight = 'bold';
      const firstTd = tr.querySelector('td');
      if (firstTd) firstTd.textContent = `★ ${idx+1}`;
    }
  });
}

function engSelectCandidate(idx) {
  const cand = engSt.candidates[idx];
  if (!cand || cand.error || !cand.verification) return;

  engSt.selectedCandidate = idx;

  engLastResult = {
    primitive_vectors: engSt.explorePrimVecs,
    orbitals: engSt.exploreOrbitals,
    hoppings: cand.hoppings,
    verification: cand.verification,
    x_k: cand.x_k,
    log: [`후보 #${idx+1}: offset=${cand.offset}, t=${cand.t}, delta=${cand.delta}, R_cut0=${cand.R_cut0}, score=${cand.score.toFixed(2)}`],
  };

  renderEngMetrics(engLastResult, engSt.exploreTarget);

  const logOutput = document.getElementById('eng-log-output');
  if (logOutput) {
    logOutput.textContent += `\n--- 후보 #${idx+1} 선택됨 ---\n`;
    logOutput.scrollTop = logOutput.scrollHeight;
  }

  document.querySelectorAll('#eng-candidates-table tbody tr').forEach(tr => {
    tr.style.background = (parseInt(tr.dataset.candIdx) === idx) ? '#eff6ff' : '';
  });
}

function loadDesignedModel() {
  if (!engLastResult) {
    alert('불러올 설계 결과가 없습니다.');
    return;
  }
  
  if (!confirm('현재 워크스페이스의 격자 및 해밀토니안 설정을 이 설계된 모델로 덮어쓰시겠습니까?')) {
    return;
  }
  
  // Format the spec object to match what applySpecToState expects
  const spec = {
    lattice: {
      dimension: 2,
      primitive_vectors: engLastResult.primitive_vectors,
      orbitals: engLastResult.orbitals
    },
    hoppings: engLastResult.hoppings,
    options: {
      k_grid: [40, 40],
      flat_tol: 1e-4,
      k_path_str: 'Γ - X - M - Γ',
      plot_n: 60
    }
  };
  
  applySpecToState(spec);
  rebuildLatticeUI();
  rebuildHamiltonianEditor();

  // Auto scroll to left panel or switch to bands tab to preview
  showPanel('bands');

  alert('설계된 2D Flat Band 모델이 워크스페이스에 로드되었습니다!\n해밀토니안 행렬 H(k)을 확인한 뒤 [밴드 미리보기] 또는 [CLS 전체 분석]을 실행해 보세요.');

  // 평탄한 호핑 목록 대신, 어떤 항이 행렬의 어느 위치에 들어가는지 한눈에
  // 보이도록 H(k) 행렬 미리보기를 즉시 큰 모달로 표시한다.
  openHamiltonianMatrixModal();
}


async function runGlobalTopologyAnalysis() {
  const spec = buildSpec();
  const loading = document.getElementById('topo-loading');
  if (loading) loading.classList.remove('hidden');

  // Parse occupied bands
  const bandsStr = document.getElementById('topo-bands-input').value.trim();
  let bandIndices = null;
  if (bandsStr) {
    bandIndices = bandsStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
  }

  const Nx = parseInt(document.getElementById('topo-nx').value) || 40;
  const Ny = parseInt(document.getElementById('topo-ny').value) || 40;
  const cylNx = parseInt(document.getElementById('topo-cyl-nx').value) || 40;

  try {
    // 1. Run Wilson Loop / WCC Flow
    const wccRes = await fetch('/api/wilson_loop', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        spec: spec,
        band_indices: bandIndices,
        n_x: Nx,
        n_y: Ny
      })
    });
    const wccData = await wccRes.json();
    if (wccData.error) {
      alert(`Wilson Loop 에러: ${wccData.error}`);
      if (loading) loading.classList.add('hidden');
      return;
    }

    plotWCCFlow(wccData);

    // 2. Run Cylinder Entanglement Spectrum
    const esRes = await fetch('/api/entanglement_spectrum', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        spec: spec,
        band_indices: bandIndices,
        N_x: cylNx,
        n_y: 60
      })
    });
    const esData = await esRes.json();
    if (esData.error) {
      alert(`Entanglement Spectrum 에러: ${esData.error}`);
      if (loading) loading.classList.add('hidden');
      return;
    }

    plotEntanglementSpectrum(esData);

    // 3. Run Fu-Kane Symmetry Indicators
    await runFuKaneAnalysis(bandIndices);

  } catch (err) {
    alert(`계산 중 오류가 발생했습니다: ${err}`);
  } finally {
    if (loading) loading.classList.add('hidden');
  }
}

function plotWCCFlow(data) {
  const container = document.getElementById('topo-wcc-plot');
  if (!container) return;

  const traces = [];
  const k_y = data.k_y;
  const tracks = data.tracks;

  tracks.forEach((track, tid) => {
    // Add null values at wrapping points to prevent draw lines crossing vertically
    const x_plot = [];
    const y_plot = [];
    for (let i = 0; i < k_y.length; i++) {
      x_plot.push(k_y[i]);
      y_plot.push(track[i]);
      if (i < k_y.length - 1 && Math.abs(track[i+1] - track[i]) > 0.45) {
        x_plot.push(null);
        y_plot.push(null);
      }
    }

    traces.push({
      x: x_plot,
      y: y_plot,
      mode: 'lines+markers',
      name: `WCC Track ${tid + 1}`,
      line: { width: 2 },
      marker: { size: 4 }
    });
  });

  const layout = {
    title: { text: 'Wannier Charge Center (WCC) Flow vs k_y', font: { size: 14 } },
    xaxis: { title: 'k_y' },
    yaxis: { title: 'WCC ν (mod 1)', range: [-0.51, 0.51] },
    margin: { l: 50, r: 20, t: 40, b: 40 },
    hovermode: 'closest',
    plot_bgcolor: '#f8fafc',
    paper_bgcolor: 'white'
  };

  Plotly.newPlot(container, traces, layout, { responsive: true });

  // Update text result
  const resDiv = document.getElementById('topo-wcc-result');
  if (resDiv) {
    let html = `점유 밴드: [${data.band_indices.join(', ')}] | `;
    html += `FHS Chern 수 합계: <strong>C = ${data.chern}</strong> | `;
    html += `Z2 Invariant: <strong>Z2 = ${data.z2}</strong>`;
    resDiv.innerHTML = html;
  }
}

function plotEntanglementSpectrum(data) {
  const container = document.getElementById('topo-es-plot');
  if (!container) return;

  const traces = [];
  const k_y = data.k_y;
  const spectrum = data.spectrum;
  const dim_A = spectrum[0].length;

  for (let m = 0; m < dim_A; m++) {
    const x_plot = [];
    const y_plot = [];
    for (let j = 0; j < k_y.length; j++) {
      x_plot.push(k_y[j]);
      y_plot.push(spectrum[j][m]);
    }

    traces.push({
      x: x_plot,
      y: y_plot,
      mode: 'lines',
      line: { color: 'rgba(99, 102, 241, 0.6)', width: 1.5 },
      showlegend: false
    });
  }

  const layout = {
    title: { text: 'Cylinder Entanglement Spectrum ε_m(k_y)', font: { size: 14 } },
    xaxis: { title: 'k_y' },
    yaxis: { title: 'Entanglement Energy ε' },
    margin: { l: 50, r: 20, t: 40, b: 40 },
    hovermode: 'closest',
    plot_bgcolor: '#f8fafc',
    paper_bgcolor: 'white'
  };

  Plotly.newPlot(container, traces, layout, { responsive: true });

  // Update text result
  const resDiv = document.getElementById('topo-es-result');
  if (resDiv) {
    resDiv.innerHTML = `Region A 차원: ${dim_A} | ε = 0 부근의 Edge Mode 존재 여부를 차트에서 확인하세요.`;
  }
}

async function runFuKaneAnalysis(bandIndices = null) {
  const spec = buildSpec();
  const resDiv = document.getElementById('topo-fukane-result');
  if (!resDiv) return;

  if (!bandIndices) {
    const bandsStr = document.getElementById('topo-bands-input').value.trim();
    if (bandsStr) {
      bandIndices = bandsStr.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n));
    }
  }

  // Parse parity matrix diagonals
  const parityStr = document.getElementById('topo-parity-input').value.trim();
  let parityList = null;
  if (parityStr) {
    parityList = parityStr.split(',').map(s => parseFloat(s.trim())).filter(n => !isNaN(n));
  }

  try {
    const res = await fetch('/api/fu_kane', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        spec: spec,
        band_indices: bandIndices,
        P_matrix_list: parityList
      })
    });
    const data = await res.json();
    
    if (data.error) {
      resDiv.innerHTML = `<span style="color: #b91c1c;">⚠️ Fu-Kane 에러: ${data.error}</span>`;
      return;
    }

    let html = '';
    if (!data.symmetric) {
      html += `<div style="color: #b91c1c; font-weight: bold; margin-bottom: 8px;">⚠️ 공간 반전 대칭 깨짐!</div>`;
      html += `TRIM 지점에서 해밀토니안과 Parity 연산자가 교환하지 않습니다 ([H(k), P] ≠ 0).<br>`;
      html += `교환 오류 norms: ` + Object.entries(data.comm_errors).map(([k, v]) => `${k}:${v.toFixed(4)}`).join(', ') + `<br>`;
      html += `<strong>안내:</strong> 대칭성이 깨져 Fu-Kane 공식을 신뢰할 수 없습니다. 위의 Wilson Loop 결과를 확인해 주세요.`;
      resDiv.style.background = '#fef2f2';
      resDiv.style.borderColor = '#fca5a5';
      resDiv.style.color = '#991b1b';
    } else {
      html += `<div style="font-weight: bold; color: #1e1b4b; margin-bottom: 8px;">✅ 공간 반전 대칭성 검증 통과 ([H(k), P] = 0)</div>`;
      html += `<table style="width: 100%; border-collapse: collapse; margin-bottom: 10px; font-size: 0.78rem;">`;
      html += `<tr style="border-bottom: 1px solid #ddd; font-weight: bold;"><th style="text-align: left; padding: 4px;">TRIM 지점</th><th style="text-align: left; padding: 4px;">Parity 고유값 목록</th><th style="text-align: left; padding: 4px;">곱 (δ_i)</th></tr>`;
      data.parity_details.forEach(det => {
        html += `<tr style="border-bottom: 1px solid #eee;">`;
        html += `<td style="padding: 4px; font-weight: bold;">${det.point}</td>`;
        html += `<td style="padding: 4px; font-family: monospace;">[${det.parity_vals.join(', ')}]</td>`;
        html += `<td style="padding: 4px; font-weight: bold; color: ${det.product == -1 ? '#dc2626' : '#2563eb'}">${det.product == -1 ? '-' : '+'}${Math.abs(det.product)}</td>`;
        html += `</tr>`;
      });
      html += `</table>`;
      html += `전체 TRIM Parity 곱 합계: <strong>${data.z2 === 1 ? '-1' : '+1'}</strong><br>`;
      html += `Fu-Kane 공식으로 도출된 <strong>Z2 invariant: ${data.z2}</strong> `;
      html += `(${data.z2 === 1 ? '<span style="color: #b91c1c; font-weight:bold;">Topological (Non-trivial)</span>' : '<span style="color: #475569; font-weight:bold;">Trivial</span>'})`;
      resDiv.style.background = '#fef8ff';
      resDiv.style.borderColor = '#e9d5ff';
      resDiv.style.color = '#581c87';
    }
    resDiv.innerHTML = html;

  } catch (err) {
    resDiv.innerHTML = `<span style="color: #b91c1c;">⚠️ 계산 중 오류가 발생했습니다: ${err}</span>`;
  }
}




