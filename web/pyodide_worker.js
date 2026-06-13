// pyodide_worker.js
// Runs Pyodide in a background thread to prevent UI freezing.

importScripts("https://cdn.jsdelivr.net/pyodide/v0.26.4/full/pyodide.js");

let pyodide = null;

const PY_FILES = [
  'cls_finder/__init__.py',
  'cls_finder/core/__init__.py',
  'cls_finder/core/laurent.py',
  'cls_finder/core/matrixpoly.py',
  'cls_finder/core/lattice.py',
  'cls_finder/core/lattice_classify.py',
  'cls_finder/io/__init__.py',
  'cls_finder/io/parser.py',
  'cls_finder/band/__init__.py',
  'cls_finder/band/bands.py',
  'cls_finder/eigen/__init__.py',
  'cls_finder/eigen/eigenstate.py',
  'cls_finder/classify/__init__.py',
  'cls_finder/classify/singularity.py',
  'cls_finder/cls/__init__.py',
  'cls_finder/cls/analytic.py',
  'cls_finder/cls/numeric.py',
  'cls_finder/cls/reduce.py',
  'cls_finder/cls/noncontractible.py',
  'cls_finder/cls/gauge_analysis.py',
  'cls_finder/models/__init__.py',
  'cls_finder/models/library.py',
  'cls_finder/rbm/__init__.py',
  'cls_finder/rbm/boundary_mode.py',
  'cls_finder/classify/chern.py',
];

async function initPyodide(msgId) {
  try {
    postMessage({ id: msgId, type: 'progress', pct: 5, msg: 'Pyodide 런타임 다운로드 중...' });
    pyodide = await loadPyodide({
      indexURL: 'https://cdn.jsdelivr.net/pyodide/v0.26.4/full/'
    });

    postMessage({ id: msgId, type: 'progress', pct: 25, msg: 'NumPy 로드 중...' });
    await pyodide.loadPackage('numpy');

    postMessage({ id: msgId, type: 'progress', pct: 45, msg: 'SciPy 로드 중...' });
    await pyodide.loadPackage('scipy');

    postMessage({ id: msgId, type: 'progress', pct: 65, msg: 'SymPy 로드 중...' });
    await pyodide.loadPackage('sympy');

    postMessage({ id: msgId, type: 'progress', pct: 78, msg: 'CLS Finder 모듈 로드 중...' });
    await loadClsModules();

    postMessage({ id: msgId, type: 'progress', pct: 92, msg: 'Bridge 초기화 중...' });
    const bridgeResp = await fetch('bridge.py?t=' + new Date().getTime());
    const bridgeCode = await bridgeResp.text();
    await pyodide.runPythonAsync(bridgeCode);

    postMessage({ id: msgId, type: 'progress', pct: 100, msg: '완료!' });

    // Retrieve models list to send back
    const raw = pyodide.globals.get('get_models_list')();
    postMessage({ id: msgId, type: 'init-success', modelsList: raw });
  } catch (err) {
    postMessage({ id: msgId, type: 'init-failure', error: err.message });
  }
}

async function loadClsModules() {
  const dirs = new Set();
  for (const f of PY_FILES) {
    const parts = f.split('/');
    for (let i = 1; i <= parts.length - 1; i++) {
      dirs.add(parts.slice(0, i).join('/'));
    }
  }
  for (const d of dirs) {
    try { pyodide.FS.mkdir(d); } catch (_) {}
  }

  for (const f of PY_FILES) {
    try {
      const resp = await fetch('../' + f + '?t=' + new Date().getTime());
      if (!resp.ok) {
        pyodide.FS.writeFile(f, '');
        continue;
      }
      const txt = await resp.text();
      pyodide.FS.writeFile(f, txt);
    } catch (e) {
      pyodide.FS.writeFile(f, '');
    }
  }

  pyodide.runPython(`
import sys
if '.' not in sys.path: sys.path.insert(0, '.')
`);
}

self.onmessage = async function(e) {
  const data = e.data;
  const msgId = data.id;

  if (data.type === 'init') {
    await initPyodide(msgId);
  } else if (data.type === 'get_model_spec') {
    try {
      const raw = pyodide.globals.get('get_model_spec')(data.modelId);
      postMessage({ id: msgId, type: 'get_model_spec-success', specJson: raw });
    } catch (err) {
      postMessage({ id: msgId, type: 'get_model_spec-failure', error: err.message });
    }
  } else if (data.type === 'run_analysis') {
    try {
      pyodide.globals.set('_web_spec_json', data.specJson);
      const raw = await pyodide.runPythonAsync('run_analysis(_web_spec_json)');
      postMessage({ id: msgId, type: 'run_analysis-success', resultJson: raw });
    } catch (err) {
      postMessage({ id: msgId, type: 'run_analysis-failure', error: err.message });
    }
  }
};
