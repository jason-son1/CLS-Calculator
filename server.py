"""
FastAPI backend for CLS Finder.
Replaces Pyodide/WebAssembly with native Python for full CPU performance.

Usage:
    uvicorn server:app --host localhost --port 8765 --reload
"""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "web"))

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Any

class NoCacheStaticFiles(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False
        
    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response
import bridge

app = FastAPI(title="CLS Finder API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class SpecPayload(BaseModel):
    spec: Any


@app.get("/api/models")
def api_models():
    return JSONResponse(content=json.loads(bridge.get_models_list()))


@app.get("/api/model_spec/{model_id}")
def api_model_spec(model_id: str):
    result = json.loads(bridge.get_model_spec(model_id))
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return JSONResponse(content=result)


class SimplifyMatrixPayload(BaseModel):
    matrix: list[list[str]]
    precision: int = 5
    threshold: float = 1e-4


@app.post("/api/simplify_matrix")
def api_simplify_matrix(payload: SimplifyMatrixPayload):
    import sympy
    import re

    simplified_matrix = []
    kx, ky, kz = sympy.symbols('kx ky kz', real=True)
    locals_dict = {
        'I': sympy.I, 'pi': sympy.pi,
        'sqrt': sympy.sqrt, 'exp': sympy.exp,
        'cos': sympy.cos, 'sin': sympy.sin,
        'tan': sympy.tan, 'abs': sympy.Abs
    }

    precision = payload.precision
    threshold = payload.threshold

    def clean_float_string(expr_str):
        # Convert floats like 2.0 to 2
        return re.sub(r'\b(\d+)\.0\b', r'\1', expr_str)

    def replace_fn(x):
        if abs(x) < threshold:
            return sympy.Float(0)
        return sympy.Float(round(x, precision))

    for row in payload.matrix:
        simplified_row = []
        for cell in row:
            cell_str = str(cell).strip()
            if cell_str == '0' or cell_str == '':
                simplified_row.append('0')
                continue
            try:
                # Pre-process: add implicit multiplication between digit and letter
                preprocessed = cell_str
                preprocessed = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', preprocessed)
                preprocessed = re.sub(r'(\))(\w)', r'\1*\2', preprocessed)

                expr = sympy.sympify(preprocessed, locals=locals_dict)
                
                # Round and prune float constants
                if threshold > 0:
                    expr = expr.replace(lambda x: isinstance(x, sympy.Float), replace_fn)
                elif precision is not None:
                    expr = expr.replace(lambda x: isinstance(x, sympy.Float), lambda x: sympy.Float(round(x, precision)))
                
                # Rewrite exponentials to cosine/sine where possible, and simplify
                simplified_expr = expr.rewrite(sympy.cos).simplify()
                simplified_str = str(simplified_expr)

                # Post-process to make it look clean
                simplified_str = clean_float_string(simplified_str)
                simplified_row.append(simplified_str)
            except Exception:
                # Fallback to original cell if simplification fails
                simplified_row.append(cell_str)
        simplified_matrix.append(simplified_row)

    return JSONResponse(content={"matrix": simplified_matrix})


@app.post("/api/band_data")
def api_band_data(payload: SpecPayload):
    result = bridge.get_band_data(json.dumps(payload.spec))
    return JSONResponse(content=json.loads(result))


@app.post("/api/run_analysis")
def api_run_analysis(payload: SpecPayload):
    result = bridge.run_analysis(json.dumps(payload.spec))
    return JSONResponse(content=json.loads(result))


@app.post("/api/run_analysis_stream")
def api_run_analysis_stream(payload: SpecPayload):
    def event_generator():
        for chunk in bridge.run_analysis_stream(json.dumps(payload.spec)):
            yield chunk + "\n"
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


class RibbonPayload(BaseModel):
    spec: Any
    Nx: int
    Nk: int
    selected_bands: list[int] = None
    ky_min: float = -3.141592653589793
    ky_max: float = 3.141592653589793
    periodic_dir: str = "y"
    k_fixed: float = 0.0


class RibbonStatePayload(BaseModel):
    spec: Any
    Nx: int
    ky: float
    band_idx: int
    periodic_dir: str = "y"


@app.post("/api/nanoribbon_data")
def api_nanoribbon_data(payload: RibbonPayload):
    result = bridge.get_nanoribbon_data(
        json.dumps(payload.spec), 
        payload.Nx, 
        payload.Nk, 
        payload.selected_bands,
        payload.ky_min,
        payload.ky_max,
        payload.periodic_dir,
        payload.k_fixed
    )
    return JSONResponse(content=json.loads(result))


@app.post("/api/nanoribbon_state")
def api_nanoribbon_state(payload: RibbonStatePayload):
    result = bridge.get_nanoribbon_state(
        json.dumps(payload.spec),
        payload.Nx,
        payload.ky,
        payload.band_idx,
        payload.periodic_dir
    )
    return JSONResponse(content=json.loads(result))


class RBMPayload(BaseModel):
    spec: Any
    Nx: int
    Ny: int | None = None
    Nz: int | None = None
    band_index: int | None = None
    defect_cell: list[int] | None = None
    k0_override: list[float] | None = None


@app.post("/api/rbm_data")
def api_rbm_data(payload: RBMPayload):
    result = bridge.compute_rbm(
        json.dumps(payload.spec),
        payload.Nx,
        payload.Ny,
        payload.Nz,
        payload.band_index,
        payload.defect_cell,
        payload.k0_override,
    )
    return JSONResponse(content=json.loads(result))


class ChernPayload(BaseModel):
    spec: Any
    band_index: int | None = None
    grid_n: int = 24
    scan_n: int = 120


@app.post("/api/chern")
def api_chern(payload: ChernPayload):
    result = bridge.compute_chern(
        json.dumps(payload.spec),
        payload.band_index,
        payload.grid_n,
        payload.scan_n,
    )
    return JSONResponse(content=json.loads(result))

class FlatBandDesignPayload(BaseModel):
    lattice_spec: Any
    target: Any
    E0: float = 0.0
    t: float = 0.3
    delta: float = 0.5
    n_grid_ift: int = 24
    R_cut: int = 3
    max_retries: int = 8
    max_rcut_retries: int = 4
    r_max: int | None = None
    cls_size: int | None = None


@app.post("/api/design_flat_band")
def api_design_flat_band(payload: FlatBandDesignPayload):
    result = bridge.design_flat_band(
        json.dumps(payload.lattice_spec),
        json.dumps(payload.target),
        payload.E0,
        payload.t,
        payload.delta,
        payload.n_grid_ift,
        payload.R_cut,
        payload.max_retries,
        payload.max_rcut_retries,
        payload.r_max,
        payload.cls_size,
    )
    return JSONResponse(content=json.loads(result))


class AnalyzeManualClsPayload(BaseModel):
    lattice_spec: Any
    cls_sites: Any
    E0: float = 0.0
    t: float = 0.3
    delta: float = 0.5
    n_grid_ift: int = 24
    R_cut: int = 3
    max_rcut_retries: int = 4


@app.post("/api/analyze_manual_cls")
def api_analyze_manual_cls(payload: AnalyzeManualClsPayload):
    result = bridge.analyze_manual_cls(
        json.dumps(payload.lattice_spec),
        json.dumps(payload.cls_sites),
        payload.E0,
        payload.t,
        payload.delta,
        payload.n_grid_ift,
        payload.R_cut,
        payload.max_rcut_retries,
    )
    return JSONResponse(content=json.loads(result))


class FlatBandExplorePayload(BaseModel):
    lattice_spec: Any
    target: Any
    E0: float = 0.0
    mk_variants: list[list[float]] | None = None
    offsets: list[int] | None = None
    rcut_variants: list[int] | None = None
    cls_sizes: list[int] | None = None
    n_grid_ift: int = 24
    max_retries: int = 2
    max_rcut_retries: int = 3
    max_candidates: int = 24


@app.post("/api/design_flat_band_explore_stream")
def api_design_flat_band_explore_stream(payload: FlatBandExplorePayload):
    def event_generator():
        for chunk in bridge.design_flat_band_explore_stream(
            json.dumps(payload.lattice_spec),
            json.dumps(payload.target),
            payload.E0,
            json.dumps(payload.mk_variants) if payload.mk_variants is not None else None,
            json.dumps(payload.offsets) if payload.offsets is not None else None,
            json.dumps(payload.rcut_variants) if payload.rcut_variants is not None else None,
            payload.n_grid_ift,
            payload.max_retries,
            payload.max_rcut_retries,
            payload.max_candidates,
            json.dumps(payload.cls_sizes) if payload.cls_sizes is not None else None,
        ):
            yield chunk + "\n"
    return StreamingResponse(event_generator(), media_type="application/x-ndjson")


# Serve static web files — must be mounted last so API routes take priority
app.mount("/", NoCacheStaticFiles(directory=str(ROOT / "web"), html=True), name="static")
