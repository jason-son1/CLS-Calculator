import pytest
import json
import web.bridge as bridge
from cls_finder.models import library

def test_run_analysis_stream():
    # Load preset model spec
    zigzag_spec = library.zigzag_chain()
    spec_json = json.dumps(zigzag_spec)
    
    # 1. Test run_analysis_stream generator behavior
    stream = bridge.run_analysis_stream(spec_json)
    
    chunks = list(stream)
    assert len(chunks) > 0, "Stream should yield at least one chunk"
    
    # Verify all chunks are valid JSON strings
    parsed_chunks = []
    for chunk in chunks:
        assert isinstance(chunk, str)
        parsed = json.loads(chunk)
        parsed_chunks.append(parsed)
        
    # First chunk should be a log entry
    assert parsed_chunks[0]["type"] == "log"
    assert parsed_chunks[0]["step"]["status"] == "running"
    
    # Last chunk should be the result
    assert parsed_chunks[-1]["type"] == "result"
    assert "result" in parsed_chunks[-1]
    
    result = parsed_chunks[-1]["result"]
    assert "steps" in result
    assert "band_plot" in result
    assert "flat_bands" in result
    
    # 2. Test run_analysis wrapper (compatibility check)
    result_str = bridge.run_analysis(spec_json)
    assert isinstance(result_str, str)
    
    compat_result = json.loads(result_str)
    assert "steps" in compat_result
    assert "band_plot" in compat_result
    assert "flat_bands" in compat_result
    
    # Compare with streaming result
    assert len(compat_result["steps"]) == len(result["steps"])
    assert compat_result["flat_bands"] == result["flat_bands"]


def test_nearly_flat_band_singularity_processing():
    spec = library.kagome_nn()
    spec["H_symbolic"][0][0] = "0.05"  # perturb flat band to be dispersive
    spec["options"]["k_grid"] = [20, 20]  # faster grid for testing
    spec["options"]["nearly_flat_ratio"] = 0.08
    spec_json = json.dumps(spec)
    
    stream = bridge.run_analysis_stream(spec_json)
    parsed_chunks = [json.loads(c) for c in stream]
    
    result = parsed_chunks[-1]["result"]
    flat_bands = result.get("flat_bands", [])
    
    assert len(flat_bands) > 0, "Nearly flat bands should be processed when flat bands are absent"
    assert any(fb.get("is_nearly_flat") for fb in flat_bands)
