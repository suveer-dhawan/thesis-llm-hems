# Observation-only wrapper: logs per-attempt usage metadata; no behavioural change.
import json, runpy, sys, time
from google.genai import models
LOG = sys.argv.pop(1)
_orig = models.Models.generate_content
def wrapped(self, *a, **kw):
    t0 = time.time(); rec = {"t_start": t0}
    try:
        r = _orig(self, *a, **kw)
        u = r.usage_metadata
        rec.update(ok=True, model_version=r.model_version,
                   finish_reason=str(r.candidates[0].finish_reason) if r.candidates else None,
                   prompt=getattr(u,"prompt_token_count",None), output=getattr(u,"candidates_token_count",None),
                   thinking=getattr(u,"thoughts_token_count",None), total=getattr(u,"total_token_count",None),
                   text_len=len(r.text or ""))
        return r
    except Exception as e:
        rec.update(ok=False, error=repr(e)[:500]); raise
    finally:
        rec["secs"] = time.time() - t0
        with open(LOG, "a") as f: f.write(json.dumps(rec) + "\n")
models.Models.generate_content = wrapped
sys.argv = ["experiments/run_phase3.py"] + sys.argv[1:]
runpy.run_path("experiments/run_phase3.py", run_name="__main__")
