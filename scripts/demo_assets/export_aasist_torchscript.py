# scripts/export_aasist_torchscript.py
import sys, json, argparse
from pathlib import Path
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--aasist-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "aasist"))
parser.add_argument("--ckpt", type=str, default="")
parser.add_argument("--config-py", type=str, default="", help="Path to AASIST config .py exposing d_args/model_config (optional if using .conf)")
parser.add_argument("--config-json", type=str, default="", help="Path to AASIST JSON .conf (contains model_config)")
parser.add_argument("--out", type=str, default=str(Path(__file__).resolve().parents[1] / "backend" / "models" / "aasist_scripted.pt"))
args = parser.parse_args()

# Paths
ROOT = Path(__file__).resolve().parents[1]          # repo root
AASIST_DIR = Path(args.aasist_dir)
if args.ckpt:
    CKPT = Path(args.ckpt)
else:
    CKPT = AASIST_DIR / "models" / "weights" / "AASIST-L.pth"
    if not CKPT.exists():
        CKPT = AASIST_DIR / "model" / "weights" / "AASIST-L.pth"
    if not CKPT.exists():
        CKPT = AASIST_DIR / "models" / "weights" / "AASIST.pth"

OUT = Path(args.out)
OUT.parent.mkdir(parents=True, exist_ok=True)

# Import the model class from the cloned repo
sys.path.append(str(AASIST_DIR))
sys.path.append(str(AASIST_DIR / "models"))
try:
    from AASIST import Model  # type: ignore
except Exception:
    # Some repos package model under submodule, try alternative import
    from models.AASIST import Model  # type: ignore

# Resolve d_args
def load_d_args() -> dict:
    # 0) JSON .conf supplied (preferred with this repo)
    if args.config_json:
        import json
        with open(args.config_json, "r") as f:
            cfg = json.load(f)
        d = cfg.get("model_config") or cfg.get("d_args") or {}
        if isinstance(d, dict):
            return d
        raise RuntimeError("--config-json provided but no model_config/d_args found")
    # 1) If a config .py is provided, import and try common symbols
    if args.config_py:
        import importlib.util
        cfg_path = Path(args.config_py)
        spec = importlib.util.spec_from_file_location("aasist_cfg", str(cfg_path))
        if spec and spec.loader:
            cfg = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cfg)  # type: ignore
            for name in ("d_args", "model_config", "args", "config"):
                if hasattr(cfg, name):
                    obj = getattr(cfg, name)
                    if isinstance(obj, dict):
                        return obj
                    if callable(obj):
                        try:
                            val = obj()
                            if isinstance(val, dict):
                                return val
                        except Exception:
                            pass
        raise RuntimeError("--config-py provided but no dict-like d_args/model_config found")

    # 2) Try to load from checkpoint if it contains config-like entries
    try:
        state = torch.load(CKPT, map_location="cpu")
        for key in ("d_args", "model_config", "config", "args"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    except Exception:
        pass

    # 3) Attempt a repo-local config module
    for candidate in [AASIST_DIR / "config.py", AASIST_DIR / "models" / "config.py"]:
        if candidate.exists():
            import importlib.util
            spec = importlib.util.spec_from_file_location("aasist_cfg", str(candidate))
            if spec and spec.loader:
                cfg = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(cfg)  # type: ignore
                for name in ("d_args", "model_config", "args", "config"):
                    if hasattr(cfg, name):
                        obj = getattr(cfg, name)
                        if isinstance(obj, dict):
                            return obj
                        if callable(obj):
                            try:
                                val = obj()
                                if isinstance(val, dict):
                                    return val
                            except Exception:
                                pass

    raise RuntimeError(
        "Could not infer AASIST model config. Pass --config-py pointing to the repo's config .py that defines d_args/model_config."
    )

# Build model & load weights
_d_args = load_d_args()
model = Model(d_args=_d_args)
state = torch.load(CKPT, map_location="cpu")
state_dict = state.get("state_dict", state)
state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
model.load_state_dict(state_dict, strict=True)
model.eval()

# --- TorchScript export ---
# This AASIST variant expects raw waveform of length nb_samp (e.g., 64600)
nb_samp = int(_d_args.get("nb_samp", 64600))
example = torch.randn(1, nb_samp)

try:
    scripted = torch.jit.script(model)   # best if control-flow is supported
except Exception:
    # Fallback to trace if torchscript fails
    scripted = torch.jit.trace(model, example, strict=False)

# optional: small runtime optim
try:
    scripted = torch.jit.optimize_for_inference(scripted)
except Exception:
    pass

scripted.save(str(OUT))
print(json.dumps({"ok": True, "saved_to": str(OUT), "ckpt": str(CKPT)}))