#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import importlib.util
from typing import Optional

# ---------- Configure your module (basename must match PYBIND11_MODULE) ----------
MODULE_NAME = "zcx_2d_natural_convection_RL_python"

# Optional: allow overriding the search directory via env
ENV_DIR = os.environ.get("SPH_PYBIND_LIB_DIR", "").strip()


# ---------- Helpers ----------
def _candidate_dirs() -> list[str]:
    """
    Build candidate directories to search for the compiled extension.
    Priority:
      1) ENV_DIR (if provided)
      2) lib/Release
      3) lib/RelWithDebInfo
      4) lib/Debug
      5) lib
    Base is the project folder = one level up from this script's dir.
    """
    # this script is typically under .../bin/bind/pybind_test.py
    base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # -> .../<case>/
    lib = os.path.join(base, "lib")
    dirs = []
    if ENV_DIR:
        dirs.append(ENV_DIR)
    # common cmake configurations
    for cfg in ("Release", "RelWithDebInfo", "Debug", ""):
        d = os.path.join(lib, cfg) if cfg else lib
        dirs.append(d)
    # unique + keep order
    seen, uniq = set(), []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            uniq.append(d)
    return uniq


def _candidate_patterns() -> list[str]:
    """
    Build filename patterns for Windows/Linux ABI variants.
    """
    pyver = f"{sys.version_info.major}{sys.version_info.minor}"
    patterns = []

    if os.name == "nt":
        # Typical pybind11 wheel name produced by cmake on Windows
        patterns += [
            f"{MODULE_NAME}.cp{pyver}-win_amd64.pyd",  # most common
            f"{MODULE_NAME}.pyd",  # fallback (rare)
        ]
    else:
        # Linux: allow various cpython ABI suffixes and plain .so
        patterns += [
            f"{MODULE_NAME}.cpython-{pyver}*.so",  # e.g. cpython-310-x86_64-linux-gnu.so
            f"{MODULE_NAME}.abi3*.so",  # abi3 builds (unlikely here)
            f"{MODULE_NAME}.so",  # plain .so from cmake
        ]
    return patterns


def locate_extension() -> Optional[str]:
    """
    Search for the extension file (pyd/so) and return its absolute path.
    """
    for d in _candidate_dirs():
        if not os.path.isdir(d):
            continue
        for pat in _candidate_patterns():
            matches = glob.glob(os.path.join(d, pat))
            if matches:
                # If multiple matches, prefer the most recently modified
                matches.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return os.path.abspath(matches[0])
    return None


def load_extension(path: str):
    """
    Load the extension module from an absolute path via importlib, bypassing sys.path.
    Returns the loaded module object.
    """
    spec = importlib.util.spec_from_file_location(MODULE_NAME, path)
    if not spec or not spec.loader:
        raise ImportError(f"Cannot create spec/loader for {MODULE_NAME} at: {path}")
    mod = importlib.util.module_from_spec(spec)
    # register in sys.modules for downstream imports that rely on module name
    sys.modules[MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------- Load module (robust) ----------
def ensure_module():
    """
    Ensure MODULE_NAME is importable as 'mod' either by absolute load or fallback to sys.path insert.
    """
    # 1) Try absolute-path load first
    ext = locate_extension()
    if ext and os.path.exists(ext):
        print(f"[Loader] Using compiled extension: {ext}")
        return load_extension(ext)

    # 2) Fallback: try to import normally after pushing search dirs to sys.path
    for d in _candidate_dirs():
        if os.path.isdir(d) and d not in sys.path:
            sys.path.insert(0, d)
    try:
        __import__(MODULE_NAME)
        print(f"[Loader] Imported '{MODULE_NAME}' via sys.path: {sys.modules[MODULE_NAME].__file__}")
        return sys.modules[MODULE_NAME]
    except Exception as e:
        msg = [
            f"Failed to import '{MODULE_NAME}'.",
            f"Checked directories: {', '.join([d for d in _candidate_dirs() if os.path.isdir(d)])}",
            "Tips:",
            "  - Make sure you built the same Python version/arch (e.g. cp310-win_amd64).",
            "  - Set SPH_PYBIND_LIB_DIR to your custom lib directory if needed.",
            "  - Verify PYBIND11_MODULE name matches MODULE_NAME.",
        ]
        raise ImportError("\n".join(msg)) from e


# ---------- Your case runner ----------
def run_case():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel_env", default=0, type=int)
    parser.add_argument("--episode_env", default=0, type=int)
    parser.add_argument("--end_time", default=120.0, type=float)
    args = parser.parse_args()

    mod = ensure_module()

    # Inspect exports to help debugging name mismatches
    # print("[Exports]", [n for n in dir(mod) if "convection" in n.lower() or "sph" in n.lower()])

    # Call into the bound function/class; change the name here if your binding exports a class instead
    project = getattr(mod, "natural_convection_from_sph_cpp")(args.parallel_env, args.episode_env)

    # Keep method names consistent with what pybind11 exports (case-sensitive!)
    if getattr(project, "cmake_test")() == 1:
        project.run_case(args.end_time)
    else:
        print("[Warn] cmake_test() failed — check runtime working directory and input files.")


if __name__ == "__main__":
    run_case()
