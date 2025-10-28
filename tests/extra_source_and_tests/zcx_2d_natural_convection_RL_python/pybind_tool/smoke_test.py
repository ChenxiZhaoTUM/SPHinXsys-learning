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
def run_smoke_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel_env", default=0, type=int)
    parser.add_argument("--episode_env", default=0, type=int)
    parser.add_argument("--extra_run", default=20.0, type=float,
                        help="Extra physical time to advance after debug_smoke_test()")
    args = parser.parse_args()

    mod = ensure_module()

    solver = getattr(mod, "natural_convection_from_sph_cpp")(args.parallel_env, args.episode_env)

    print("=== [1] calling debug_smoke_test() ===")
    # this runs: set some temps, advance 10s, print flux etc. on the C++ side
    ret = solver.debug_smoke_test()
    print(f"debug_smoke_test() returned: {ret}")

    print("\n=== [2] manual temperature update / advance / readback ===")

    # Pick a new temperature pattern for the 3 segments.
    # IMPORTANT: This length must match n_seg in your C++ world (currently assumed =3).
    new_temps = [1.0, 0.5, 1.5, 1.0]
    print(f"Setting segment temps to {new_temps}")
    solver.set_segment_temperatures(new_temps)

    # Advance some more physical time.
    # NOTE: run_case(target_time) uses ABSOLUTE physical time,
    # so we just say "go to current_time + extra_run".
    # We don't know current_time directly from Python, so the easiest hack is:
    #   - just call run_case(1e9) for testing and trust it won't blow up.
    # For now we'll just do a modest guess and hope it's ahead of current time.
    solver.run_case(args.extra_run)  # absolute time

    # read observables
    global_flux = solver.get_global_heat_flux()
    flux_0 = solver.get_local_phi_flux(0)
    flux_1 = solver.get_local_phi_flux(1)
    flux_2 = solver.get_local_phi_flux(2)
    flux_3 = solver.get_local_phi_flux(3)
    ke_global = solver.get_global_kinetic_energy()
    vx0 = solver.get_local_velocity(0, 0)
    vy0 = solver.get_local_velocity(0, 1)

    print("After manual run_case:")
    print(f"  global_flux = {global_flux}")
    print(f"  local_flux[0,1,2,3] = {flux_0}, {flux_1}, {flux_2}, {flux_3}")
    print(f"  global_kinetic_energy = {ke_global}")
    print(f"  probe0 vel = ({vx0}, {vy0})")

    print("\nSmoke test finished.\n")


if __name__ == "__main__":
    run_smoke_test()
