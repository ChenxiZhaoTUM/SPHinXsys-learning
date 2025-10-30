#!/usr/bin/env python3
import os
import sys
import glob
import argparse
import importlib.util
from typing import Optional
import re

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


# ---------- small utils ----------
def mkdir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def latest_restart_step(restart_dir: str) -> int:
    """Parse *_rst_XXXXXXXXXX.xml and return the max XXXXX..."""
    steps = []
    for f in glob.glob(os.path.join(restart_dir, "*_rst_*.xml")):
        m = re.search(r"_rst_(\d+)\.xml$", f.replace("\\", "/"))
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else 0


# ---------- Your case runner ----------
def run_case():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parallel_env", default=0, type=int)
    parser.add_argument("--episode_env", default=0, type=int)
    parser.add_argument("--set_restart_step", default=0, type=int)
    parser.add_argument("--n_seg", default=3, type=int)

    # how long to warm up the baseline (uniform T=2.0)
    parser.add_argument("--warmup_time", default=120.0, type=float)

    # how long to continue after we apply a control action
    # (this should match your env.window_total / sim_time chunk, e.g. 5.0s)
    parser.add_argument("--control_horizon", default=5.0, type=float)

    args = parser.parse_args()
    mod = ensure_module()

    bind_dir = os.path.dirname(__file__)
    bind_results_dir = mkdir(os.path.join(bind_dir, "bind_results"))

    # --------------------------------------------------------------------------------
    # Stage A. Baseline warmup: bottom wall = [2.0, 2.0, ...], run to warmup_time
    # --------------------------------------------------------------------------------
    print("\n=== Stage A: warm-up & write restart ===")
    os.chdir(bind_results_dir)
    # 1) create solver instance
    simA = getattr(mod, "natural_convection_from_sph_cpp")(args.parallel_env, args.episode_env, args.set_restart_step)
    print("[Info] Solver constructed.")

    baseline_wall = [2.0] * args.n_seg
    print(f"[Info] Setting initial baseline wall temps = {baseline_wall}")
    simA.set_segment_temperatures(baseline_wall)

    # advance CFD to warmup_time (absolute physical time = args.warmup_time)
    print(f"[Info] Running baseline to t = {args.warmup_time} ...")
    simA.run_case(args.warmup_time)

    # read baseline observation
    base_global_flux = simA.get_global_heat_flux()
    base_flux0 = simA.get_local_phi_flux(0)
    base_flux1 = simA.get_local_phi_flux(1)
    base_flux2 = simA.get_local_phi_flux(2)
    base_ke_global = simA.get_global_kinetic_energy()
    base_vx0 = simA.get_local_velocity(0, 0)
    base_vy0 = simA.get_local_velocity(0, 1)

    print("=== Baseline state after warmup ===")
    print(f"  global_heat_flux         = {base_global_flux}")
    print(f"  local_heat_flux[0,1,2]   = {base_flux0}, {base_flux1}, {base_flux2}")
    print(f"  global_kinetic_energy    = {base_ke_global}")
    print(f"  probe0 velocity (vx,vy)  = ({base_vx0}, {base_vy0})")
    print("===================================")

    restart_dir = os.path.join(bind_results_dir, "restart")
    latest_step = latest_restart_step(restart_dir)
    print(f"[Warmup] latest restart step = {latest_step}")

    # --------------------------------------------------------------------------------
    # Stage B. Apply a "control" temperature pattern and run a short horizon
    # --------------------------------------------------------------------------------
    print("\n=== Stage B: load restart, top-up to 120s, control +5s ===")

    step_to_load = latest_step
    if step_to_load <= 0:
        raise RuntimeError("No restart step found to load. Did Stage A write files?")

    simB = getattr(mod, "natural_convection_from_sph_cpp")(args.parallel_env, args.episode_env, int(step_to_load))
    print(f"[Info] Solver B constructed from restart step {step_to_load}.")

    # After loading, the physical time is approximately 119s.
    # First, advance to 120s (target absolute time = warmup_time)
    # print(f"[Info] Advancing to exactly t={args.warmup_time} ...")
    # simB.run_case(args.warmup_time)

    # This mimics one RL step: choose action → convert to per-segment temps → advance
    # Here we just hand-pick something not uniform to see an effect:
    control_wall = [2.3, 2.0, 1.7, 2.0]
    print(f"[Info] Applying control wall temps = {control_wall}")
    simB.set_segment_temperatures(control_wall)

    target_time = args.warmup_time + args.control_horizon
    print(f"[Info] Advancing simulation to t = {target_time} ...")
    simB.run_case(target_time)

    # read post-control observation
    post_global_flux = simB.get_global_heat_flux()
    post_flux0 = simB.get_local_phi_flux(0)
    post_flux1 = simB.get_local_phi_flux(1)
    post_flux2 = simB.get_local_phi_flux(2)
    post_ke_global = simB.get_global_kinetic_energy()
    post_vx0 = simB.get_local_velocity(0, 0)
    post_vy0 = simB.get_local_velocity(0, 1)

    print("=== Post-control state ===")
    print(f"  global_heat_flux         = {post_global_flux}")
    print(f"  local_heat_flux[0,1,2]   = {post_flux0}, {post_flux1}, {post_flux2}")
    print(f"  global_kinetic_energy    = {post_ke_global}")
    print(f"  probe0 velocity (vx,vy)  = ({post_vx0}, {post_vy0})")
    print("===========================")

    # If we got this far without crashing, bindings + stepping loop basically work.
    print("[Info] pybind smoke run complete.")


if __name__ == "__main__":
    run_case()
