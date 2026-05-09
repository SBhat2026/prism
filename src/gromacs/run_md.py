"""
GROMACS MD pipeline wrapper for interface energy refinement.

Workflow: pdb2gmx → editconf → solvate → genion → em → nvt → (short NPT) → extract energy
This is the optional physics-based refinement step. The buried SASA from extract_sasa.py
is the baseline; this provides a more accurate ΔG_interface for the GNN training targets.
"""

import os
import subprocess
import shutil
import tempfile
from pathlib import Path

GMX = os.environ.get("GMX_BIN", "gmx")
MDP_DIR = Path(__file__).parent.parent.parent / "mdp"


def _run(cmd: list, cwd: str, stdin: str = "") -> tuple[str, str]:
    result = subprocess.run(
        cmd, cwd=cwd, capture_output=True, text=True, input=stdin
    )
    if result.returncode != 0:
        raise RuntimeError(f"{cmd[0]} {cmd[1]} failed:\n{result.stderr[-1000:]}")
    return result.stdout, result.stderr


def prepare_topology(pdb_path: str, workdir: str, ff: str = "amber99sb-ildn") -> str:
    """Run pdb2gmx. Returns path to processed .gro."""
    gro = os.path.join(workdir, "processed.gro")
    _run(
        [GMX, "pdb2gmx", "-f", pdb_path, "-o", gro,
         "-p", os.path.join(workdir, "topol.top"),
         "-ff", ff, "-water", "tip3p", "-quiet"],
        cwd=workdir,
    )
    return gro


def solvate_and_ion(gro_path: str, top_path: str, workdir: str) -> str:
    """Box → solvate → neutralize. Returns final .gro path."""
    boxed = os.path.join(workdir, "boxed.gro")
    solvated = os.path.join(workdir, "solvated.gro")
    ions_tpr = os.path.join(workdir, "ions.tpr")
    neutral = os.path.join(workdir, "neutral.gro")

    _run([GMX, "editconf", "-f", gro_path, "-o", boxed,
          "-c", "-d", "1.0", "-bt", "cubic", "-quiet"], cwd=workdir)
    _run([GMX, "solvate", "-cp", boxed, "-cs", "spc216.gro",
          "-o", solvated, "-p", top_path, "-quiet"], cwd=workdir)
    _run([GMX, "grompp", "-f", str(MDP_DIR / "ions.mdp"), "-c", solvated,
          "-p", top_path, "-o", ions_tpr, "-quiet", "-maxwarn", "2"], cwd=workdir)
    _run([GMX, "genion", "-s", ions_tpr, "-o", neutral,
          "-p", top_path, "-pname", "NA", "-nname", "CL", "-neutral", "-quiet"],
         cwd=workdir, stdin="SOL\n")
    return neutral


def energy_minimization(gro_path: str, top_path: str, workdir: str) -> str:
    """EM run. Returns em.gro path."""
    tpr = os.path.join(workdir, "em.tpr")
    out = os.path.join(workdir, "em.gro")
    _run([GMX, "grompp", "-f", str(MDP_DIR / "em.mdp"), "-c", gro_path,
          "-p", top_path, "-o", tpr, "-quiet", "-maxwarn", "2"], cwd=workdir)
    _run([GMX, "mdrun", "-s", tpr, "-o", os.path.join(workdir, "em.trr"),
          "-c", out, "-e", os.path.join(workdir, "em.edr"),
          "-g", os.path.join(workdir, "em.log"), "-quiet"], cwd=workdir)
    return out


def nvt_equilibration(gro_path: str, top_path: str, workdir: str) -> str:
    """NVT run. Returns nvt.gro path."""
    tpr = os.path.join(workdir, "nvt.tpr")
    out = os.path.join(workdir, "nvt.gro")
    _run([GMX, "grompp", "-f", str(MDP_DIR / "nvt.mdp"), "-c", gro_path,
          "-p", top_path, "-o", tpr, "-quiet", "-maxwarn", "2",
          "-r", gro_path], cwd=workdir)
    _run([GMX, "mdrun", "-s", tpr, "-o", os.path.join(workdir, "nvt.trr"),
          "-c", out, "-e", os.path.join(workdir, "nvt.edr"),
          "-g", os.path.join(workdir, "nvt.log"), "-quiet"], cwd=workdir)
    return out


def extract_interface_energy(edr_path: str, workdir: str) -> dict[str, float]:
    """
    Extract LJ-SR and Coul-SR between Protein_chain_A and Protein_chain_B.
    Returns dict with keys: lj_sr, coul_sr, total.
    """
    stdout, _ = _run(
        [GMX, "energy", "-f", edr_path, "-o", os.path.join(workdir, "energy.xvg")],
        cwd=workdir,
        stdin="Coul-SR:Protein_chain_A-Protein_chain_B\n"
              "LJ-SR:Protein_chain_A-Protein_chain_B\n0\n",
    )
    energies: dict[str, float] = {}
    for line in stdout.splitlines():
        if "Coul-SR" in line:
            parts = line.split()
            energies["coul_sr"] = float(parts[1])
        if "LJ-SR" in line:
            parts = line.split()
            energies["lj_sr"] = float(parts[1])
    energies["total"] = energies.get("coul_sr", 0) + energies.get("lj_sr", 0)
    return energies


def run_interface_pipeline(
    pdb_path: str,
    run_nvt: bool = False,
    keep_workdir: bool = False,
) -> dict[str, float]:
    """
    Full GROMACS pipeline: pdb2gmx → solvate → EM → (NVT) → interface energy.
    Returns energy dict. workdir is deleted unless keep_workdir=True.
    """
    workdir = tempfile.mkdtemp(prefix="gmx_interface_")
    try:
        gro  = prepare_topology(pdb_path, workdir)
        top  = os.path.join(workdir, "topol.top")
        gro  = solvate_and_ion(gro, top, workdir)
        gro  = energy_minimization(gro, top, workdir)
        edr  = os.path.join(workdir, "em.edr")
        if run_nvt:
            gro = nvt_equilibration(gro, top, workdir)
            edr = os.path.join(workdir, "nvt.edr")
        result = extract_interface_energy(edr, workdir)
        if keep_workdir:
            result["workdir"] = workdir
        return result
    finally:
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
