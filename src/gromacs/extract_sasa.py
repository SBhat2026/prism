"""
GROMACS-based interface SASA computation.
Implements the Marsh 2013 interface-area heuristic as the baseline scoring term.

Usage:
    sasa = GromacsSASA(gmx_bin="gmx")
    buried = sasa.interface_buried_area("complex.pdb", chain_a="A", chain_b="B")
"""

import os
import subprocess
import tempfile
import shutil
import re
from pathlib import Path


GMX = os.environ.get("GMX_BIN", "gmx")


def _run(cmd: list[str], cwd: str) -> str:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"GROMACS error:\n{result.stderr}")
    return result.stdout + result.stderr


def _parse_xvg_total(xvg_path: str) -> float:
    """Return the mean SASA value from an .xvg file (nm²)."""
    values = []
    with open(xvg_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(("#", "@")):
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    values.append(float(parts[1]))
                except ValueError:
                    pass
    return sum(values) / len(values) if values else 0.0


def _split_chains(pdb_path: str, chains: list[str], outdir: str) -> dict[str, str]:
    """Write one PDB per chain subset using ATOM records."""
    chain_pdbs: dict[str, str] = {}
    for chain_set in chains:
        chain_ids = set(chain_set)
        out_path = os.path.join(outdir, f"chain_{''.join(sorted(chain_ids))}.pdb")
        lines = []
        with open(pdb_path) as f:
            for line in f:
                if line[:4] in ("ATOM", "HETA") and line[21:22] in chain_ids:
                    lines.append(line)
                elif line[:3] == "END":
                    lines.append(line)
        if not lines:
            raise ValueError(f"No atoms found for chains {chain_ids} in {pdb_path}")
        lines.append("END\n")
        with open(out_path, "w") as f:
            f.writelines(lines)
        chain_pdbs[chain_set] = out_path
    return chain_pdbs


def _gmx_sasa_on_pdb(pdb_path: str, workdir: str, tag: str) -> float:
    """
    Compute total SASA (nm²) of all atoms in pdb_path using gmx sasa.
    Creates a minimal .gro via gmx editconf, then runs gmx sasa with 'System' group.
    """
    gro = os.path.join(workdir, f"{tag}.gro")
    xvg = os.path.join(workdir, f"{tag}_sasa.xvg")

    _run([GMX, "editconf", "-f", pdb_path, "-o", gro, "-quiet"], cwd=workdir)

    # gmx sasa with echo "0" to select System group
    proc = subprocess.run(
        [GMX, "sasa", "-f", gro, "-s", gro, "-o", xvg, "-quiet"],
        input="0\n",
        cwd=workdir,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"gmx sasa failed:\n{proc.stderr}")

    return _parse_xvg_total(xvg)


class GromacsSASA:
    def __init__(self, gmx_bin: str = "gmx"):
        global GMX
        GMX = gmx_bin

    def interface_buried_area(self, pdb_path: str, chain_a: str, chain_b: str) -> float:
        """
        Returns buried surface area (nm²) at the A–B interface:
            ΔSASA = SASA(A) + SASA(B) - SASA(AB)
        Uses a temporary directory; safe for parallel calls.
        """
        with tempfile.TemporaryDirectory(prefix="sasa_") as tmpdir:
            parts = _split_chains(pdb_path, [chain_a, chain_b, chain_a + chain_b], tmpdir)
            sasa_a  = _gmx_sasa_on_pdb(parts[chain_a],            tmpdir, f"sasa_A")
            sasa_b  = _gmx_sasa_on_pdb(parts[chain_b],            tmpdir, f"sasa_B")
            sasa_ab = _gmx_sasa_on_pdb(parts[chain_a + chain_b],  tmpdir, f"sasa_AB")
        return max(0.0, sasa_a + sasa_b - sasa_ab)

    def subunit_vs_complex_buried(
        self, pdb_path: str, new_chain: str, complex_chains: list[str]
    ) -> float:
        """
        Buried area when new_chain docks onto the existing partial complex.
        complex_chains: list of chain IDs already in the complex.
        """
        all_chains = "".join(sorted(complex_chains + [new_chain]))
        with tempfile.TemporaryDirectory(prefix="sasa_") as tmpdir:
            parts = _split_chains(
                pdb_path,
                [new_chain, "".join(sorted(complex_chains)), all_chains],
                tmpdir,
            )
            sasa_new     = _gmx_sasa_on_pdb(parts[new_chain],                        tmpdir, "sasa_new")
            sasa_complex = _gmx_sasa_on_pdb(parts["".join(sorted(complex_chains))],  tmpdir, "sasa_complex")
            sasa_all     = _gmx_sasa_on_pdb(parts[all_chains],                       tmpdir, "sasa_all")
        return max(0.0, sasa_new + sasa_complex - sasa_all)
