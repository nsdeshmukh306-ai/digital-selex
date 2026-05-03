#!/usr/bin/env python3
"""
Script 01 – Fetch TCA metabolite structures from PubChem.

Downloads SDF from PubChem REST API for each target in config/targets.yaml,
converts to 3D using RDKit ETKDG, and saves as SDF + SMILES.

PubChem REST API: https://pubchem.ncbi.nlm.nih.gov/docs/pug-rest
"""

import argparse
import os
import sys
import time
import logging

import requests
import yaml

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

from utils.docking_utils import smiles_to_sdf

log = logging.getLogger(__name__)


def fetch_sdf_from_pubchem(cid: int, out_path: str, retries: int = 3) -> bool:
    """Download 3D SDF from PubChem for a given CID."""
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{cid}/SDF"
    params = {"record_type": "3d"}

    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            if resp.status_code == 200 and "V2000" in resp.text:
                with open(out_path, "w") as fh:
                    fh.write(resp.text)
                log.info(f"  Downloaded PubChem 3D SDF for CID {cid}")
                return True
            elif resp.status_code == 404:
                log.warning(f"  PubChem has no 3D SDF for CID {cid}; will use RDKit")
                return False
            else:
                log.warning(f"  PubChem HTTP {resp.status_code} (attempt {attempt+1})")
        except requests.exceptions.RequestException as e:
            log.warning(f"  Network error: {e} (attempt {attempt+1})")
        time.sleep(2 ** attempt)  # exponential back-off

    return False


def main():
    parser = argparse.ArgumentParser(description="Fetch TCA metabolite structures")
    parser.add_argument("--config",  default=os.path.join(ROOT_DIR, "config", "config.yaml"))
    parser.add_argument("--targets", default=os.path.join(ROOT_DIR, "config", "targets.yaml"))
    parser.add_argument("--outdir",  default=os.path.join(ROOT_DIR, "data", "ligands"))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    with open(args.targets) as fh:
        targets = yaml.safe_load(fh)["targets"]

    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"], logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.join(ROOT_DIR, cfg["logging"]["file"]), mode="a"),
        ],
    )
    os.makedirs(os.path.dirname(os.path.join(ROOT_DIR, cfg["logging"]["file"])), exist_ok=True)

    summary = []
    for t in targets:
        name = t["name"]
        cid  = t["pubchem_cid"]
        smiles = t["smiles"]
        log.info(f"Processing {name} (CID={cid})")

        sdf_path = os.path.join(args.outdir, f"{name}.sdf")

        # Try PubChem 3D SDF first
        if not fetch_sdf_from_pubchem(cid, sdf_path):
            log.info(f"  Falling back to RDKit 3D generation from SMILES")
            ok = smiles_to_sdf(smiles, sdf_path, name=name)
            if not ok:
                log.error(f"  FAILED to generate 3D structure for {name}")
                continue

        # Also save SMILES for reference
        smiles_path = os.path.join(args.outdir, f"{name}.smi")
        with open(smiles_path, "w") as fh:
            fh.write(f"{smiles}\t{name}\n")

        summary.append({"name": name, "cid": cid, "sdf": sdf_path})
        log.info(f"  Saved: {sdf_path}")

        time.sleep(0.3)  # PubChem rate limit: 5 req/s

    log.info(f"\nDone. {len(summary)}/{len(targets)} metabolites prepared.")
    return 0 if len(summary) == len(targets) else 1


if __name__ == "__main__":
    sys.exit(main())
