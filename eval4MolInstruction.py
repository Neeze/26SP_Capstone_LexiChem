"""
eval4MolInstruction.py
======================
Evaluate a trained checkpoint on the **mol-inst** test split with
per-task metrics.

Tasks covered
-------------
- reagent_prediction
- retrosynthesis
- description_guided_molecule_design
- forward_reaction_prediction

For every task the following metrics are computed:
  * Exact-match (canonicalised SMILES InChI comparison)
  * Validity (fraction of parseable SMILES)
  * BLEU (token-level, on canonical SMILES)
  * Levenshtein distance (on canonical SMILES)
  * MACCS / RDKit / Morgan Tanimoto fingerprint similarity

Usage
-----
    python eval4MolInstruction.py --config <path-to-yaml>

The YAML config must contain at least the fields used by the regular
training configs plus ``dataset_name: mol-inst``.
"""

import os
import sys
import yaml
import glob
import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from collections import defaultdict

from transformers import AutoTokenizer
from rdkit import Chem, RDLogger
from rdkit.Chem import MACCSkeys, AllChem
from rdkit import DataStructs
from nltk.translate.bleu_score import corpus_bleu
from rapidfuzz.distance import Levenshtein
import selfies as sf

from lexichem.utils import set_nested_attr
from lexichem.datasets import get_mol_instruction_val_dataloaders_per_task, MOL_INST_TASKS
from lexichem.trainers import T5BaseModel, T5AlignerModel

RDLogger.DisableLog("rdApp.*")
lev = Levenshtein.distance

YELLOW = "\033[93m"
GREEN  = "\033[92m"
RESET  = "\033[0m"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def selfies_to_smiles(selfie):
    try:
        return sf.decoder(selfie)
    except Exception:
        return None

def canonical(smiles: str):
    """Return canonical SMILES or None if unparseable."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=False, canonical=True)
    except Exception:
        return None


def print_args(args, indent=0):
    for arg in vars(args):
        val = getattr(args, arg)
        if isinstance(val, Namespace):
            print("  " * indent + f"{arg}:")
            print_args(val, indent + 1)
        else:
            print("  " * indent + f"{arg}: {val}")


# ---------------------------------------------------------------------------
# Per-task metric computation
# ---------------------------------------------------------------------------

def compute_task_metrics(pred_smiles_list, gt_smiles_list, verbose=False):
    """
    Compute molecule-level metrics for one task.

    Parameters
    ----------
    pred_smiles_list : list[str]   raw predicted strings (SMILES or reaction SMILES)
    gt_smiles_list   : list[str]   ground-truth strings

    Returns
    -------
    dict with keys: validity, exact_match, bleu, levenshtein,
                    maccs_sim, rdk_sim, morgan_sim
    """
    total = len(gt_smiles_list)
    bad   = 0

    valid_pairs = []   # (canonical_gt, canonical_pred, mol_gt, mol_pred)
    refs_bleu   = []
    hyps_bleu   = []
    levs        = []

    for pred_raw, gt_raw in zip(pred_smiles_list, gt_smiles_list):
        # For reaction tasks the output can be "A.B>>C" – canonicalise the
        # *product* part only when possible.  Fall back to full string.
        pred_can = canonical(pred_raw) or canonical(pred_raw.split(">>")[-1])
        gt_can   = canonical(gt_raw)   or canonical(gt_raw.split(">>")[-1])

        if pred_can is None:
            bad += 1
            levs.append(lev(pred_raw, gt_raw))
            refs_bleu.append([[c for c in (gt_can or gt_raw)]])
            hyps_bleu.append([c for c in pred_raw])
            continue

        if gt_can is None:
            levs.append(lev(pred_can, gt_raw))
            refs_bleu.append([[c for c in gt_raw]])
            hyps_bleu.append([c for c in pred_can])
            continue

        mol_pred = Chem.MolFromSmiles(pred_can)
        mol_gt   = Chem.MolFromSmiles(gt_can)

        if mol_pred is None or mol_gt is None:
            bad += 1
            levs.append(lev(pred_can, gt_can))
            refs_bleu.append([[c for c in gt_can]])
            hyps_bleu.append([c for c in pred_can])
            continue

        valid_pairs.append((gt_can, pred_can, mol_gt, mol_pred))
        levs.append(lev(pred_can, gt_can))
        refs_bleu.append([[c for c in gt_can]])
        hyps_bleu.append([c for c in pred_can])

    validity  = 1 - bad / total if total > 0 else 0.0
    bleu      = corpus_bleu(refs_bleu, hyps_bleu) if hyps_bleu else 0.0
    lev_mean  = float(np.mean(levs)) if levs else 0.0

    if not valid_pairs:
        return {
            "validity":    validity,
            "exact_match": 0.0,
            "bleu":        bleu,
            "levenshtein": lev_mean,
            "maccs_sim":   0.0,
            "rdk_sim":     0.0,
            "morgan_sim":  0.0,
            "n_samples":   total,
        }

    num_exact = 0
    MACCS_sims, RDK_sims, morgan_sims = [], [], []

    for gt_can, pred_can, mol_gt, mol_pred in valid_pairs:
        try:
            if Chem.MolToInchi(mol_pred) == Chem.MolToInchi(mol_gt):
                num_exact += 1
        except Exception:
            pass

        try:
            MACCS_sims.append(DataStructs.FingerprintSimilarity(
                MACCSkeys.GenMACCSKeys(mol_gt),
                MACCSkeys.GenMACCSKeys(mol_pred),
                metric=DataStructs.TanimotoSimilarity,
            ))
            RDK_sims.append(DataStructs.FingerprintSimilarity(
                Chem.RDKFingerprint(mol_gt),
                Chem.RDKFingerprint(mol_pred),
                metric=DataStructs.TanimotoSimilarity,
            ))
            morgan_sims.append(DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprint(mol_gt, 2),
                AllChem.GetMorganFingerprint(mol_pred, 2),
            ))
        except Exception:
            pass

    return {
        "validity":    validity,
        "exact_match": num_exact / len(valid_pairs),
        "bleu":        bleu,
        "levenshtein": lev_mean,
        "maccs_sim":   float(np.mean(MACCS_sims)) if MACCS_sims else 0.0,
        "rdk_sim":     float(np.mean(RDK_sims))   if RDK_sims   else 0.0,
        "morgan_sim":  float(np.mean(morgan_sims)) if morgan_sims else 0.0,
        "n_samples":   total,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    print("--- Loaded Configuration ---")
    print_args(args)
    print("---------------------------")

    project_name = args.project.name
    method       = args.method
    seeds        = args.seeds if isinstance(args.seeds, list) else [args.seeds]

    # ── locate checkpoint ─────────────────────────────────────────────────
    found_folders = []
    for seed in seeds:
        pattern  = os.path.join(args.output_folder, f"{project_name}_{method}_seed{seed}*")
        matched  = sorted(glob.glob(pattern))
        matched  = [f for f in matched if os.path.isdir(f)]
        found_folders.extend(matched)

    print(f"\nFound {len(found_folders)} matching folder(s):")
    for i, folder in enumerate(found_folders):
        print(f"  [{i+1}] {folder}")

    if not found_folders:
        print("No folders found."); return

    while True:
        try:
            idx = int(input("\nSelect folder by index: ")) - 1
            if 0 <= idx < len(found_folders):
                selected_folder = found_folders[idx]; break
        except ValueError:
            pass
        print(f"Please enter 1–{len(found_folders)}.")

    ckpt_files = sorted(glob.glob(os.path.join(selected_folder, "*.ckpt")))
    if not ckpt_files:
        ckpt_files = sorted(glob.glob(os.path.join(selected_folder, "**", "*.ckpt"), recursive=True))

    print(f"\nFound {len(ckpt_files)} checkpoint(s):")
    for i, c in enumerate(ckpt_files):
        print(f"  [{i+1}] {os.path.basename(c)}")

    if not ckpt_files:
        print("No checkpoints found."); return

    while True:
        try:
            idx = int(input("\nSelect checkpoint by index: ")) - 1
            if 0 <= idx < len(ckpt_files):
                selected_ckpt = ckpt_files[idx]; break
        except ValueError:
            pass
        print(f"Please enter 1–{len(ckpt_files)}.")

    # ── setup ─────────────────────────────────────────────────────────────
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.t5.pretrained_model_name_or_path)

    args.dataset_name_or_path = "thienphuprogrammer/mol-instructions-extend"
    args.tokenizer            = Namespace()
    args.tokenizer.pad_token_id = tokenizer.pad_token_id

    if args.method == "base":
        model = T5BaseModel(args)
    elif args.method == "aligner":
        model = T5AlignerModel(args)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    print(YELLOW + f"\nLoading checkpoint: {selected_ckpt}" + RESET)
    model.load_state_dict(
        torch.load(selected_ckpt, map_location=device)["state_dict"],
        strict=False,
    )
    model.to(device)
    model.eval()

    # ── per-task dataloaders (test split) ─────────────────────────────────
    task_loaders = get_mol_instruction_val_dataloaders_per_task(
        args, tokenizer, batch_size=args.batch_size, num_workers=args.num_workers
    )

    # ── inference + metrics ───────────────────────────────────────────────
    all_results   = {}
    all_rows      = []

    for task in MOL_INST_TASKS:
        loader = task_loaders[task]
        if len(loader.dataset) == 0:
            print(f"[SKIP] {task} — no samples in test split.")
            continue

        print(YELLOW + f"\n{'='*60}" + RESET)
        print(YELLOW + f"Evaluating task: {task}  ({len(loader.dataset)} samples)" + RESET)
        print(YELLOW + f"{'='*60}" + RESET)

        pred_list = []
        gt_list   = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=task):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)

                pred_selfies_list = model.generate_molecule(batch, tokenizer=tokenizer)
                gt_selfies_list   = batch['selfies']
                captions          = batch['caption']
                task_labels       = batch['task']

                for pred_selfie, gt_selfie, caption, t in zip(
                    pred_selfies_list, gt_selfies_list, captions, task_labels
                ):
                    pred_smiles = selfies_to_smiles(pred_selfie)
                    gt_smiles   = selfies_to_smiles(gt_selfie)

                    pred_list.append(pred_smiles or '')
                    gt_list.append(gt_smiles or '')

                    all_rows.append({
                        'task':         t,
                        'caption':      caption,
                        'gt_selfies':   gt_selfie,
                        'gt_smiles':    gt_smiles or '',
                        'pred_selfies': pred_selfie,
                        'pred_smiles':  pred_smiles or '',
                    })

        metrics = compute_task_metrics(pred_list, gt_list, verbose=True)
        all_results[task] = metrics

        print(f"\n  Results for [{task}]:")
        for k, v in metrics.items():
            if k == "n_samples":
                print(f"    {k:15}: {v}")
            else:
                print(f"    {k:15}: {v:.4f}")

    # ── save outputs ─────────────────────────────────────────────────────
    out_dir = os.path.join(project_name, "mol_inst_eval")
    os.makedirs(out_dir, exist_ok=True)

    json_path = os.path.join(out_dir, "metrics_per_task.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=4)
    print(GREEN + f"\nMetrics saved → {json_path}" + RESET)

    csv_path  = os.path.join(out_dir, "predictions.csv")
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(GREEN + f"Predictions saved → {csv_path}" + RESET)

    # ── summary table ─────────────────────────────────────────────────────
    print(YELLOW + "\n" + "=" * 80)
    print("SUMMARY — per-task metrics")
    print("=" * 80 + RESET)
    metric_keys = ["validity", "exact_match", "bleu", "levenshtein",
                   "maccs_sim", "rdk_sim", "morgan_sim", "n_samples"]
    header = f"{'task':<42}" + "".join(f"{k:>14}" for k in metric_keys)
    print(header)
    print("-" * len(header))
    for task, m in all_results.items():
        row = f"{task:<42}"
        for k in metric_keys:
            v = m.get(k, 0)
            row += f"{v:>14.4f}" if isinstance(v, float) else f"{v:>14}"
        print(row)
    print("=" * 80)


if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate mol-inst model per task")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to YAML config file (same format as train.py)")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    for key, value in cfg.items():
        set_nested_attr(args, key, value)
    if hasattr(args, "trainer"):
        for key, value in vars(args.trainer).items():
            setattr(args, key, value)

    # HuggingFace / dotenv login (optional – graceful skip if no token)
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from huggingface_hub import login
        import os
        token = os.getenv("HF_TOKEN")
        if token:
            login(token=token)
    except Exception:
        pass

    main(args)
