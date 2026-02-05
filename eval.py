import os
import yaml
import torch
import glob
import json
import pandas as pd
import selfies as sf
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer
from lexichem.utils import set_nested_attr
from lexichem.datasets import get_dataloaders
from lexichem.trainers import T5BaseModel, T5AlignerModel
from lexichem.metrics.mol_translation_selfies_metrics import Mol_translation_selfies

YELLOW = "\033[93m"
GREEN = "\033[92m"
RESET = "\033[0m"

def selfies_to_smiles(selfie):
    try:
        smiles = sf.decoder(selfie)
        return smiles
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

def main(args):
    print("--- Loaded Configuration ---")
    print_args(args)
    print("---------------------------")
    project_name = args.project.name
    method = args.method
    seeds = args.seeds if isinstance(args.seeds, list) else [args.seeds]
    print(f"Searching for folders for seeds: {seeds}")

    found_folders = []
    for seed in seeds:
        # Search for folders matching the pattern: {project_name}_{method}_seed{seed}*
        search_pattern = os.path.join(args.output_folder, f"{project_name}_{method}_seed{seed}*")
        matched_folders = sorted(glob.glob(search_pattern))
        matched_folders = [f for f in matched_folders if os.path.isdir(f)]
        found_folders.extend(matched_folders)

    print(f"Found {len(found_folders)} matching folders:")
    for i, folder in enumerate(found_folders):
        print(f" [{i+1}] {folder}")

    if not found_folders:
        print("No folders found matching the criteria.")
        return

    while True:
        try:
            choice = input("\nSelect a folder by index (e.g., 1): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(found_folders):
                selected_folder = found_folders[choice_idx]
                print(f"Selected: {selected_folder}")
                break
            else:
                print(f"Invalid index. Please enter a number between 1 and {len(found_folders)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    base_ckpt_pattern = os.path.join(selected_folder, "*.ckpt")
    ckpt_files = sorted(glob.glob(base_ckpt_pattern))
    
    if not ckpt_files:
        subdir_pattern = os.path.join(selected_folder, "**", "*.ckpt")
        ckpt_files = sorted(glob.glob(subdir_pattern, recursive=True))

    print(f"\nFound {len(ckpt_files)} .ckpt files in {selected_folder}:")
    for i, ckpt in enumerate(ckpt_files):
        print(f" [{i+1}] {os.path.basename(ckpt)}")

    if not ckpt_files:
        print("No .ckpt files found.")
        return

    while True:
        try:
            choice = input("\nSelect a checkpoint by index (e.g., 1): ")
            choice_idx = int(choice) - 1
            if 0 <= choice_idx < len(ckpt_files):
                selected_ckpt = ckpt_files[choice_idx]
                print(f"Selected checkpoint: {selected_ckpt}")
                break
            else:
                print(f"Invalid index. Please enter a number between 1 and {len(ckpt_files)}.")
        except ValueError:
            print("Invalid input. Please enter a number.")
            
    output_csv = args.project.name + '/' + args.dataset_name + '_eval.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.t5.pretrained_model_name_or_path)
    
    if args.dataset_name == 'lpm-24' or args.dataset_name == 'lpm-24-extra':
        args.dataset_name_or_path = 'Neeze/LPM-24-eval-extend'
        split = 'validation'
    elif args.dataset_name == 'chebi-20':
        args.dataset_name_or_path = 'duongttr/chebi-20-new'
        split = 'test'
    else:
        raise Exception('Dataset name is invalid, please choose one in two: lpm-24, chebi-20')
    
    args.tokenizer = Namespace()
    args.tokenizer.pad_token_id = tokenizer.pad_token_id

    if args.method == 'base':
        model = T5BaseModel(args)
    elif args.method == 'aligner':
        model = T5AlignerModel(args)
    else:
        raise Exception('Method name is invalid, please choose one in two: base, aligner')
    
    print(YELLOW + f"Loading checkpoint from {selected_ckpt}..." + RESET, flush=True)
    model.load_state_dict(
        torch.load(selected_ckpt, map_location=device)['state_dict'], strict=False
    )

    model.to(device)
    model.eval()

    dataloader = get_dataloaders(args, tokenizer, batch_size=args.batch_size, num_workers=args.num_workers, split=split)
    
    molecule_metric = Mol_translation_selfies()
    
    predictions = []
    references = []
    
    print(YELLOW + "Starting evaluation..." + RESET, flush=True)
    with torch.no_grad():
        for batch in tqdm(dataloader):
            # Move relevant batch tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            
            # Generate molecule SELFIES
            pred_selfies = model.generate_molecule(batch, tokenizer=tokenizer)
            
            # Ground truth from batch
            gt_selfies = batch['selfies']
            captions = batch['caption']
            
            for pred_selfie, gt_selfie, caption in zip(pred_selfies, gt_selfies, captions):
                pred_smiles = selfies_to_smiles(pred_selfie)
                gt_smiles = selfies_to_smiles(gt_selfie)
                
                predictions.append([pred_smiles if pred_smiles else "", pred_selfie])
                references.append([caption, gt_smiles if gt_smiles else "", gt_selfie])

    print(YELLOW + "Computing metrics..." + RESET, flush=True)
    results = molecule_metric._compute(
        predictions=predictions,
        references=references,
        tsv_path=output_csv.replace('.csv', '.tsv'),
        verbose=True
    )

    print(YELLOW + "\n" + "="*30, flush=True)
    print("EVALUATION RESULTS", flush=True)
    print("="*30, flush=True)
    for key, value in results.items():
        print(f"{key:15}: {value:.4f}", flush=True)
    print("="*30 + RESET, flush=True)

    # Save results to JSON
    output_json = output_csv.replace('.csv', '.json')
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=4)

    print(GREEN + f"Metrics saved to {output_json}" + RESET, flush=True)

    # Save results to CSV
    df_results = pd.DataFrame({
        'caption': [ref[0] for ref in references],
        'gt_selfie': [ref[2] for ref in references],
        'gt_smiles': [ref[1] for ref in references],
        'pred_selfie': [pred[1] for pred in predictions],
        'pred_smiles': [pred[0] for pred in predictions]
    })
    
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_results.to_csv(output_csv, index=False)
    print(GREEN + f"Results saved to {output_csv}" + RESET, flush=True)
    
if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate script")
    parser.add_argument('--config', type=str, help="Path to the configuration file")
    args = parser.parse_args()
    train_config = yaml.safe_load(open(args.config, 'r'))
    for key, value in train_config.items():
        set_nested_attr(args, key, value)
    if hasattr(args, 'trainer'):
        for key, value in vars(args.trainer).items():
            setattr(args, key, value)
    main(args)