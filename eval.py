import yaml
import torch
import pandas as pd
import selfies as sf
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from transformers import AutoTokenizer
from lexichem.utils import set_nested_attr
from lexichem.datasets import get_dataloaders
from lexichem.trainers import T5BaseModel, T5AlignerModel
from lexichem.metrics.mol_translation_selfies_metrics import Mol_translation_selfies

def selfies_to_smiles(selfie):
    try:
        smiles = sf.decoder(selfie)
        return smiles
    except Exception:
        return None

def main(args):
    output_csv = args.project.name + '/' + args.dataset_name + '_eval.csv'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.t5.pretrained_model_name_or_path)
    
    if args.dataset_name == 'lpm-24':
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
    
    print(f"Loading checkpoint from {args.checkpoint_path}...", flush=True)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    
    model.to(device)
    model.eval()

    dataloader = get_dataloaders(args, tokenizer, batch_size=args.batch_size, num_workers=args.num_workers, split=split)
    
    molecule_metric = Mol_translation_selfies()
    
    predictions = []
    references = []
    
    print("Starting evaluation...")
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

    print("Computing metrics...", flush=True)
    results = molecule_metric._compute(
        predictions=predictions,
        references=references,
        tsv_path=output_csv.replace('.csv', '.tsv'),
        verbose=True
    )

    print("\n" + "="*30, flush=True)
    print("EVALUATION RESULTS", flush=True)
    print("="*30, flush=True)
    for key, value in results.items():
        print(f"{key:15}: {value:.4f}", flush=True)
    print("="*30, flush=True)

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
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    
    cmd_args = parser.parse_args()
    
    model_config = yaml.safe_load(open(args.model_config, 'r'))
    for key, value in model_config.items():
        set_nested_attr(args, key, value)
        
    main(args)