from transformers import AutoTokenizer
from lexichem.datasets import get_dataloaders, get_mol_instruction_dataloaders
import lightning as pl
from lexichem.trainers import T5BaseModel, T5AlignerModel
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from argparse import ArgumentParser, Namespace
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch import seed_everything
import yaml
import os
from huggingface_hub import login
from lexichem.utils import set_nested_attr
from lightning.pytorch.strategies import (
    DDPStrategy,
    FSDPStrategy
)
import wandb
import sys
import time
import optuna
import pandas as pd
from dotenv import load_dotenv
load_dotenv()

# Login Wandb
wandb.login(key=os.getenv("WANDB_API_KEY"))
# Login HuggingFace
login(token=os.getenv("HF_TOKEN"))

class Tee(object):
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()


LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

def print_args(args, indent=0):
    for arg in vars(args):
        val = getattr(args, arg)
        if isinstance(val, Namespace):
            print("  " * indent + f"{arg}:")
            print_args(val, indent + 1)
        else:
            print("  " * indent + f"{arg}: {val}")

def run_training(args, seed, trial=None):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"{args.project.name}_{args.method}_seed{seed}_{timestamp}"
    if trial:
        run_name += f"_trial{trial.number}"
    
    checkpoint_dir = os.path.join(args.output_folder, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)
    log_file_path = os.path.join(LOG_DIR, f"{run_name}.txt")
    
    f = open(log_file_path, 'w')
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(sys.stdout, f)
    sys.stderr = Tee(sys.stderr, f)

    try:
        print(f"\033[93m\n" + "="*50)
        print(f"RUNNING EXPERIMENT WITH SEED: {seed}")
        if trial:
             print(f"OPTUNA TRIAL: {trial.number}")
             print(f"PARAMS: {trial.params}")
        print("="*50 + "\033[0m\n", flush=True)            
        seed_everything(seed)
        tokenizer = AutoTokenizer.from_pretrained(args.t5.pretrained_model_name_or_path)
        if args.method == 'base':
            model = T5BaseModel(args)
        elif args.method == 'aligner':
            model = T5AlignerModel(args)
        else:
            raise Exception('Method name is invalid, please choose one in two: base, aligner')
        model.tokenizer = tokenizer

        if args.dataset_name == 'lpm-24':
            args.dataset_name_or_path = 'Neeze/LPM-24-extend'
        elif args.dataset_name == 'lpm-24-extra':
            args.dataset_name_or_path = 'Neeze/LPM-24-extra-extend'
        elif args.dataset_name == 'chebi-20':
            args.dataset_name_or_path = 'duongttr/chebi-20-new'
        elif args.dataset_name == 'mol-inst':
            args.dataset_name_or_path = 'thienphuprogrammer/mol-instructions-extend'
        else:
            raise Exception('Dataset name is invalid, please choose one in: lpm-24, lpm-24-extra, chebi-20, mol-inst')

        if args.dataset_name == 'mol-inst':
            # Mix train + validation for training; use test split for validation
            train_dataloader = get_mol_instruction_dataloaders(
                args, tokenizer, batch_size=args.batch_size,
                num_workers=args.num_workers, splits='train',
                do_enumeration=args.do_enumeration
            )
            val_dataloader = get_mol_instruction_dataloaders(
                args, tokenizer, batch_size=args.batch_size,
                num_workers=args.num_workers, splits='validation'
            )
        else:
            train_dataloader = get_dataloaders(
                args, tokenizer, batch_size=args.batch_size,
                num_workers=args.num_workers, split='train',
                do_enumeration=args.do_enumeration,
                do_multitask=getattr(args, 'do_multitask', False)
            )
            val_dataloader = get_dataloaders(
                args, tokenizer, batch_size=args.batch_size,
                num_workers=args.num_workers, split='validation',
                do_multitask=getattr(args, 'do_multitask', False)
            )
        args.train_data_len = len(train_dataloader) // args.grad_accum
        args.tokenizer = Namespace()
        args.tokenizer.pad_token_id = tokenizer.pad_token_id

        on_best_eval_loss_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='ckpt_{epoch}_{val_seq2seq_loss:.4f}',
            save_top_k=3,
            verbose=True,
            monitor='val_seq2seq_loss',
            mode='min'
        )

        wandb_logger = WandbLogger(
            log_model=False,
            project=args.project.name,
            name=run_name
        )
        wandb_logger.watch(model, log="all")
        lr_monitor = LearningRateMonitor(logging_interval='step')
        early_stopping = EarlyStopping(
            monitor='val_seq2seq_loss',
            patience=5,
            mode='min'
        )
        callbacks = [on_best_eval_loss_callback, lr_monitor, early_stopping]
        
        if args.strategy == 'ddp':
            strategy = DDPStrategy()
        elif args.strategy == 'fsdp':
            strategy = FSDPStrategy()
        else:
            raise Exception('Strategy name is invalid, please choose one in two: ddp, fsdp')
        
        trainer = pl.Trainer(
            accelerator='cuda' if args.cuda else 'cpu',
            devices=args.num_devices,
            strategy=strategy,
            max_epochs=-1,
            max_steps=args.max_steps,
            val_check_interval=args.val_check_interval,
            check_val_every_n_epoch=None,
            callbacks=callbacks,
            logger=wandb_logger,
            gradient_clip_val=None,
            gradient_clip_algorithm=None,
            accumulate_grad_batches=args.grad_accum,
            precision=args.precision,
            deterministic=True,
            enable_checkpointing=True,
            sync_batchnorm=True if args.strategy == 'ddp' else False,
        )
        trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)            
        return trainer.callback_metrics.get("val_seq2seq_loss", float("inf")).item()

    finally:
        wandb.finish()
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        f.close()

def main(args):
    print("--- Loaded Configuration ---")
    print_args(args)
    print("---------------------------")

    if hasattr(args, 'optuna') and args.optuna.use_optuna:
        def objective(trial):
            # Discrete values for grid search as requested by user
            dropout = trial.suggest_categorical("dropout", [0.05, 0.1, 0.15, 0.2])
            lr_backbone = trial.suggest_categorical("lr_backbone", [5e-4, 2e-4, 5e-5, 2e-5])
            lr_projector = trial.suggest_categorical("lr_projector", [8e-4, 5e-4, 8e-5, 5e-5])

            # Apply to args
            args.t5.dropout = dropout
            args.projector.dropout = dropout # Applying to both as requested for 'dropout'
            args.lr.backbone = lr_backbone
            args.lr.projector = lr_projector

            # Use the first seed for all tuning trials to ensure comparability
            seed = args.seeds[0] if isinstance(args.seeds, list) else args.seeds
            return run_training(args, seed, trial=trial)

        # Search all combinations if GridSampler is needed or just optimize
        # The user requested "search all", suggesting comprehensive search.
        # We'll use GridSampler if common, or just categorical suggestions with enough trials.
        search_space = {
            "dropout": [0.05, 0.1, 0.15, 0.2],
            "lr_backbone": [5e-4, 2e-4, 5e-5, 2e-5],
            "lr_projector": [8e-4, 5e-4, 8e-5, 5e-5]
        }
        study = optuna.create_study(
            direction="minimize", 
            sampler=optuna.samplers.GridSampler(search_space)
        )
        
        n_trials = getattr(args.optuna, 'n_trials', None)
        # If n_trials is set, it might limit the grid search. 
        # For a full grid search (64 trials), we can just set it to len(combinations).
        if n_trials is None:
            n_trials = 4 * 4 * 4
        
        study.optimize(objective, n_trials=n_trials)

        print("\n\033[92m" + "="*50)
        print("OPTUNA TUNING COMPLETE")
        print(f"Best Trial: {study.best_trial.number}")
        print(f"Best Value: {study.best_value}")
        print(f"Best Params: {study.best_params}")
        print("="*50 + "\033[0m\n")

        # Export report table
        df = study.trials_dataframe()
        report_dir = "lexichem/configs/tuner"
        os.makedirs(report_dir, exist_ok=True)
        report_path = os.path.join(report_dir, f"optuna_report_{time.strftime('%Y%m%d_%H%M%S')}.csv")
        df.to_csv(report_path, index=False)
        print(f"Full report exported to: {report_path}")
        
        # Also print markdown table of top trials
        top_df = df.sort_values("value").head(10)
        print("\nTop 10 Experiments:")
        print(top_df[["number", "value", "params_dropout", "params_lr_backbone", "params_lr_projector"]].to_markdown(index=False))

    else:
        # Standard training loop over seeds
        seeds = args.seeds if isinstance(args.seeds, list) else [args.seeds]
        for seed in seeds:
            run_training(args, seed)


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script")
    parser.add_argument('--config', type=str, help="Path to the train configuration file")

    args = parser.parse_args()
    train_config = yaml.safe_load(open(args.config, 'r'))
    for key, value in train_config.items():
        set_nested_attr(args, key, value)
    
    # Flatten trainer config to top level for compatibility with existing code and models
    if hasattr(args, 'trainer'):
        for key, value in vars(args.trainer).items():
            setattr(args, key, value)
    
    main(args)