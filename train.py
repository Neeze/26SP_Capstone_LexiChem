from transformers import AutoTokenizer
from lexichem.datasets import get_dataloaders
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

def main(args):
    print("--- Loaded Configuration ---")
    print_args(args)
    print("---------------------------")

    # Ensure seeds is a list
    seeds = args.seeds if isinstance(args.seeds, list) else [args.seeds]

    for seed in seeds:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        run_name = f"{args.project.name}_{args.method}_seed{seed}_{timestamp}"
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
            else:
                raise Exception('Dataset name is invalid, please choose one in three: lpm-24, lpm-24-extra, chebi-20')
            train_dataloader = get_dataloaders(
                args, tokenizer, batch_size=args.batch_size,
                num_workers=args.num_workers, split='train',
                do_enumeration=True
            )
            val_dataloader = get_dataloaders(
                args, tokenizer, batch_size=args.batch_size,
                num_workers=args.num_workers, split='validation'
            )
            args.train_data_len = len(train_dataloader) // args.grad_accum
            args.tokenizer = Namespace()
            args.tokenizer.pad_token_id = tokenizer.pad_token_id

            on_best_eval_loss_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                filename='ckpt_{epoch}_{val_total_loss:.4f}',
                save_top_k=3,
                verbose=True,
                monitor='val_total_loss',
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
                monitor='val_total_loss',
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
                max_epochs=args.epochs,
                callbacks=callbacks,
                logger=wandb_logger,
                gradient_clip_val=1.0 if args.strategy != 'fsdp' else None,
                gradient_clip_algorithm="norm" if args.strategy != 'fsdp' else None,
                accumulate_grad_batches=args.grad_accum,
                precision=args.precision,
                deterministic=True,
            )
            trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)            
            
        finally:
            wandb.finish()
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            f.close()


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