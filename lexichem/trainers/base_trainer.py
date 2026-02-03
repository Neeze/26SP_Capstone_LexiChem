import lightning as pl
from lexichem.backbones import T5ForConditionalGeneration
import torch
from torch import optim
import math
from argparse import Namespace


class T5BaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Initialize multimodal text-based model
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            args.t5.pretrained_model_name_or_path
        )
        self.t5_model.gradient_checkpointing_enable()

        # Inference
        self.generation_config = {
            "max_length": 512,
            "num_beams": 4, 
            "do_sample": False,
            "early_stopping": True,
            "no_repeat_ngram_size": 0,
            "length_penalty": 1.0,
        }
        
    def resize_token_embeddings(self, len_embeddings):
        self.t5_model.resize_token_embeddings(len_embeddings)
        
    def __prepare_inputs(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        
        return input_ids, attention_mask, labels

    def prepare_inputs(self, batch):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        return input_ids, attention_mask, labels
    
    def forward(self, input_ids, 
                attention_mask, 
                labels=None,):
        labels[labels == self.args.tokenizer.pad_token_id] = -100
        
        output = self.t5_model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            output_attentions=True
        )
        
        return output.loss, output.logits
    
    def forward2(self, input_ids, 
                attention_mask, 
                labels=None,):
        labels[labels == self.args.tokenizer.pad_token_id] = -100
        
        output = self.t5_model.forward_train(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels,
            output_attentions=True # ADD HERE
        )
        
        return output
    
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = self.__prepare_inputs(batch)
        loss, _ = self(input_ids, attention_mask, labels)
        self.log('train/total_loss', loss, prog_bar=True, logger=True)
        self.log("train/seq2seq_loss", loss, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = self.__prepare_inputs(batch)
        loss, _ = self(input_ids, attention_mask, labels)
        self.log('val/total_loss', loss, prog_bar=True, logger=True)
        self.log('val/seq2seq_loss', loss, prog_bar=True, logger=True)
        self.log('val_total_loss', loss, prog_bar=False, logger=False)
        return loss
    
    def configure_optimizers(self):
        lr_backbone = getattr(self.args.lr, "backbone", None)
        if lr_backbone is None:
            raise ValueError("lr_backbone must be specified")
             
        param_groups = [
            {"params": self.t5_model.parameters(), "lr": lr_backbone, "name": "backbone"},
        ]

        optimizer = torch.optim.AdamW(
            param_groups,
            lr=3e-4,
            betas=(0.9, 0.999),
            eps=1e-10,
            weight_decay=0.01
        )
        assert self.trainer is not None
        max_iter = self.trainer.estimated_stepping_batches
        print(f"[Lightning estimated steps = {max_iter}]", flush=True)
        scheduler = {
        "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=[self.args.lr], 
                total_steps=max_iter,
                pct_start=self.args.warmup_ratio, 
                div_factor=25.0,
                final_div_factor=1e4,   
                anneal_strategy='cos'
            ),
            "name": "learning_rate",
            "interval": "step",
            "frequency": 1,
        }
        return [optimizer], [scheduler]
    
    def generate_molecule(self, inputs, tokenizer, **kwargs):
        input_ids, attention_mask, _ = self.__prepare_inputs(inputs)
        
        generation_kwargs = {
            "decoder_start_token_id": tokenizer.pad_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
            **self.generation_config,
            **kwargs
        }

        outputs = self.t5_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_kwargs
        )
        decoded_sequences = tokenizer.batch_decode(outputs, skip_special_tokens=False)
        
        clean_outputs = []
        for s in decoded_sequences:
            s = s.replace('<unk>', '').replace('<pad>', '').replace('</s>', '').strip()
            clean_outputs.append(s)
            
        return clean_outputs
        
    @staticmethod
    def cosine_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            progress = current_step - warmup_steps
            progress /= max(1, training_steps - warmup_steps)
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)