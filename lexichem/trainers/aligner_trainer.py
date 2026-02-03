import lightning as pl
from transformers import AutoModel
from lexichem.backbones import (
    T5ForConditionalGeneration,
    Projector
)
import torch.nn as nn
import torch
from torch import optim
import math
from typing import Optional, Tuple

import math
import warnings
from transformers.modeling_outputs import (
    BaseModelOutput,
    Seq2SeqLMOutput
)
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

@dataclass
class T5WithLossOutput:
    """
    Extension of Seq2SeqLMOutput with language modeling loss and contrastive loss.
    """
    loss: Optional[torch.FloatTensor] = None
    seq2seq_loss: Optional[torch.FloatTensor] = None
    contrastive_loss: Optional[torch.FloatTensor] = None

class T5AlignerModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.t5_model = T5ForConditionalGeneration.from_pretrained(
            args.t5.pretrained_model_name_or_path
        ).to(device)
        self.molecule_proj = Projector(
            input_dim=self.t5_model.config.d_model,
            output_dim=args.projector.latent_dim,
            hidden_dim=args.projector.hidden_dim,
            num_layers=args.projector.num_layers,
            dropout=args.projector.dropout
        ).to(device)
        self.language_proj = Projector(
            input_dim=self.t5_model.config.d_model,
            output_dim=args.projector.latent_dim,
            hidden_dim=args.projector.hidden_dim,
            num_layers=args.projector.num_layers,
            dropout=args.projector.dropout
        ).to(device)
        self.config = self.t5_model.config
        self.seq2seq_lambda = args.loss.seq2seq_lambda
        self.alignment_lambda = args.loss.alignment_lambda
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
    
    def forward(self, 
                input_ids: Optional[torch.LongTensor] = None,
                attention_mask: Optional[torch.FloatTensor] = None,
                decoder_input_ids: Optional[torch.LongTensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                head_mask: Optional[torch.FloatTensor] = None,
                decoder_head_mask: Optional[torch.FloatTensor] = None,
                cross_attn_head_mask: Optional[torch.Tensor] = None,
                encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
                inputs_embeds: Optional[torch.FloatTensor] = None,
                decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
                labels: Optional[torch.LongTensor] = None,
                use_cache: Optional[bool] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None
        ):
        # Prepare decoder inputs for next token prediction (Clone labels before masking)
        decoder_input_ids_pass1 = None
        decoder_attention_mask_pass1 = None
        if labels is not None:
             decoder_input_ids_pass1 = labels.clone()
             decoder_attention_mask_pass1 = (decoder_input_ids_pass1 != self.args.tokenizer.pad_token_id).long()

        labels[labels == self.args.tokenizer.pad_token_id] = -100
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(FutureWarning)
                decoder_head_mask = head_mask
        
        if input_ids is not None:
            batch_size = input_ids.shape[0]
            device = input_ids.device
        elif inputs_embeds is not None:
            batch_size = inputs_embeds.shape[0]
            device = inputs_embeds.device
        else:
            if encoder_outputs is not None:
                enc_last = encoder_outputs[0] if not isinstance(encoder_outputs, BaseModelOutput) else encoder_outputs.last_hidden_state
                batch_size = enc_last.shape[0]
                device = enc_last.device
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds for forward_train")
        
        # Prepare decoder inputs for next token prediction
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = self.t5_model._shift_right(labels)
            
        # ========================================================================
        # ENCODER: Process input text
        # Output: encoder_hidden_states for cross-attention and contrastive learning
        # ========================================================================
        if encoder_outputs is None:
            encoder_outputs = self.t5_model.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=True,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        encoder_hidden_states = encoder_outputs.last_hidden_state

        if attention_mask is not None:
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(encoder_hidden_states.size()).float()
            sum_hidden = torch.sum(encoder_hidden_states * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            encoder_embeddings = sum_hidden / sum_mask  # Shape: (batch_size, hidden_size)
        else:
            encoder_embeddings = encoder_hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_size)
        
        # ========================================================================
        # DECODER PASS 1: Self-attention only (NO cross-attention)
        # Purpose: Generate decoder embeddings for contrastive learning
        # ========================================================================
        decoder_outputs_pass1 = self.t5_model.decoder(
            input_ids=decoder_input_ids_pass1 if decoder_input_ids_pass1 is not None else decoder_input_ids,
            attention_mask=decoder_attention_mask_pass1 if decoder_attention_mask_pass1 is not None else decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=None,
            use_cache=False,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
            enable_causal=False # Disable causale mask
        )
        decoder_hidden_states_pass1 = decoder_outputs_pass1.last_hidden_state
        
        if decoder_attention_mask is not None:
            mask_expanded = decoder_attention_mask.unsqueeze(-1).expand(decoder_hidden_states_pass1.size()).float()
            sum_hidden = torch.sum(decoder_hidden_states_pass1 * mask_expanded, dim=1)
            sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
            decoder_embeddings = sum_hidden / sum_mask
        else:
            decoder_embeddings = decoder_hidden_states_pass1.mean(dim=1)
        
        # ========================================================================
        # DECODER PASS 2: Full decoder (self-attention + cross-attention)
        # ========================================================================
        decoder_outputs_pass2 = self.t5_model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,  # WITH cross-attention to encoder
            encoder_attention_mask=attention_mask,
            past_key_values=past_key_values,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            enable_causal=True, # with causal mask
        )
        
        sequence_output = decoder_outputs_pass2.last_hidden_state
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.t5_model.model_dim**-0.5)
        lm_logits = self.t5_model.lm_head(sequence_output)
        
        # ========================================================================
        # SEQ2SEQ LOSS: Next token prediction (cross-entropy)
        # Standard language modeling loss
        # ========================================================================
        seq2seq_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            labels = labels.to(lm_logits.device)
            seq2seq_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            seq2seq_loss = seq2seq_loss * self.seq2seq_lambda
        
        # ========================================================================
        # CONTRASTIVE LOSS: Encoder-Decoder embedding alignment
        # Align encoder embeddings (from input) with decoder embeddings
        # ========================================================================
        
        contrastive_loss = torch.tensor(0.0, device=device)
        lang_emb = self.language_proj(encoder_embeddings.detach())
        mol_emb = self.molecule_proj(decoder_embeddings)  
        vicreg_val = self.vicreg_loss(lang_emb, mol_emb)
        contrastive_loss = self.alignment_lambda * vicreg_val

        # ========================================================================
        # TOTAL LOSS: Seq2Seq + Contrastive
        # ========================================================================
        total_loss = None
        if seq2seq_loss is not None:
            total_loss = seq2seq_loss + contrastive_loss
            
            
        if not return_dict:
            output = (lm_logits,) + decoder_outputs_pass2[1:] + (encoder_outputs,)
            return ((total_loss,) + output) if total_loss is not None else output
        
        return T5WithLossOutput(
            loss=total_loss,
            seq2seq_loss=seq2seq_loss,
            contrastive_loss=contrastive_loss
        )
        
    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels= self.__prepare_inputs(batch)
        losses = self(input_ids, attention_mask, labels=labels)
        loss = losses.loss
        seq2seq_loss = losses.seq2seq_loss
        contrastive_loss = losses.contrastive_loss
        self.log('train/total_loss', loss, prog_bar=True, logger=True)
        self.log("train/seq2seq_loss", seq2seq_loss, prog_bar=False, logger=True)
        self.log("train/contrastive_loss", contrastive_loss, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = self.__prepare_inputs(batch)
        losses = self(input_ids, attention_mask, labels=labels)
        loss = losses.loss
        seq2seq_loss = losses.seq2seq_loss
        contrastive_loss = losses.contrastive_loss
        self.log('val/total_loss', loss, prog_bar=True, logger=True)
        self.log('val_total_loss', loss, prog_bar=False, logger=False)
        self.log("val/seq2seq_loss", seq2seq_loss, prog_bar=False, logger=True)
        self.log("val_seq2seq_loss", seq2seq_loss, prog_bar=False, logger=False)
        self.log("val/contrastive_loss", contrastive_loss, prog_bar=False, logger=True)

        return loss

    def on_before_optimizer_step(self, optimizer):
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        self.log("train/grad_norm", total_norm, prog_bar=True, logger=True)
            
    def configure_optimizers(self):
        lr_backbone = getattr(self.args.lr, "backbone", None)
        lr_proj = getattr(self.args.lr, "projector", None)
        if lr_backbone is None or lr_proj is None:
            raise ValueError("lr_backbone and lr_proj must be specified")

        param_groups = [
            {"params": self.t5_model.parameters(), "lr": lr_backbone, "name": "backbone"},
            {"params": self.language_proj.parameters(), "lr": lr_proj, "name": "language_proj"},
            {"params": self.molecule_proj.parameters(), "lr": lr_proj, "name": "molecule_proj"},
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
                max_lr=[lr_backbone, lr_proj, lr_proj], 
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

    @staticmethod
    def constant_scheduler(optimizer, training_steps, warmup_steps):
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return current_step / max(1, warmup_steps)
            return 1.0

        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ================================================================
    # Contrastive Loss Function
    # ================================================================
    
    def clip_contrastive_pair_loss(self, x, y, logit_scale):
        B = x.size(0)
        logits = torch.matmul(x, y.t()) * logit_scale
        target = torch.arange(B, device=x.device)
        return F.cross_entropy(logits, target)
    
    def negative_cosine_similarity_loss(self, x, y):
        return -F.cosine_similarity(x, y, dim=-1).mean()
    
    def vicreg_loss(self, x, y):
        lam, mu, nu = 25.0, 25.0, 1.0
        B, D = x.shape

        # 1. Invariance: MSE Loss
        sim_loss = F.mse_loss(x, y)

        # 2. Variance: Hinge loss std >= 1
        std_x = torch.sqrt(x.var(dim=0) + 1e-4)
        std_y = torch.sqrt(y.var(dim=0) + 1e-4)
        std_loss = F.relu(1 - std_x).mean() + F.relu(1 - std_y).mean()

        # 3. Covariance
        x = x - x.mean(dim=0)
        y = y - y.mean(dim=0)
        cov_x = (x.T @ x) / (B - 1)
        cov_y = (y.T @ y) / (B - 1)
        cov_loss = ((cov_x.pow(2).sum() - cov_x.diag().pow(2).sum()) + 
                    (cov_y.pow(2).sum() - cov_y.diag().pow(2).sum())) / D

        return lam * sim_loss + mu * std_loss + nu * cov_loss