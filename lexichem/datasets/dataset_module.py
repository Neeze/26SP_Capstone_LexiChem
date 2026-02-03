import os
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from torchvision import transforms
from transformers import AutoTokenizer
import rdkit.Chem as Chem
import selfies as sf
from torch.nn.utils.rnn import pad_sequence

class DynamicPadCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        out_batch = {}
        for key in batch[0].keys():
            if batch[0][key] is None:
                continue
            if isinstance(batch[0][key], torch.Tensor):
                tensors = [sample[key] for sample in batch]
                pad_val = self.tokenizer.pad_token_id
                out_batch[key] = pad_sequence(tensors, batch_first=True, padding_value=pad_val)
            else:
                out_batch[key] = [sample[key] for sample in batch]
        
        return out_batch

class MoleculeGeneration(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 dataset_name_or_path='duongttr/LPM-24-extend', 
                 split='train',
                 add_instruction=True,
                 do_enumeration=False):
        super().__init__()
        num_cores = os.cpu_count()
        self.dataset = load_dataset(dataset_name_or_path, split=split, use_auth_token=True, num_proc=num_cores)
        
        if 'LPM-24' in dataset_name_or_path:
            self.dataset = self.dataset.filter(lambda sample: sample['selfies'] != '', num_proc=num_cores)
            
        self.is_lpm_24 = 'LPM-24' in dataset_name_or_path
        self.add_instruction = add_instruction
        self.do_enumeration = do_enumeration
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index, add_padding=True):
        sample = self.dataset[index]
        
        if self.is_lpm_24:
            sample_selfies = sample['selfies']
            sample_caption = sample['caption']
            sample_smiles = sample.get('smiles', '')
        else:
            sample_selfies = sample['SELFIES']
            sample_caption = sample['description']
            sample_smiles = sample.get('smiles', '') # Fallback if key doesn't exist, though typically it might be 'SMILES' or similar depending on dataset

        if self.do_enumeration and sample_smiles:
            try:
                mol = Chem.MolFromSmiles(sample_smiles)
                if mol is not None:
                    random_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                    sample_selfies = sf.encoder(random_smiles)
            except Exception as e:
                # Fallback to original selfies if enumeration fails
                pass

        if self.add_instruction:
            model_input = (
                f"Task: Translate description to SELFIES representation.\n"
                f"Input: {sample_caption}\n"
                f"Output:"
            )
        else:
            model_input = sample_caption
        
        input_data = self.tokenizer(
            model_input,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        output_data = self.tokenizer(
            sample_selfies,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = input_data['input_ids'].flatten()
        attention_mask = input_data['attention_mask'].flatten()
        labels = output_data['input_ids'].flatten()

        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'selfies': sample_selfies,
            'caption': sample_caption,
        }

class MoleculeGeneration_InferLPM24(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 dataset_name_or_path='duongttr/LPM-24-eval-caption', 
                 split='train',
                 add_instruction=True):
        super().__init__()
        self.dataset = load_dataset(dataset_name_or_path, split=split, use_auth_token=True)
        
        if 'LPM-24' in dataset_name_or_path:
            self.dataset = self.dataset.filter(lambda sample: sample['selfies'] != '')
            
        self.is_lpm_24 = 'LPM-24' in dataset_name_or_path
        self.add_instruction = add_instruction
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        sample = self.dataset[index]
        
        
        sample_selfies = sample['selfies']
        sample_caption = sample['caption']

        if self.add_instruction:
            model_input = (
                f"Task: Translate description to SELFIES representation.\n"
                f"Input: {sample_caption}\n"
                f"Output:"
            )
        else:
            model_input = sample_caption
        
        
        input_data = self.tokenizer(
            model_input,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        output_data = self.tokenizer(
            sample_selfies,
            add_special_tokens=True,
            padding=False,
            truncation=False,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        input_ids = input_data['input_ids'].flatten()
        attention_mask = input_data['attention_mask'].flatten()
        labels = output_data['input_ids'].flatten()
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': attention_mask,
            'selfies': sample_selfies,
            'caption': sample_caption
        }

def get_dataloaders(args, tokenizer, batch_size=8, num_workers=4, split='train', add_instruction=True, do_enumeration=False):
    dataset = MoleculeGeneration(
        args,
        tokenizer=tokenizer,
        dataset_name_or_path=args.dataset_name_or_path,
        split=split,
        add_instruction=add_instruction,
        do_enumeration=do_enumeration
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == 'train'),
        pin_memory=True,
        collate_fn=DynamicPadCollator(tokenizer)
    )

def get_dataloaders_inferlpm24(args, tokenizer, batch_size=8, num_workers=4, split='train', add_instruction=True):
    dataset = MoleculeGeneration_InferLPM24(
        args,
        tokenizer=tokenizer,
        dataset_name_or_path=args.dataset_name_or_path,
        split=split,
        add_instruction=add_instruction
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == 'train'),
        pin_memory=True,
        collate_fn=DynamicPadCollator(tokenizer)
    )