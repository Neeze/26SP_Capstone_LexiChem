import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import rdkit.Chem as Chem
import selfies as sf

class MoleculeGeneration(Dataset):
    def __init__(self,
                 args,
                 tokenizer,
                 dataset_name_or_path='duongttr/LPM-24-extend', 
                 split='train',
                 input_max_length=512,
                 output_max_length=512,
                 add_instruction=True,
                 do_enumeration=False,):
        super().__init__()
        num_cores = os.cpu_count()
        self.dataset = load_dataset(dataset_name_or_path, split=split, use_auth_token=True, num_proc=num_cores)
        
        # preprocessing data
        if 'LPM-24' in dataset_name_or_path:
            self.dataset = self.dataset.filter(lambda sample: sample['selfies'] != '', num_proc=num_cores)
            
        self.is_lpm_24 = 'LPM-24' in dataset_name_or_path
        self.add_instruction = add_instruction
        self.do_enumeration = do_enumeration
            
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length

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
                pass

        if self.add_instruction:
            model_input = (
                f"Task: Translate description to SELFIES representation.\n"
                f"Input: {sample_caption}\n"
                f"Output:"
            )
        else:
            model_input = sample_caption
        
        if add_padding:
            input = self.tokenizer(
                model_input,
                add_special_tokens=True,
                max_length=self.input_max_length,
                padding = 'max_length',
                truncation = True,
                return_attention_mask = True,
                return_tensors='pt'
            )
            
            output = self.tokenizer(
                sample_selfies,
                add_special_tokens=True,
                max_length=self.output_max_length,
                padding = 'max_length',
                truncation = True,
                return_attention_mask = True,
                return_tensors='pt'
            )
        else:
            input = self.tokenizer(
                model_input,
                add_special_tokens=True,
                return_attention_mask = True,
                return_tensors='pt'
            )
            
            output = self.tokenizer(
                sample_selfies,
                add_special_tokens=True,
                return_attention_mask = True,
                return_tensors='pt'
            )
        
        input_ids = input['input_ids'].flatten()
        attention_mask = input['attention_mask'].flatten()
        labels = output['input_ids'].flatten()

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
                 input_max_length=512,
                 output_max_length=512,
                 add_instruction=True):
        super().__init__()
        self.dataset = load_dataset(dataset_name_or_path, split=split, use_auth_token=True)
        
        # preprocessing data
        if 'LPM-24' in dataset_name_or_path:
            self.dataset = self.dataset.filter(lambda sample: sample['selfies'] != '') # remove invalid selfies
            
        self.is_lpm_24 = 'LPM-24' in dataset_name_or_path
        self.add_instruction = add_instruction
            
        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
    
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
        
        
        input = self.tokenizer(
            model_input,
            add_special_tokens=True,
            max_length=self.input_max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors='pt'
        )
        
        output = self.tokenizer(
            sample_selfies,
            add_special_tokens=True,
            max_length=self.output_max_length,
            padding = 'max_length',
            truncation = True,
            return_attention_mask = True,
            return_tensors='pt'
        )
        
        input_ids = input['input_ids'].flatten()
        attention_mask = input['attention_mask'].flatten()
        labels = output['input_ids'].flatten()
        
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
        input_max_length=512,
        output_max_length=512,
        add_instruction=add_instruction,
        do_enumeration=do_enumeration
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == 'train'),
        pin_memory=True
    )

def get_dataloaders_inferlpm24(args, tokenizer, batch_size=8, num_workers=4, split='train', add_instruction=True):
    dataset = MoleculeGeneration(
        args,
        tokenizer=tokenizer,
        dataset_name_or_path=args.dataset_name_or_path,
        split=split,
        input_max_length=512,
        output_max_length=512,
        add_instruction=add_instruction
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=(split == 'train'),
        pin_memory=True
    )