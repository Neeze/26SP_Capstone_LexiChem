import os
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, concatenate_datasets
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

# ---------------------------------------------------------------------------
# Mol-Instructions dataset  (thienphuprogrammer/mol-instructions-extend)
# Tasks: reagent_prediction | retrosynthesis |
#        description_guided_molecule_design | forward_reaction_prediction
# ---------------------------------------------------------------------------

MOL_INST_TASKS = [
    'reagent_prediction',
    'retrosynthesis',
    'description_guided_molecule_design',
    'forward_reaction_prediction',
]




class MolInstructionDataset(Dataset):
    """Dataset wrapper for ``thienphuprogrammer/mol-instructions-extend``.

    Columns: instruction, input, output, metadata (contains 'task' key).

    Parameters
    ----------
    splits : str or list[str]
        One or more HuggingFace split names.  When a list is provided the
        resulting datasets are concatenated (mix train+validation for training).
    task_filter : str or None
        If given, only keep samples whose ``task`` column equals this value.
    """

    def __init__(
        self,
        args,
        tokenizer,
        dataset_name_or_path='thienphuprogrammer/mol-instructions-extend',
        splits='train',
        task_filter=None,
        input_max_length=512,
        output_max_length=512,
        add_instruction=True,
        do_enumeration=False,
    ):
        super().__init__()
        num_cores = os.cpu_count()

        if isinstance(splits, str):
            splits = [splits]

        loaded = []
        for sp in splits:
            ds = load_dataset(
                dataset_name_or_path,
                split=sp,
                use_auth_token=True,
                num_proc=num_cores,
            )
            loaded.append(ds)

        self.dataset = concatenate_datasets(loaded) if len(loaded) > 1 else loaded[0]

        # Optionally restrict to a single task for per-task evaluation
        if task_filter is not None:
            self.dataset = self.dataset.filter(
                lambda s: s['task'] == task_filter,
                num_proc=num_cores,
            )

        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.output_max_length = output_max_length
        self.add_instruction = add_instruction
        self.do_enumeration = do_enumeration

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]

        sample_caption = sample['caption']
        sample_selfies  = sample['selfies']
        task = sample['task']

        if self.do_enumeration:
            try:
                sample_smiles = sf.decoder(sample_selfies)
                mol = Chem.MolFromSmiles(sample_smiles)
                if mol is not None:
                    random_smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
                    sample_selfies = sf.encoder(random_smiles)
            except Exception:
                pass

        if self.add_instruction:
            model_input = (
                f"Task: Translate description to SELFIES representation.\n"
                f"Input: {sample_caption}\n"
                f"Output:"
            )
        else:
            model_input = sample_caption

        encoded_input = self.tokenizer(
            model_input,
            add_special_tokens=True,
            max_length=self.input_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        encoded_output = self.tokenizer(
            sample_selfies,
            add_special_tokens=True,
            max_length=self.output_max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoded_input['input_ids'].flatten(),
            'attention_mask': encoded_input['attention_mask'].flatten(),
            'labels': encoded_output['input_ids'].flatten(),
            'selfies': sample_selfies,
            'caption': sample_caption,
            'task': task,
        }


def get_mol_instruction_dataloaders(
    args,
    tokenizer,
    batch_size=8,
    num_workers=4,
    splits='train',
    task_filter=None,
    add_instruction=True,
    do_enumeration=False,
):
    """Return a DataLoader for the mol-inst dataset.

    Pass ``splits=['train', 'validation']`` to mix both splits for training.
    Pass ``task_filter='retrosynthesis'`` to restrict to one task.
    """
    dataset = MolInstructionDataset(
        args,
        tokenizer=tokenizer,
        dataset_name_or_path=args.dataset_name_or_path,
        splits=splits,
        task_filter=task_filter,
        input_max_length=512,
        output_max_length=512,
        add_instruction=add_instruction,
        do_enumeration=do_enumeration,
    )
    shuffle = splits == 'train' or (isinstance(splits, list) and 'train' in splits)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=True,
    )


def get_mol_instruction_val_dataloaders_per_task(args, tokenizer, batch_size=8, num_workers=4):
    """Return ``{task_name: DataLoader}`` for the **test** split, one per task.

    Used by ``eval4MolInstruction.py``.
    """
    loaders = {}
    for task in MOL_INST_TASKS:
        loaders[task] = get_mol_instruction_dataloaders(
            args,
            tokenizer,
            batch_size=batch_size,
            num_workers=num_workers,
            splits='test',
            task_filter=task,
        )
    return loaders


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