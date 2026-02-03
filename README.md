<h1 align="center">ChemAligner-T5</h1>
<p align="center"><a href="#abstract">📝 Paper</a> | <a href="#3-benchmark-datasets">🤗 Benchmark datasets</a> | <a href="https://huggingface.co/collections/Neeze/chemaligner-t5">🚩 Checkpoints</a> | <a href="https://huggingface.co/collections/Neeze/chemaligner-t5">⚙️ Application</a> | <a href="#citation">📚 Cite our paper!</a></p>

The official implementation of manuscript **"ChemAligner-T5: A Unified Text-to-Molecule Model via Representation Alignment"**

## Abstract
> Molecular generation from natural language descriptions is becoming an important approach for guided molecule design, as it allows researchers to express chemical objectives directly in textual form. However, string representations such as SMILES and SELFIES reside in embedding spaces that differ significantly from natural language, creating a mismatch that prevents generative models from accurately capturing the intended chemical semantics. This gap raises the question of whether a shared representation space can be constructed in which textual descriptions and molecular strings converge in a controlled manner. Motivated by this gap, we introduce ChemAligner-T5, a BioT5+ base model enhanced with a contrastive learning mechanism to directly align textual and molecular representations. On the L+M-24 test set, ChemAligner-T5 achieves a BLEU score of 69.77\% and a Levenshtein distance of 31.28\%, outperforming MolT5-base and Meditron on both metrics. Visual analysis shows that the model successfully reproduces the structural scaffold and key functional groups of the target molecule. These results highlight the importance of text–molecule representation alignment for the Text2Mol task and strengthen the potential of language models as direct interfaces for molecule design and drug discovery guided by natural-language descriptions.


## How to use

### 1. Environment preparation
After cloning the repo, run the following command to install required packages:

```zsh
conda create -n ChemAligner python=3.10
conda activate ChemAligner
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
pip install python-dotenv optuna huggingface_hub
pip install -r requirements.txt
```

Create `.env` file contains:

```
WANDB_API_KEY=''
HF_TOKEN=''
```

### 2. Pretrained models
- BioT5+: [HuggingFace](https://huggingface.co/collections/QizhiPei/biot5)

### 3. Benchmark datasets
- LPM-24: [HuggingFace](https://huggingface.co/datasets/Neeze/LPM-24-extra-extend)
- LPM-24-Extra: [HuggingFace](https://huggingface.co/datasets/Neeze/LPM-24-extra-extend)
- CheBI-20: [HuggingFace](https://huggingface.co/datasets/duongttr/chebi-20-new)

### 3. Training model

#### LPM-24 dataset:

**Base Training:**
```bash
python train.py --config lexichem/configs/base/config_lpm24_train.yaml
```

**Aligner Training:**
```bash
python train.py --config lexichem/configs/aligner/config_lpm24_train.yaml
```

#### LPM-24-Extra dataset:

**Base Training:**
```bash
python train.py --config lexichem/configs/base/config_lpm24_extra_train.yaml
```

**Aligner Training:**
```bash
python train.py --config lexichem/configs/aligner/config_lpm24_extra_train.yaml
```


#### CheBI-20 dataset:

**Base Training:**
```bash
python train.py --config lexichem/configs/base/config_chebi20_train.yaml
```

**Aligner Training:**
```bash
python train.py --config lexichem/configs/aligner/config_chebi20_train.yaml
```

### 4. Evaluating model
#### Evaluate on LPM-24
```zsh

```

#### Evaluate on CheBI-20
```zsh

```

#### Push to hub

```zsh
python push_to_hub.py --model_name biot5-plus-base-sft \
                      --ckpt_path path/to/ckpt \
                      --hf_token <your_hf_token>
```

### 5. Application
#### Start the app
You can interact with the model through a user interface by running the following command:

```zsh
python app.py
```

## Citation
If you are interested in my paper, please cite:
```
@inproceedings{Phan2026ChemAlignerT5,
  title     = {ChemAligner-T5: A Unified Text-to-Molecule Model via Representation Alignment},
  author    = {Nam, Van Hai Phan and
               Khoa, Minh Nguyen and
               Phu, Nguyen Ngoc Thien and
               Nguyen, Doan Hieu Nguyen and
               Tri, Minh Pham and
               Duc, Dang Ngoc Minh},
  booktitle = {Proceedings of the 2nd International Conference on Computational Intelligence in Engineering Science},
  year      = {2026},
  month     = apr,
  address   = {Nha Trang, Khanh Hoa, Vietnam}
}
```
