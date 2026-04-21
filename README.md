<h1 align="center">LEXICHEM</h1>
<p align="center"><a href="#abstract">📝 Paper</a> | <a href="#3-benchmark-datasets">🤗 Benchmark datasets</a> | <a href="">🚩 Checkpoints</a> | <a href="">⚙️ Application</a> | <a href="#citation">📚 Cite our paper!</a></p>

The official implementation of manuscript **LEXICHEM: COEVOLVING ALIGNING LANGUAGE AND MOLECULES IN A SHARED LATENT SPACE**

## Abstract
> Translating natural language descriptions of molecules into exact molecular structures remains a formidable bottleneck in computational chemistry, revealing a profound modality gap between linguistic abstraction and topological rigor that requires deep representation alignment to resolve. Sequence-based generative models readily yield syntactically viable strings, yet they routinely fail to preserve the precise semantic intent governing human-authored descriptions. This functional deficit reveals a profound modality gap between linguistic abstraction and topological rigor, which superficial sequence-to-sequence translation cannot resolve without strict deep representation alignment. We introduce LexiChem: a unified generative framework engineered to synchronize textual and chemical representations within a continuous latent topology. The architecture utilizes parallel encoding streams for natural language prompts and molecular inputs mapped to robust Self-Referencing Embedded Strings (SELFIES) formulations. To traverse the cross-modal divide without precipitating dimensional collapse, we deploy Variance-Invariance-Covariance Regularization (VICReg) coupled with asymmetric gradient interruption. This dual protocol safeguards the sophisticated reasoning circuitry of the pretrained language backbone while precisely steering the molecular encoder toward established semantic manifolds. Conditioned upon this integrated feature space, an autoregressive decoder synthesizes targeted chemical structures. We optimize contrastive alignment and generative objectives jointly to maximize cross-modal fidelity. An integrated multimodal data pipeline, featuring rigorous RDKit-mediated validity filtering and curation, strictly governs the underlying training distribution. Extensive evaluation across string-level parameters and domain-specific topological metrics, including Tanimoto fingerprint similarities, conclusively validates the methodology. Our findings confirm that this alignment-centric architecture delivers robust generative precision alongside structurally flawless molecular geometries. Ultimately, LexiChem establishes that explicitly unifying the latent domains of text and chemistry yields highly faithful, semantically governed molecular design.


## How to use

### 1. Environment preparation
After cloning the repo, run the following command to install required packages:

```zsh
conda create -n LEXICHEM python=3.10
conda activate LEXICHEM
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
The evaluation script is interactive. It will ask you to select the experiment folder and the checkpoint to evaluate.

#### Evaluate on LPM-24
```zsh
python eval.py --config lexichem/configs/aligner/config_lpm24_train.yaml
```

#### Evaluate on CheBI-20
```zsh
python eval.py --config lexichem/configs/aligner/config_chebi20_train.yaml
```

#### Push to hub

```zsh
python push_to_hub.py --model_name biot5-plus-base-sft \
                      --ckpt_path path/to/ckpt \
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
