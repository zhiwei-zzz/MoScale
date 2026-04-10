# Next-Scale Autoregressive Models for Text-to-Motion Generation

<a href="https://arxiv.org/abs/2604.03799"><img src="https://img.shields.io/badge/arXiv-2604.03799-b31b1b.svg" alt="arXiv"></a>
<a href="https://zhiwei-zzz.github.io/MoScale"><img src="https://img.shields.io/badge/Project-Website-green" alt="Project Page"></a>


## 🛠️ Installation

### 1. Clone the repository

```bash
git clone git@github.com:zhiwei-zzz/MoScale.git
cd MoScale
```

### 2. Create the conda environment

```bash
conda env create -f environment.yml
conda activate moscale
```

## 📦 Dataset

This project uses [HumanML3D](https://github.com/EricGuo5513/HumanML3D). Follow the instructions in that repository to prepare the dataset, then set `data.root_dir` in all `config/*.yaml` files to your local HumanML3D path.

The dataset directory should have the following structure:

```
dataset/
└── HumanML3D/
    ├── new_joint_vecs/   # 263-dim motion features (.npy per clip)
    ├── new_joints/       # 3D joint positions (.npy per clip)
    ├── texts/            # text annotations
    ├── train.txt
    ├── val.txt
    ├── test.txt
    ├── Mean.npy
    └── ...
```

Set `data.root_dir` in all `config/*.yaml` files to the `HumanML3D/` subdirectory, e.g. `/your/path/dataset/HumanML3D`.

### 📥 Checkpoints

Download the motion evaluator, GloVe word vectors, and checkpoints:

```bash
bash prepare/download_evaluators.sh
bash prepare/download_glove.sh
bash prepare/download_model.sh
```

## 📊 Evaluation

### Evaluate HRVQVAE reconstruction quality

```bash
python eval_hrvqvae.py
```

Configuration: `config/eval_hrvqvae.yaml`. Set `data.root_dir` to your dataset path.

### Evaluate MoScale generation quality

```bash
python eval_moscale.py
```

Configuration: `config/eval_moscale.yaml`. Set `data.root_dir` to your dataset path.

## 🚀 Training

Training is a two-stage process. Update `data.root_dir` in the config files before running.

### Stage 1: Train HRVQVAE tokenizer

```bash
python train_hrvqvae.py
```

Configuration: `config/train_hrvqvae.yaml`


### Stage 2: Train MoScale Transformer

Set `vq_name` and `vq_ckpt` in `config/train_moscale.yaml` to point to your trained HRVQVAE checkpoint, then run:

```bash
python train_moscale.py
```

Configuration: `config/train_moscale.yaml`


## 📝 Todo

- [x] Release inference code
- [x] Release checkpoints
- [x] Release training code
- [ ] Release editing code


## 🙏 Acknowledgements

The code is built upon open-source projects including [MoMask++](https://github.com/snap-research/SnapMoGen) and [VAR](https://github.com/FoundationVision/VAR). We thank the authors for their helpful code.

## 📜 Citation

If you find this work useful, please cite:

```bibtex
@article{zheng2026moscale,
  title={Next-Scale Autoregressive Models for Text-to-Motion Generation},
  author={Zheng, Zhiwei and Jin, Shibo and Liu, Lingjie and Zhao, Mingmin},
  journal={arXiv preprint arXiv:2604.03799},
  year={2026}
}
```
