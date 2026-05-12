# UTRGen

UTRGen is a unified framework for full-spectrum design of mRNA 5' UTRs based on the GPT-2 architecture. It integrates sequence generation, multi-property prediction, and function-guided optimization within a single system.

## Installation

```bash
conda create -n utrgen python=3.10
conda activate utrgen
pip install -r requirements.txt
```

## Project Structure

```
utrgen_release/
├── pretrain/           # Stage I: Autoregressive pre-training
├── downstream/         # Stage II: Downstream task fine-tuning
│   ├── te_el/          #   TE and EL prediction (10-fold CV)
│   ├── mrl/            #   MRL prediction (ranking split)
│   └── ires/           #   IRES classification (10-fold CV)
├── rl/                 # Stage III: Reinforcement learning (GRPO)
├── generate/           # Sequence generation
├── ckpts/              # Model checkpoints (download separately)
└── data/               # Data directory (download separately)
```

Each module is self-contained. `cd` into any module directory and run the scripts directly.

## Usage

### 1. Pre-training

Pre-trains a GPT-2 model on large-scale 5' UTR sequences using next-token prediction. The model learns nucleotide-level sequence patterns and long-range dependencies, providing a general-purpose backbone for downstream tasks. Training uses AdamW optimizer with cosine annealing learning rate schedule.

```bash
cd pretrain
python pretrain.py \
  --vocab_file ../vocab.txt \
  --train_files ../data/pretrain/utr5_train_nr_0.8_c0.8.fasta \
  --test_files ../data/pretrain/utr5_val_nr_0.8_c0.8.fasta \
  --n_embd 384 --n_layer 8 --n_head 16 --n_ctx 256 \
  --batch_size 512 --lr 3e-3 --epoch 100 --n_gpu 2
```

| Key Parameters | Description |
|----------------|-------------|
| `--vocab_file` | Path to `vocab.txt` (required) |
| `--train_files` / `--test_files` | Training/validation FASTA files (required) |
| `--n_embd` / `--n_layer` / `--n_head` | Model architecture (default: 384/8/16) |
| `--n_ctx` | Max sequence length (default: 256) |
| `--batch_size` / `--lr` / `--epoch` | Training hyperparameters |
| `--n_gpu` | Number of GPUs (default: 1) |
| `--checkpoint_dir` | Output directory for checkpoints |
| `--resume_from_checkpoint` | Resume training from a checkpoint |

### 2. Downstream Fine-tuning

Attaches lightweight prediction heads (1D-CNN or MLP) to the pre-trained backbone and fine-tunes on downstream tasks with Huber loss. All scripts support loading pre-trained weights and optional encoder freezing.

#### TE/EL Tasks

Predicts Translation Efficiency (TE) or Expression Level (EL) from endogenous 5' UTR sequences. Uses 10-fold cross-validation to evaluate performance across multiple cell lines (HEK, Muscle, PC3). The `--label_column` parameter selects the target: `te_log` for TE, `rnaseq_log` for EL.

```bash
cd downstream/te_el
python downstream_10fold_cv_train.py \
  --data_path ../../data/te_el/HEK_sequence.csv \
  --label_column te_log --data_column utr \
  --vocab_file ../../vocab.txt \
  --pretrain_ckpt path/to/pretrained.ckpt \
  --batch_size 64 --epochs 200 --lr 2e-4
```

| Key Parameters | Description |
|----------------|-------------|
| `--data_path` | CSV file with sequences and labels (required) |
| `--label_column` | `te_log` for TE, `rnaseq_log` for EL (required) |
| `--data_column` | Sequence column name, e.g. `utr` (required) |
| `--pretrain_ckpt` | Pre-trained checkpoint path |
| `--regressor` | Head type: 1=CNN, 2=ResNet, 3=MLP (default: 1) |
| `--n_splits` | Number of CV folds (default: 10) |

#### MRL Task

Predicts Mean Ribosome Load (MRL) from synthetic 5' UTR libraries. Uses separate train/test CSV files (ranking-based split) rather than cross-validation. Supports multiple GSM datasets with different nucleotide modification conditions.

```bash
cd downstream/mrl
python downstream_mrl_ranking_split_finetuned.py \
  --train_path ../../data/mrl/4.1_train_data_GSM3130435_egfp_unmod_1.csv \
  --test_path ../../data/mrl/4.1_test_data_GSM3130435_egfp_unmod_1.csv \
  --label_column label --data_column utr \
  --vocab_file ../../vocab.txt \
  --pretrain_ckpt path/to/pretrained.ckpt
```

| Key Parameters | Description |
|----------------|-------------|
| `--train_path` / `--test_path` | Training/test CSV paths (required) |
| `--label_column` | Label column name (required) |
| `--data_column` | Sequence column name (required) |
| `--pretrain_ckpt` | Pre-trained checkpoint path |
| `--regressor` | Head type: 1=CNN, 2=ResNet, 3=MLP (default: 1) |

#### IRES Task

Fine-tunes the pre-trained model as a binary classifier to identify Internal Ribosome Entry Sites (IRES) within 5' UTRs. Uses a CNN-based classification head and 10-fold cross-validation. The `--train_ratio` parameter supports few-shot learning experiments by subsampling the training set.

```bash
cd downstream/ires
python train_ires_cls_cv.py \
  --data_path ../../data/ires/ires_merged_clean.csv \
  --sequence_column sequence --label_column label \
  --vocab_file ../../vocab.txt \
  --pretrained path/to/pretrained.ckpt \
  --epochs 5 --batch_size 16 --lr 1e-5
```

| Key Parameters | Description |
|----------------|-------------|
| `--data_path` | IRES CSV data path (required) |
| `--sequence_column` | Sequence column (default: `sequence`) |
| `--label_column` | Binary label column (default: `label`) |
| `--pretrained` | Pre-trained checkpoint path |
| `--train_ratio` | Fraction of training data (default: 1.0) |

### 3. Reinforcement Learning Fine-tuning (GRPO)

Optimizes the pre-trained generator toward specific functional objectives (high TE or EL) using Group Relative Policy Optimization (GRPO). The policy model generates candidate sequences from single-nucleotide prompts (A/T/C/G), and a composite reward function scores them using:
- **Functional reward**: TE/EL predictor (frozen downstream oracle models)
- **GC penalty**: Constrains GC content to [0.4, 0.6] range
- **MFE reward**: Penalizes excessive secondary structure stability

A KL divergence term regularizes the policy against a frozen reference model to maintain sequence diversity and biological plausibility.

```bash
cd rl
python train.py \
  --vocab_file ../vocab.txt \
  --pretrained_ckpt path/to/pretrained.ckpt \
  --reward_func te gc mfe \
  --reward_func_weights 1.0 1.5 1.0 \
  --te_ckpt_paths path/to/oracle_1.ckpt path/to/oracle_2.ckpt path/to/oracle_3.ckpt \
  --min_gen_length 150 --max_gen_length 200 \
  --epochs 2000 --batch_size 32 --lr 1e-4 \
  --num_generations 8 --kl_beta 0.4 \
  --task_name hek_te_grpo
```

| Key Parameters | Description |
|----------------|-------------|
| `--pretrained_ckpt` | Pre-trained Lightning checkpoint (required) |
| `--reward_func` | Reward functions: `te`, `mrl`, `ires`, `gc`, `gc_mse`, `gc_tmse`, `mfe` |
| `--reward_func_weights` | Weight for each reward function |
| `--te_ckpt_paths` | TE oracle checkpoint(s) for reward scoring |
| `--num_generations` | Sequences per prompt for group advantage (G in GRPO, default: 4) |
| `--kl_beta` | KL penalty weight against reference model (default: 0.1) |
| `--min/max_gen_length` | Generation length range |
| `--task_name` | Task name for logging and checkpointing (required) |

### 4. Sequence Generation

Generates de novo 5' UTR sequences from a trained model (pre-trained or RL-finetuned). Uses A/T/C/G as single-nucleotide prompts and generates autoregressively with configurable sampling parameters (top-k, top-p, temperature, repetition penalty). Outputs unique sequences in FASTA format.

```bash
cd generate
python generate.py \
  --ckp path/to/model.ckpt \
  --vocab_path ../vocab.txt \
  --num_sample 10000 \
  --min_gen_length 100 --max_gen_length 150 \
  --top_k 50 --top_p 0.9 --temperature 1.0 \
  --output_dir output/generated
```

| Key Parameters | Description |
|----------------|-------------|
| `--ckp` | Lightning checkpoint path (required) |
| `--vocab_path` | Vocab file path |
| `--num_sample` | Target number of unique sequences (default: 3000) |
| `--min/max_gen_length` | Generation length range |
| `--top_k` / `--top_p` | Sampling truncation parameters |
| `--temperature` | Sampling temperature (>1 more random, <1 more conservative) |
| `--repetition_penalty` | Repetition penalty (>1 reduces repetitive fragments) |
| `--output_dir` | Output directory for FASTA files |

## Data & Checkpoints

The data used in this project can be downloaded from [link](https://drive.google.com/drive/folders/1E4TYXN48wdDru_-ETVxu2UWYSEKvO0wr?usp=sharing) and placed in the data/ directory after being unzipped. The ckpts files for this project can be downloaded from [link](https://drive.google.com/drive/folders/1E4TYXN48wdDru_-ETVxu2UWYSEKvO0wr?usp=sharing) and placed in the ckpts/ directory after being unzipped.

## Model Architecture

- **Backbone**: GPT-2 decoder (8 layers, 16 heads, 384 hidden dim, max 256 tokens)
- **Tokenizer**: Character-level nucleotide tokenizer (A, T, C, G + special tokens)
- **Downstream heads**: 1D-CNN, MLP, or ResNet18-style for regression; CNN for classification
- **RL**: GRPO with composite reward (TE/EL predictor + GC penalty + MFE constraint)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
