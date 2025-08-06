# CLS Token Fine-tuning for Domain Adaptation

This project implements CLS token fine-tuning for domain adaptation using DINO-Small with LoRA and cosine classification.

## Project Structure

```
project/
│
├── main.py                         # 실험 진입점 (디노+로라+코사인 조합 실행)
├── config.py                       # argparse + 실험 하이퍼파라미터
├── requirements.txt                # 프로젝트 의존성 목록
├── run_commands.sh                # 실행 명령어 모음집
├── test_imports.py                # 모듈 import 테스트
│
├── models/
│   ├── __init__.py
│   ├── model_cls.py                # DINOSmallCLSFinetune 등 모델 정의
│   ├── classifier.py               # CosineClassifier, LinearClassifier
│   └── adapter.py                  # CLSAdapter, LoRA block
│
├── dataset/
│   ├── __init__.py
│   ├── loader.py                   # FilenameBasedDataset + transform
│   └── utils.py                    # 라벨 추출, 파일 경로 수집
│
├── losses/
│   ├── __init__.py
│   └── contrastive.py              # contrastive_loss 함수 정의
│
├── trainer/
│   ├── __init__.py
│   ├── train.py                    # train_cls_finetune
│   └── validate.py                 # valid_cls_finetune
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py                  # simple_accuracy, AverageMeter
│   ├── save.py                     # save_model, count_parameters
│   ├── scheduler.py                # WarmupLinearSchedule, WarmupCosineSchedule
│   └── logger.py                   # 로깅 세팅
│
└── README.md                       # 프로젝트 문서
```

## Features

- **DINO-Small Backbone**: Pre-trained Vision Transformer for feature extraction
- **CLS Token Fine-tuning**: Domain-specific adaptation of CLS tokens
- **LoRA Integration**: Parameter-efficient fine-tuning with Low-Rank Adaptation
- **Cosine Classification**: Better generalization with cosine similarity
- **Contrastive Learning**: Domain adaptation through contrastive loss
- **Modular Design**: Clean, organized code structure

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Test Imports
```bash
python test_imports.py
```

### 3. Run Training
```bash
python main.py --name cls_lora_cosine --dataset leader_filename_47class --use_lora --use_cosine --lora_rank 8 --lora_alpha 16
```

## Usage

### Basic Command

```bash
python main.py --name cls_lora_cosine --dataset leader_filename_47class --use_lora --use_cosine --lora_rank 8 --lora_alpha 16
```

### Key Arguments

- `--name`: Experiment name for logging and checkpointing
- `--use_lora`: Enable LoRA for parameter-efficient fine-tuning
- `--use_cosine`: Use cosine classification instead of linear
- `--lora_rank`: LoRA rank (default: 16)
- `--lora_alpha`: LoRA alpha scaling factor (default: 32)
- `--learning_rate`: Learning rate (default: 3e-2)
- `--num_steps`: Total training steps (default: 5000)
- `--train_batch_size`: Training batch size (default: 4)
- `--eval_batch_size`: Evaluation batch size (default: 8)

### Command Collection

모든 실행 명령어를 확인하려면:
```bash
./run_commands.sh
```

### Dataset Paths

The code expects the following dataset structure:

- **Source**: `/DATA_17/VM_team/Dataset/LVM_Datasets/LVM_ori/SDXL_meta_v1/images` (Synthetic data)
- **Target Train**: `/DATA/yyj/WACV2025-FFTAT/data/fixed_augmented_100` (Augmented real data)
- **Target Test**: `/DATA/yyj/WACV2025-FFTAT/data/fixed_test_rest` (Original real data)

## Model Architecture

### DINOSmallCLSFinetune

- **Backbone**: DINO-Small (frozen by default)
- **CLS Adapter**: Domain-specific CLS token adaptation
- **LoRA**: Optional low-rank adaptation for efficiency
- **Classifier**: Cosine or Linear classification head
- **Projection**: Contrastive learning projection head

### Key Components

1. **CLSAdapter**: Transforms CLS tokens for domain adaptation
2. **CosineClassifier**: Uses cosine similarity for better generalization
3. **LoRALayer**: Parameter-efficient fine-tuning
4. **Contrastive Loss**: Domain adaptation through similarity learning

## Training Process

1. **Data Loading**: Loads synthetic (source) and real (target) data
2. **Label Mapping**: Creates unified class mapping across domains
3. **Model Setup**: Initializes DINO-Small with CLS adapter
4. **Training Loop**: 
   - Forward pass with source and target data
   - Classification loss for both domains
   - Contrastive loss for domain adaptation
   - Gradient clipping and optimization
5. **Validation**: Regular evaluation on target test set
6. **Checkpointing**: Saves best model based on validation accuracy

## Output

- **Checkpoints**: Saved in `./output/{dataset}/`
- **Logs**: TensorBoard logs in `./runs/{name}/`
- **Best Model**: Automatically saves best performing model

## Dependencies

- PyTorch >= 2.1.2
- torchvision >= 0.16.2
- timm >= 1.0.19
- tensorboard >= 2.8.0
- tqdm >= 4.50.2
- opencv-python >= 4.9.0.80
- matplotlib >= 3.8.2
- seaborn >= 0.13.1
- scikit-learn >= 1.7.0
- networkx >= 3.1
- Pillow >= 11.3.0
- numpy >= 1.26.4

## Notes

- The backbone is frozen by default for efficiency
- Only CLS adapter, classifier, and projection heads are trained
- LoRA can be enabled for additional parameter efficiency
- Cosine classification provides better generalization than linear
- Contrastive learning helps with domain adaptation

## Repository

GitHub: [https://github.com/yyj-xiilab/xshift-yyong.git](https://github.com/yyj-xiilab/xshift-yyong.git) 