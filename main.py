# coding=utf-8
from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import numpy as np
import torch
import torch.distributed as dist

from config import get_args
from models.model_cls import DINOSmallCLSFinetune
from utils.logger import setup_logging
from utils.save import count_parameters, load_best_checkpoint
from trainer.train import train_cls_finetune
from dataset.utils import extract_label_from_filename, extract_label_from_path, get_png_files_from_directory, get_image_files_from_directory


def set_seed(args):
    """시드 설정"""
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def setup_cls_finetune_model(args, prefix_saved_mode):
    """CLS Fine-tuning 모델 설정"""
    # 데이터셋 경로 설정
    source_dir = "/DATA_17/VM_team/Dataset/LVM_Datasets/LVM_ori/SDXL_meta_v1/images"  # Source (합성 데이터)
    target_train_dir = "/DATA/yyj/WACV2025-FFTAT/data/fixed_augmented_100"  # Target Train (증강된 실제 데이터)
    target_test_dir = "/DATA/yyj/WACV2025-FFTAT/data/fixed_test_rest"  # Target Test (원본 실제 데이터)
    
    print("🎯 CLS Token Fine-tuning: Domain Adaptation with Fixed Dataset...")
    
    # 1. Source 데이터 수집 (합성 데이터)
    source_paths = get_png_files_from_directory(source_dir)
    source_labels = [extract_label_from_filename(os.path.basename(p)) for p in source_paths]
    print(f"   SDXL_meta_v1 (source): {len(source_paths)} images")
    
    # 2. Target Train 데이터 수집 (증강된 실제 데이터)
    target_train_paths = get_image_files_from_directory(target_train_dir)
    target_train_labels = [extract_label_from_path(p) for p in target_train_paths]
    print(f"   fixed_augmented_100 (target train): {len(target_train_paths)} images")
    
    # 3. Target Test 데이터 수집 (원본 실제 데이터)
    target_test_paths = get_image_files_from_directory(target_test_dir)
    target_test_labels = [extract_label_from_path(p) for p in target_test_paths]
    print(f"   fixed_test_rest (target test): {len(target_test_paths)} images")
    
    # 4. 전체 클래스 목록 및 매핑 생성 (리더님 방식)
    all_labels = source_labels + target_train_labels + target_test_labels
    unique_classes = sorted(list(set(all_labels)))  # 오름차순 소팅
    class_to_idx = {cls: idx for idx, cls in enumerate(unique_classes)}  # 라벨명 → 아이디
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}      # 아이디 → 라벨명
    
    # === 라벨명과 인덱스 매핑 출력 ===
    print("\n[라벨명 ↔ 인덱스 매핑]")
    for cls, idx in class_to_idx.items():
        print(f"라벨명: {cls:15s} → 인덱스: {idx}")
    
    # 5. 라벨을 인덱스로 변환
    source_labels = [class_to_idx[label] for label in source_labels]
    target_train_labels = [class_to_idx.get(label, -1) for label in target_train_labels]
    target_test_labels = [class_to_idx.get(label, -1) for label in target_test_labels]
    
    # 6. 유효한 데이터만 필터링
    filtered_source_paths = []
    filtered_source_labels = []
    for path, label in zip(source_paths, source_labels):
        if label != -1:
            filtered_source_paths.append(path)
            filtered_source_labels.append(label)
    
    filtered_target_train_paths = []
    filtered_target_train_labels = []
    for path, label in zip(target_train_paths, target_train_labels):
        if label != -1:
            filtered_target_train_paths.append(path)
            filtered_target_train_labels.append(label)
    
    filtered_target_test_paths = []
    filtered_target_test_labels = []
    for path, label in zip(target_test_paths, target_test_labels):
        if label != -1:
            filtered_target_test_paths.append(path)
            filtered_target_test_labels.append(label)
    
    print(f"   필터링된 Source: {len(filtered_source_paths)} images")
    print(f"   필터링된 Target Train: {len(filtered_target_train_paths)} images")
    print(f"   필터링된 Target Test: {len(filtered_target_test_paths)} images")
    
    # 7. 유효성 검사
    invalid_source = [i for i, l in enumerate(filtered_source_labels) if l not in idx_to_class]
    invalid_target_train = [i for i, l in enumerate(filtered_target_train_labels) if l not in idx_to_class]
    invalid_target_test = [i for i, l in enumerate(filtered_target_test_labels) if l not in idx_to_class]
    
    if invalid_source:
        print(f"[경고] Source 데이터 중 매핑 안된 라벨 인덱스 존재! (총 {len(invalid_source)}개)")
    else:
        print("[확인] Source 데이터 라벨 매핑 모두 정상.")
    
    if invalid_target_train:
        print(f"[경고] Target Train 데이터 중 매핑 안된 라벨 인덱스 존재! (총 {len(invalid_target_train)}개)")
    else:
        print("[확인] Target Train 데이터 라벨 매핑 모두 정상.")
    
    if invalid_target_test:
        print(f"[경고] Target Test 데이터 중 매핑 안된 라벨 인덱스 존재! (총 {len(invalid_target_test)}개)")
    else:
        print("[확인] Target Test 데이터 라벨 매핑 모두 정상.")
    
    # 8. args 업데이트
    args.num_classes = len(unique_classes)
    
    # 8. CLS Fine-tuning 모델 설정 (옵션 추가)
    use_lora = getattr(args, 'use_lora', False)
    use_cosine = getattr(args, 'use_cosine', False)
    
    model = DINOSmallCLSFinetune(
        num_classes=args.num_classes, 
        img_size=args.img_size,
        use_lora=use_lora,
        use_cosine=use_cosine
    )
    
    # 9. 파라미터 수 출력
    total_params = count_parameters(model)
    print(f"   🎯 Trainable Parameters: {total_params:,}")
    
    # 10. 체크포인트 정리
    best_acc = load_best_checkpoint(args, model, prefix_saved_mode)
    
    return model, {
        'synthetic_paths': filtered_source_paths,
        'synthetic_labels': filtered_source_labels,
        'target_train_paths': filtered_target_train_paths,
        'target_train_labels': filtered_target_train_labels,
        'target_test_paths': filtered_target_test_paths,
        'target_test_labels': filtered_target_test_labels,
        'num_classes': args.num_classes
    }


def main():
    # Parse arguments
    args = get_args()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logger = setup_logging(args)

    set_seed(args)

    # Model & Tokenizer Setup
    model, data_info = setup_cls_finetune_model(args, args.name)
    model.to(args.device)

    # Multi-GPU training
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank,
                                                          find_unused_parameters=True)

    # Dummy cp_mask for compatibility
    cp_mask = torch.eye(325).to(args.device)

    logger.info("***** Running CLS Token Fine-tuning *****")
    logger.info(f"  Model: DINO-Small with CLS Adapter")
    logger.info(f"  LoRA: {args.use_lora}")
    logger.info(f"  Cosine Classification: {args.use_cosine}")
    logger.info(f"  Classes: {data_info['num_classes']}")
    logger.info(f"  Source images: {len(data_info['synthetic_paths'])}")
    logger.info(f"  Target images: {len(data_info['target_train_paths'])}")

    # Start training
    train_cls_finetune(args, model, cp_mask, args.name, data_info)


if __name__ == "__main__":
    main() 