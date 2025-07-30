#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
47-Class Dataset Creation Script
SDXL Synthetic → Baidu Filtered Real Domain Adaptation

데이터셋 경로:
- Source (Synthetic): /DATA_17/VM_team/Dataset/LVM_Datasets/LVM_ori/SDXL_meta_v1/images
- Target (Real): /DATA_17/DATASET/baidu_filtered_2
"""

import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

def scan_directory_classes(directory):
    """디렉토리 내 클래스(폴더) 스캔"""
    if not os.path.exists(directory):
        print(f"❌ Directory not found: {directory}")
        return []
    
    classes = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path):
            classes.append(item)
    
    return sorted(classes)

def count_images_per_class(directory, classes):
    """각 클래스별 이미지 개수 카운트"""
    class_counts = {}
    for cls in classes:
        cls_path = os.path.join(directory, cls)
        if os.path.exists(cls_path):
            images = [f for f in os.listdir(cls_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            class_counts[cls] = len(images)
        else:
            class_counts[cls] = 0
    return class_counts

def match_classes(source_classes, target_classes, max_classes=47):
    """두 데이터셋의 클래스 매칭"""
    # 공통 클래스 찾기
    common_classes = list(set(source_classes) & set(target_classes))
    
    print(f"📊 Class Matching Results:")
    print(f"   Source classes: {len(source_classes)}")
    print(f"   Target classes: {len(target_classes)}")
    print(f"   Common classes: {len(common_classes)}")
    
    if len(common_classes) >= max_classes:
        # 공통 클래스가 충분하면 그 중에서 선택
        selected_classes = sorted(common_classes)[:max_classes]
        print(f"✅ Using {max_classes} common classes")
    else:
        # 공통 클래스가 부족하면 추가 매칭 시도
        print(f"⚠️  Need additional class matching...")
        selected_classes = common_classes.copy()
        
        # 유사한 이름의 클래스 매칭 시도 (예: car vs vehicle)
        remaining = max_classes - len(selected_classes)
        for src_cls in source_classes:
            if len(selected_classes) >= max_classes:
                break
            if src_cls not in selected_classes:
                # 유사 클래스 찾기 로직 (간단한 부분 문자열 매칭)
                for tgt_cls in target_classes:
                    if (src_cls.lower() in tgt_cls.lower() or 
                        tgt_cls.lower() in src_cls.lower()):
                        if tgt_cls not in [item[1] if isinstance(item, tuple) else item 
                                         for item in selected_classes]:
                            selected_classes.append((src_cls, tgt_cls))
                            break
    
    return selected_classes[:max_classes]

def create_data_lists(source_dir, target_dir, matched_classes, output_dir):
    """데이터 리스트 파일 생성"""
    os.makedirs(output_dir, exist_ok=True)
    
    source_list = []
    target_list = []
    test_list = []
    
    class_id = 0
    
    for item in matched_classes:
        if isinstance(item, tuple):
            source_class, target_class = item
        else:
            source_class = target_class = item
            
        print(f"Processing class {class_id}: {source_class} → {target_class}")
        
        # Source data (SDXL synthetic)
        source_class_dir = os.path.join(source_dir, source_class)
        if os.path.exists(source_class_dir):
            images = [f for f in os.listdir(source_class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            for img in images:
                img_path = os.path.join(source_class_dir, img)
                source_list.append(f"{img_path} {class_id}")
        
        # Target data (Baidu real)
        target_class_dir = os.path.join(target_dir, target_class)
        if os.path.exists(target_class_dir):
            images = [f for f in os.listdir(target_class_dir) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
            
            # Target을 train/test로 분할 (80:20)
            random.shuffle(images)
            split_idx = int(len(images) * 0.8)
            train_images = images[:split_idx]
            test_images = images[split_idx:]
            
            for img in train_images:
                img_path = os.path.join(target_class_dir, img)
                target_list.append(f"{img_path} {class_id}")
                
            for img in test_images:
                img_path = os.path.join(target_class_dir, img)
                test_list.append(f"{img_path} {class_id}")
        
        class_id += 1
    
    # 파일 저장
    source_file = os.path.join(output_dir, "source_list.txt")
    target_file = os.path.join(output_dir, "target_list.txt")
    test_file = os.path.join(output_dir, "test_list.txt")
    
    with open(source_file, 'w') as f:
        f.write('\n'.join(source_list))
    
    with open(target_file, 'w') as f:
        f.write('\n'.join(target_list))
        
    with open(test_file, 'w') as f:
        f.write('\n'.join(test_list))
    
    print(f"\n📝 Data lists created:")
    print(f"   Source: {len(source_list)} images → {source_file}")
    print(f"   Target: {len(target_list)} images → {target_file}")
    print(f"   Test: {len(test_list)} images → {test_file}")
    
    return source_file, target_file, test_file

def create_class_mapping(matched_classes, output_dir):
    """클래스 매핑 정보 저장"""
    mapping_file = os.path.join(output_dir, "class_mapping.txt")
    
    with open(mapping_file, 'w') as f:
        f.write("# 47-Class Mapping: Source → Target\n")
        f.write("# Format: class_id source_class target_class\n\n")
        
        for i, item in enumerate(matched_classes):
            if isinstance(item, tuple):
                source_class, target_class = item
            else:
                source_class = target_class = item
                
            f.write(f"{i} {source_class} {target_class}\n")
    
    print(f"📋 Class mapping saved: {mapping_file}")
    return mapping_file

def main():
    # 데이터셋 경로 설정
    SYNTHETIC_DIR = "/DATA_17/VM_team/Dataset/LVM_Datasets/LVM_ori/SDXL_meta_v1/images"
    REAL_DIR = "/DATA_17/DATASET/baidu_filtered_2"
    OUTPUT_DIR = "./data/SDXL_47class"
    
    print("🚀 47-Class Dataset Creation Starting...")
    print(f"   Source (Synthetic): {SYNTHETIC_DIR}")
    print(f"   Target (Real): {REAL_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    
    # 1. 클래스 스캔
    print("\n📂 Scanning directories...")
    source_classes = scan_directory_classes(SYNTHETIC_DIR)
    target_classes = scan_directory_classes(REAL_DIR)
    
    if not source_classes:
        print("❌ No source classes found!")
        return
    if not target_classes:
        print("❌ No target classes found!")
        return
    
    # 2. 이미지 개수 확인
    print("\n📊 Counting images...")
    source_counts = count_images_per_class(SYNTHETIC_DIR, source_classes)
    target_counts = count_images_per_class(REAL_DIR, target_classes)
    
    print(f"   Source total images: {sum(source_counts.values())}")
    print(f"   Target total images: {sum(target_counts.values())}")
    
    # 3. 클래스 매칭
    print("\n🎯 Matching classes...")
    matched_classes = match_classes(source_classes, target_classes, max_classes=47)
    
    if len(matched_classes) < 47:
        print(f"⚠️  Warning: Only {len(matched_classes)} classes matched (target: 47)")
    
    # 4. 데이터 리스트 생성
    print("\n📝 Creating data lists...")
    random.seed(42)  # 재현 가능한 결과를 위해
    
    source_file, target_file, test_file = create_data_lists(
        SYNTHETIC_DIR, REAL_DIR, matched_classes, OUTPUT_DIR
    )
    
    # 5. 클래스 매핑 저장
    mapping_file = create_class_mapping(matched_classes, OUTPUT_DIR)
    
    print("\n✅ 47-Class Dataset Creation Completed!")
    print(f"\n🎯 Ready for training with:")
    print(f"   python main_47class.py \\")
    print(f"     --source_list {source_file} \\")
    print(f"     --target_list {target_file} \\")
    print(f"     --test_list {test_file} \\")
    print(f"     --num_classes {len(matched_classes)} \\")
    print(f"     --dataset SDXL_47class")

if __name__ == "__main__":
    main() 