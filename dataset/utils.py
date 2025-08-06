# coding=utf-8
import os
import glob


def extract_label_from_filename(filename):
    """파일명에서 라벨 추출: a_train_05531_meta_v1.png -> 'train'"""
    try:
        label = filename.split('_')[1]
        return label
    except Exception as e:
        print(f"⚠️ 라벨 추출 실패: {filename}, 오류: {e}")
        return "unknown"


def extract_label_from_path(file_path):
    """경로에서 라벨 추출: /path/to/Pickup truck/image.png -> 'Pickup truck'"""
    try:
        # 폴더명이 라벨
        folder_name = os.path.basename(os.path.dirname(file_path))
        return folder_name
    except Exception as e:
        print(f"⚠️ 경로에서 라벨 추출 실패: {file_path}, 오류: {e}")
        return "unknown"


def get_png_files_from_directory(directory):
    """디렉토리에서 모든 PNG 파일 경로를 가져옴"""
    pattern = os.path.join(directory, "*.png")
    return glob.glob(pattern)


def get_image_files_from_directory(directory):
    """디렉토리에서 모든 이미지 파일 경로를 가져옴 (재귀적 검색)"""
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                all_files.append(os.path.join(root, file))
    return all_files
