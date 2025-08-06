# coding=utf-8
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .utils import extract_label_from_filename, extract_label_from_path, get_png_files_from_directory, get_image_files_from_directory


class FilenameBasedDataset(Dataset):
    """파일명 기반 데이터셋"""
    def __init__(self, image_paths, labels, transform=None, mode='source'):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"⚠️ 이미지 로드 실패: {path}, 오류: {e}")
            image = Image.new('RGB', (256, 256), color='gray')
            
        if self.transform:
            image = self.transform(image)
            
        if self.mode == 'target':
            return image, label, idx
        else:
            return image, label


def get_transforms(img_size=256):
    """데이터 증강 및 정규화 변환 함수들"""
    
    transform_source = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINO normalization
    ])
    
    transform_target = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINO normalization
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.CenterCrop((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # DINO normalization
    ])
    
    return transform_source, transform_target, transform_test


def setup_data_loaders(args, data_info):
    """데이터 로더 설정"""
    from torch.utils.data import DataLoader
    
    transform_source, transform_target, transform_test = get_transforms(args.img_size)
    
    source_loader = DataLoader(
        FilenameBasedDataset(data_info['synthetic_paths'], data_info['synthetic_labels'], 
                            transform=transform_source, mode='source'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    
    target_loader = DataLoader(
        FilenameBasedDataset(data_info['target_train_paths'], data_info['target_train_labels'], 
                            transform=transform_target, mode='target'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    test_loader = DataLoader(
        FilenameBasedDataset(data_info['target_test_paths'], data_info['target_test_labels'], 
                            transform=transform_test, mode='source'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
    
    return source_loader, target_loader, test_loader
