# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
import argparse
import os
import math
import random
import numpy as np
from PIL import Image
import glob
from datetime import timedelta

import torch
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from models.modeling import VisionTransformer, CONFIGS, AdversarialNetwork
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.data_utils import get_loader
from utils.dist_util import get_world_size
from utils.transform import get_transform
from utils.utils import visda_acc, get_mask, re_org_img, mix_img

from torchvision import transforms, datasets
from data.data_list_image import ImageList, ImageListIndex, rgb_loader
from models import lossZoo
import networkx as nx
import cv2
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ==============================================
# 🚀 리더님 방식: 파일명 기반 47개 클래스 매칭
# ==============================================

def extract_label_from_filename(filename):
    """파일명에서 라벨 추출: a_train_05531_meta_v1.png -> 'train'"""
    try:
        parts = filename.split('_')
        if len(parts) >= 2:
            return parts[1]  # 두 번째 부분이 라벨
        else:
            return 'unknown'
    except:
        return 'unknown'

def get_png_files_from_directory(directory):
    """디렉토리에서 모든 PNG 파일 경로를 가져옴"""
    pattern = os.path.join(directory, "*.png")
    return glob.glob(pattern)

class FilenameBasedDataset(Dataset):
    """파일명 기반 데이터셋 (리더님 방식)"""
    def __init__(self, image_paths, labels, transform=None, mode='source'):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.mode = mode
        
    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            # 이미지 로드
            image = Image.open(image_path).convert('RGB')
        except:
            # 오류 시 더미 이미지
            image = Image.new('RGB', (256, 256), color='black')
        
        if self.transform:
            image = self.transform(image)
            
        if self.mode == 'target':
            return image, label, idx  # target은 index도 반환
        else:
            return image, label

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def save_model(args, model, prefix_saved_mode, is_adv=False, ):
    model_to_save = model.module if hasattr(model, 'module') else model
    if is_adv:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, prefix_saved_mode+args.name+"_checkpoint_adv"+"_"+".bin" )
    else:
        model_checkpoint = os.path.join(args.output_dir, args.dataset, prefix_saved_mode+args.name+"_checkpoint"+"_"+".bin")
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", os.path.join(args.output_dir, args.dataset))

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def setup_leader_filename_model(args, prefix_saved_mode):
    """리더님 방식: 파일명 기반 47개 클래스 매칭 모델 설정"""
    # 하드코딩된 경로 (리더님과 동일)
    synthetic_dir1 = "/DATA_17/VM_team/Dataset/LVM_Datasets/LVM_ori/SDXL_meta_v1/images"
    synthetic_dir2 = "./result_gen"
    val_dir = "/DATA_17/DATASET/baidu_filtered_2"
    
    print("🚀 리더님 방식: 파일명 기반 47개 클래스 매칭 시작...")
    
    # 1. Synthetic 데이터 수집 (SDXL_meta_v1 + result_gen)
    synthetic_paths = []
    synthetic_labels = []
    
    # SDXL_meta_v1에서 PNG 파일 수집
    if os.path.exists(synthetic_dir1):
        sdxl_files = get_png_files_from_directory(synthetic_dir1)
        for file_path in sdxl_files:
            filename = os.path.basename(file_path)
            label = extract_label_from_filename(filename)
            synthetic_paths.append(file_path)
            synthetic_labels.append(label)
        print(f"   SDXL_meta_v1: {len(sdxl_files)} images")
    
    # result_gen에서 PNG 파일 수집
    if os.path.exists(synthetic_dir2):
        result_files = get_png_files_from_directory(synthetic_dir2)
        for file_path in result_files:
            filename = os.path.basename(file_path)
            label = extract_label_from_filename(filename)
            synthetic_paths.append(file_path)
            synthetic_labels.append(label)
        print(f"   result_gen: {len(result_files)} images")
    
    # 2. Real validation 데이터 수집 (baidu_filtered_2)
    val_paths = []
    val_labels = []
    
    if os.path.exists(val_dir):
        val_files = get_png_files_from_directory(val_dir)
        for file_path in val_files:
            filename = os.path.basename(file_path)
            label = extract_label_from_filename(filename)
            val_paths.append(file_path)
            val_labels.append(label)
        print(f"   baidu_filtered_2: {len(val_files)} images")
    
    # 3. 공통 클래스 찾기 및 47개로 제한
    synthetic_classes = set(synthetic_labels)
    val_classes = set(val_labels)
    common_classes = list(synthetic_classes.intersection(val_classes))
    
    # 47개로 제한
    if len(common_classes) > 47:
        common_classes = sorted(common_classes)[:47]
    
    print(f"   공통 클래스: {len(common_classes)}개")
    print(f"   클래스 목록: {common_classes[:10]}..." if len(common_classes) > 10 else f"   클래스 목록: {common_classes}")
    
    # 4. 클래스 매핑 생성 (문자열 -> 숫자)
    class_to_idx = {cls: idx for idx, cls in enumerate(common_classes)}
    
    # 5. 공통 클래스만 필터링
    filtered_synthetic_paths = []
    filtered_synthetic_labels = []
    for path, label in zip(synthetic_paths, synthetic_labels):
        if label in class_to_idx:
            filtered_synthetic_paths.append(path)
            filtered_synthetic_labels.append(class_to_idx[label])
    
    filtered_val_paths = []
    filtered_val_labels = []
    for path, label in zip(val_paths, val_labels):
        if label in class_to_idx:
            filtered_val_paths.append(path)
            filtered_val_labels.append(class_to_idx[label])
    
    print(f"   필터링된 Synthetic: {len(filtered_synthetic_paths)} images")
    print(f"   필터링된 Real: {len(filtered_val_paths)} images")
    
    if len(filtered_val_paths) > 0:
        ratio = len(filtered_synthetic_paths) / len(filtered_val_paths)
        print(f"   데이터 비율 (Synthetic:Real): {ratio:.1f}:1")
    else:
        print("   Real 데이터 없음!")
    
    # 6. args 업데이트
    args.num_classes = len(common_classes)
    
    # 7. 모델 설정
    config = CONFIGS[args.model_type]
    model = VisionTransformer(config, args.img_size, zero_head=True, 
                              num_classes=args.num_classes, msa_layer=args.msa_layer)
    
    # 8. 체크포인트 정리
    best_acc = 0
    best_model = None
    if(not os.path.exists('./output/'+args.dataset)):
        os.makedirs('./output/'+args.dataset)
        
    for file in os.listdir('./output/'+args.dataset):
        if(prefix_saved_mode in file and 'checkpoint' in file ):
            try:
                if(best_acc > float(file.split('_')[4] ) ):
                    os.remove('./output/'+args.dataset+'/'+file)
            except (ValueError, IndexError):
                continue

    # 9. 사전훈련 모델 로드
    if(best_model is not None):
        model_checkpoint = os.path.join(args.output_dir, args.dataset, best_model)
        model.load_state_dict(torch.load(model_checkpoint))
    elif args.pretrained_dir is not None and os.path.exists(args.pretrained_dir):
        print('pretrained model:', args.pretrained_dir)
        model.load_from(np.load(args.pretrained_dir))
    else:
        print('No pretrained model found. Starting from scratch.')

    num_params = count_parameters(model)
    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)
    print(f"Total Parameters: {num_params:.1f}M")

    # 10. 데이터 정보 반환
    data_info = {
        'synthetic_paths': filtered_synthetic_paths,
        'synthetic_labels': filtered_synthetic_labels,
        'val_paths': filtered_val_paths,
        'val_labels': filtered_val_labels,
        'num_classes': len(common_classes),
        'class_to_idx': class_to_idx,
        'idx_to_class': {idx: cls for cls, idx in class_to_idx.items()}
    }
    
    return args, model, data_info

def valid(args, model, writer, test_loader, global_step, cp_mask, ad_net, prefix_saved_mode):
    # Validation!
    eval_losses = AverageMeter()

    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    ad_net.eval()
    all_preds, all_label = [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=args.local_rank not in [-1, 0])
    loss_fct = torch.nn.CrossEntropyLoss()
    valid_cp = 0 
    for step, batch in enumerate(epoch_iterator):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch

        with torch.no_grad():
            logits, _, _, temp_cp = model(x_s = x, ad_net=ad_net, cp_mask=cp_mask, 
                 optimal_flag = args.optimal, )
            valid_cp = valid_cp + temp_cp.detach()

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label = all_preds[0], all_label[0]
    accuracy = simple_accuracy(all_preds, all_label)

    logger.info("\n")
    logger.info("Validation Results of: %s" % args.name)
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    return accuracy, None

def train_leader_filename(args, model, cp_mask, prefix_saved_mode, data_info):
    """리더님 방식: 파일명 기반 Domain Adaptation 학습"""
    best_acc = 0
    for file in os.listdir('./output/'+args.dataset):
        if(prefix_saved_mode in file and 'checkpoint' in file ):
            try:
                if(best_acc < float(file.split('_')[4] ) ):
                    best_acc = float(file.split('_')[4])
            except (ValueError, IndexError):
                continue

    if args.local_rank in [-1, 0]:
        os.makedirs(os.path.join(args.output_dir, args.dataset), exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join("logs", args.dataset, args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # 리더님 방식에 맞는 변환 (리더님과 동일)
    transform_source = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 리더님과 동일
    ])
    
    transform_target = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 리더님과 동일
    ])
    
    transform_test = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.CenterCrop((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 리더님과 동일
    ])
    
    source_loader = torch.utils.data.DataLoader(
        FilenameBasedDataset(data_info['synthetic_paths'], data_info['synthetic_labels'], 
                            transform=transform_source, mode='source'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)
    
    target_loader = torch.utils.data.DataLoader(
        FilenameBasedDataset(data_info['val_paths'], data_info['val_labels'], 
                            transform=transform_target, mode='target'),
        batch_size=args.train_batch_size, shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(
        FilenameBasedDataset(data_info['val_paths'], data_info['val_labels'], 
                            transform=transform_test, mode='source'),
        batch_size=args.eval_batch_size, shuffle=False, num_workers=4)
    
    config = CONFIGS[args.model_type]
    ad_net = AdversarialNetwork(config.hidden_size, config.hidden_size//4)
    ad_net.to(args.device)
    ad_net_local = AdversarialNetwork(config.hidden_size//12, config.hidden_size//12)
    ad_net_local.to(args.device)
    
    optimizer_ad = torch.optim.SGD(list(ad_net.parameters())+list(ad_net_local.parameters()),
                            lr=args.learning_rate/10, 
                            momentum=0.9,
                            weight_decay=args.weight_decay)
    
    optimizer = torch.optim.SGD([
                                    {'params': model.transformer.parameters(), 'lr': args.learning_rate/10},
                                    {'params': model.decoder.parameters(), 'lr': args.learning_rate},
                                    {'params': model.head.parameters()},
                                ],
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupCosineSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        scheduler_ad = WarmupLinearSchedule(optimizer_ad, warmup_steps=args.warmup_steps, t_total=t_total)
        
    model.zero_grad()
    ad_net.zero_grad()
    ad_net_local.zero_grad()
        
    set_seed(args)
    losses = AverageMeter()
    best_classWise_acc = ''

    len_source = len(source_loader)
    len_target = len(target_loader)            

    print(f"🚀 Starting Leader's Filename-Based Domain Adaptation Training...")
    print(f"   Source: Synthetic ({len_source} batches, {len(data_info['synthetic_paths'])} images)")
    print(f"   Target: Real ({len_target} batches, {len(data_info['val_paths'])} images)")
    print(f"   Classes: {data_info['num_classes']}")

    for global_step in range(1, t_total):
        model.train()
        ad_net.train()
        ad_net_local.train()

        if (global_step-1) % (len_source-1) == 0:
            iter_source = iter(source_loader)    
        if (global_step-1) % (len_target-1) == 0:
            iter_target = iter(target_loader)
        
        data_source = next(iter_source)
        data_target = next(iter_target)

        x_s, y_s = tuple(t.to(args.device) for t in data_source)
        x_t, y_t, index_t = tuple(t.to(args.device) for t in data_target)
        
        if( not args.use_cp ):
            cp_mask = np.ones( (257, 257))
            cp_mask = torch.from_numpy(cp_mask).float().to(args.device)

        logits_s, logits_t, loss_ad_local, loss_rec, x_s, x_t, temp_mask = model(x_s = x_s, x_t = x_t, ad_net = ad_net_local, cp_mask=cp_mask, 
            optimal_flag = args.optimal, )
        cp_mask = temp_mask

        loss_fct = CrossEntropyLoss()
        loss_clc = loss_fct(logits_s.view(-1, args.num_classes), y_s.view(-1))
        
        loss_im = lossZoo.im(logits_t.view(-1, args.num_classes))
        loss_ad_global = lossZoo.adv(torch.cat((x_s[:,0], x_t[:,0]), 0), ad_net)
        loss = loss_clc + args.beta * loss_ad_global + args.gamma * loss_ad_local

        if args.use_im:
            loss += (args.theta * loss_im)
            
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net.parameters(), args.max_grad_norm)
        torch.nn.utils.clip_grad_norm_(ad_net_local.parameters(), args.max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        optimizer_ad.step()
        optimizer_ad.zero_grad()
        scheduler_ad.step()
        
        if args.local_rank in [-1, 0]:
            writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar("train/loss_clc", scalar_value=loss_clc.item(), global_step=global_step)
            writer.add_scalar("train/loss_ad_global", scalar_value=loss_ad_global.item(), global_step=global_step)
            writer.add_scalar("train/loss_ad_local", scalar_value=loss_ad_local.item(), global_step=global_step)
            writer.add_scalar("train/loss_rec", scalar_value=loss_rec.item(), global_step=global_step)
            writer.add_scalar("train/loss_im", scalar_value=loss_im.item(), global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
        
        if global_step % args.eval_every == 0 and args.local_rank in [-1, 0]:
            accuracy, classWise_acc = valid(args, model, writer, test_loader, global_step, cp_mask, ad_net_local, prefix_saved_mode)
            if best_acc < accuracy:
                best_acc = accuracy

                save_model(args, model, prefix_saved_mode +str(best_acc) +'_',  is_adv=False, )
                save_model(args, ad_net_local, prefix_saved_mode +str(best_acc) +'_', is_adv=True, )
                for file in os.listdir('./output/'+args.dataset):
                    if(prefix_saved_mode in file and 'checkpoint' in file ):
                        try:
                            if(best_acc > float(file.split('_')[4] ) ):
                                os.remove('./output/'+args.dataset+'/'+file)
                        except (ValueError, IndexError):
                            continue

                if classWise_acc is not None:
                    best_classWise_acc = classWise_acc
            model.train()
            ad_net_local.train()
            logger.info("Current Best Accuracy: %2.5f" % best_acc)
            logger.info("Current Best element-wise acc: %s" % best_classWise_acc)
        
    if args.local_rank in [-1, 0]:
        writer.close()
    logger.info("Best Accuracy: \t%f" % best_acc)
    logger.info("Best element-wise Accuracy: \t%s" % best_classWise_acc)
    logger.info("End Training!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_cp", default=False, action="store_true",
                        help="Use Core periphery constraint.")

    parser.add_argument("--name", default = 'leader47_filename_da', type=str, 
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default = 'leader_filename_47class', type=str, help="Which downstream task.")
    parser.add_argument("--num_classes", default=47, type=int,
                        help="Number of classes in the dataset (will be auto-detected).")
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_16",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/imagenet21k_ViT-B_16.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=256, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=4, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=100, type=int,
                        help="Run prediction on validation set every so many steps.")

    parser.add_argument("--beta", default=0.1, type=float,
                        help="The importance of the adversarial loss.")
    parser.add_argument("--gamma", default=0.2, type=float,
                        help="The importance of the local adversarial loss.")
    parser.add_argument("--theta", default=0.1, type=float,
                        help="The importance of the IM loss.")
    parser.add_argument("--use_im", default=True, action="store_true",
                        help="Use information maximization loss.")
    parser.add_argument("--msa_layer", default=12, type=int,
                        help="The layer that incorporates local alignment.")
    parser.add_argument("--is_test", default=False, action="store_true",
                        help="If in test mode.")

    parser.add_argument("--learning_rate", default=0.05, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0.01, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=3000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=300, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3'].")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability.")
    parser.add_argument('--gpu_id', default='0', type = str,
                        help="gpu id")
    parser.add_argument('--optimal', default=1, type = int,
                        help="use optimal linear transform noise, 1 means use, 0 means no use")
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    cuda = True if torch.cuda.is_available() else False
    if torch.cuda.device_count() > 0:
       print("Let's use", torch.cuda.device_count(), "GPUs!")

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s" %
                   (args.local_rank, args.device, args.n_gpu, bool(args.local_rank != -1), args.fp16))
    
    set_seed(args)

    if( not args.use_cp ):
        prefix_saved_mode = args.name+'_NoCP_'+'Perturbation_' + str(args.optimal) + '_'
    else:
        prefix_saved_mode = args.name+'_SoftCP_'+'Perturbation_' + str(args.optimal) + '_'

    num_patchs = int(args.img_size * args.img_size / 16 / 16)
    cp_mask = np.ones( (num_patchs+1, num_patchs+1))
    cp_mask = torch.from_numpy(cp_mask).float().to(args.device)
    
    # 🚀 리더님 방식: 파일명 기반 47개 클래스 매칭
    args, model, data_info = setup_leader_filename_model(args, prefix_saved_mode)
    model.to(args.device)
    train_leader_filename(args, model, cp_mask, prefix_saved_mode, data_info)

if __name__ == "__main__":
    main() 