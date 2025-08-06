# coding=utf-8
import torch
import numpy as np
from tqdm import tqdm
from torch.nn import CrossEntropyLoss

from utils.metrics import simple_accuracy, AverageMeter


def valid_cls_finetune(args, model, writer, test_loader, global_step, cp_mask, ad_net, prefix_saved_mode, best_acc):
    """CLS Fine-tuning 검증"""
    model.eval()
    all_preds, all_label = [], []
    losses = AverageMeter()
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_loader, desc="Validating...")):
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch[0], batch[1]
            
            logits, _, _, _ = model(x_s=x)
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, args.num_classes), y.view(-1))
            
            losses.update(loss.item())
            
            if len(all_preds) == 0:
                all_preds = logits.detach().cpu().numpy()
                all_label = y.detach().cpu().numpy()
            else:
                all_preds = np.append(all_preds, logits.detach().cpu().numpy(), axis=0)
                all_label = np.append(all_label, y.detach().cpu().numpy(), axis=0)
    
    all_preds = np.argmax(all_preds, axis=1)
    accuracy = simple_accuracy(all_preds, all_label)
    
    if args.local_rank in [-1, 0]:
        writer.add_scalar("valid/loss", scalar_value=losses.avg, global_step=global_step)
        writer.add_scalar("valid/accuracy", scalar_value=accuracy, global_step=global_step)
        
        print(f"\n")
        print(f"🎯 CLS Fine-tuning Validation Results of: {prefix_saved_mode}")
        print(f"Global Steps: {global_step}")
        print(f"Valid Loss: {losses.avg:.5f}")
        print(f"Valid Accuracy: {accuracy:.5f}")
        print(f"🏆 Best Accuracy: {best_acc:.5f}")
        print(f"Current Best element-wise acc: ")
    
    return accuracy, ''


def calculate_train_accuracy(args, model, source_loader, target_loader):
    """Train accuracy 계산"""
    model.eval()
    all_preds_s, all_label_s = [], []
    all_preds_t, all_label_t = [], []
    
    with torch.no_grad():
        # Source accuracy
        for step, batch in enumerate(source_loader):
            if step >= 10:  # 최대 10 배치만 계산 (속도 향상)
                break
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch[0], batch[1]
            
            logits, _, _, _ = model(x_s=x)
            
            if len(all_preds_s) == 0:
                all_preds_s = logits.detach().cpu().numpy()
                all_label_s = y.detach().cpu().numpy()
            else:
                all_preds_s = np.append(all_preds_s, logits.detach().cpu().numpy(), axis=0)
                all_label_s = np.append(all_label_s, y.detach().cpu().numpy(), axis=0)
        
        # Target accuracy
        for step, batch in enumerate(target_loader):
            if step >= 10:  # 최대 10 배치만 계산 (속도 향상)
                break
            batch = tuple(t.to(args.device) for t in batch)
            x, y, _ = batch[0], batch[1], batch[2]
            
            logits, _, _, _ = model(x_s=x)
            
            if len(all_preds_t) == 0:
                all_preds_t = logits.detach().cpu().numpy()
                all_label_t = y.detach().cpu().numpy()
            else:
                all_preds_t = np.append(all_preds_t, logits.detach().cpu().numpy(), axis=0)
                all_label_t = np.append(all_label_t, y.detach().cpu().numpy(), axis=0)
    
    all_preds_s = np.argmax(all_preds_s, axis=1)
    all_preds_t = np.argmax(all_preds_t, axis=1)
    
    train_acc_s = simple_accuracy(all_preds_s, all_label_s)
    train_acc_t = simple_accuracy(all_preds_t, all_label_t)
    
    return train_acc_s, train_acc_t
