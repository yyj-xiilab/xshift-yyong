# coding=utf-8
import os
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import AverageMeter
from utils.save import save_model, load_best_checkpoint
from trainer.validate import valid_cls_finetune, calculate_train_accuracy
from losses.contrastive import contrastive_loss
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule


def train_cls_finetune(args, model, cp_mask, prefix_saved_mode, data_info):
    """CLS Fine-tuning 학습"""
    # TensorBoard writer
    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=f"./runs/{prefix_saved_mode}")
    else:
        writer = None
    
    # Data loaders
    from dataset.loader import setup_data_loaders
    source_loader, target_loader, test_loader = setup_data_loaders(args, data_info)
    
    # Optimizer (CLS adapter와 classifier만 학습)
    trainable_params = []
    for name, param in model.named_parameters():
        if 'cls_adapter' in name or 'classifier' in name or 'projection' in name:
            trainable_params.append(param)
            print(f"   🎯 Trainable: {name}")
        else:
            param.requires_grad = False
            print(f"   ❄️ Frozen: {name}")
    
    optimizer = torch.optim.AdamW(trainable_params,
                                lr=args.learning_rate,
                                weight_decay=0.01)
    
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
        
    model.zero_grad()
        
    # Set seed
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
    
    losses = AverageMeter()
    
    # 기존 체크포인트에서 best_acc 복원
    best_acc = load_best_checkpoint(args, model, prefix_saved_mode)
    
    print(f"🎯 Starting CLS Token Fine-tuning...")
    print(f"   Source: Synthetic ({len(source_loader)} batches, {len(data_info['synthetic_paths'])} images)")
    print(f"   Target Train: Fixed Augmented ({len(target_loader)} batches, {len(data_info['target_train_paths'])} images)")
    print(f"   Target Test: Fixed Rest ({len(data_info['target_test_paths'])} images)")
    print(f"   Classes: {data_info['num_classes']}")
    print(f"   🏆 Best accuracy so far: {best_acc:.4f}")

    len_source = len(source_loader)
    len_target = len(target_loader)            

    for global_step in range(1, t_total):
        model.train()

        if (global_step-1) % (len_source-1) == 0:
            iter_source = iter(source_loader)    
        if (global_step-1) % (len_target-1) == 0:
            iter_target = iter(target_loader)
        
        data_source = next(iter_source)
        data_target = next(iter_target)

        x_s, y_s = tuple(t.to(args.device) for t in data_source)
        x_t, y_t, index_t = tuple(t.to(args.device) for t in data_target)
        
        # CLS Fine-tuning forward pass
        logits_s, logits_t, loss_ad_local, loss_rec, x_s_feat, x_t_feat, temp_mask, proj_s, proj_t = model(
            x_s=x_s, x_t=x_t, ad_net=None, cp_mask=cp_mask, optimal_flag=args.optimal)

        # Classification loss
        loss_fct = CrossEntropyLoss()
        loss_clc_s = loss_fct(logits_s.view(-1, args.num_classes), y_s.view(-1))
        loss_clc_t = loss_fct(logits_t.view(-1, args.num_classes), y_t.view(-1))
        
        # Contrastive loss for domain adaptation
        loss_contrastive = contrastive_loss(proj_s, proj_t, temperature=0.1)
        
        # Combined loss
        loss = loss_clc_s + loss_clc_t + 0.1 * loss_contrastive  # Contrastive loss weight: 0.1
            
        loss.backward()

        # 그래디언트 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        
        if args.local_rank in [-1, 0] and writer is not None:
            writer.add_scalar("train/loss", scalar_value=loss.item(), global_step=global_step)
            writer.add_scalar("train/loss_clc_s", scalar_value=loss_clc_s.item(), global_step=global_step)
            writer.add_scalar("train/loss_clc_t", scalar_value=loss_clc_t.item(), global_step=global_step)
            writer.add_scalar("train/loss_contrastive", scalar_value=loss_contrastive.item(), global_step=global_step)
            writer.add_scalar("train/lr", scalar_value=scheduler.get_last_lr()[0], global_step=global_step)
        
        if global_step % args.eval_every == 0 and args.local_rank in [-1, 0] and writer is not None:
            # Train accuracy 계산
            train_acc_s, train_acc_t = calculate_train_accuracy(args, model, source_loader, target_loader)
            
            # Validation accuracy 계산
            accuracy, classWise_acc = valid_cls_finetune(args, model, writer, test_loader, global_step, cp_mask, None, prefix_saved_mode, best_acc)
            
            # Train accuracy 출력
            print(f"📊 Train Source Acc: {train_acc_s:.5f}, Train Target Acc: {train_acc_t:.5f}")
            
            if best_acc < accuracy:
                best_acc = accuracy
                print(f"   🎉 New best accuracy: {best_acc:.4f}")

                save_model(args, model, prefix_saved_mode +str(best_acc) +'_')
                for file in os.listdir('./output/'+args.dataset):
                    if(prefix_saved_mode in file and 'checkpoint' in file ):
                        try:
                            if(best_acc > float(file.split('_')[4] ) ):
                                os.remove('./output/'+args.dataset+'/'+file)
                        except:
                            continue
