# coding=utf-8
import os
import torch
import logging

logger = logging.getLogger(__name__)


def save_model(args, model, prefix_saved_mode, is_adv=False):
    """모델 저장"""
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(output_dir, "%s_checkpoint.bin" % prefix_saved_mode)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)


def count_parameters(model):
    """모델 파라미터 수 계산"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_best_checkpoint(args, model, prefix_saved_mode):
    """최고 성능 체크포인트 로드"""
    best_acc = 0
    best_model = None
    
    if not os.path.exists('./output/'+args.dataset):
        os.makedirs('./output/'+args.dataset)
        
    for file in os.listdir('./output/'+args.dataset):
        if prefix_saved_mode in file and 'checkpoint' in file:
            try:
                acc = float(file.split('_')[4])
                if acc > best_acc:
                    best_acc = acc
                    best_model = file
            except:
                continue
    
    if best_model is not None:
        print(f"   🏆 Best checkpoint found: {best_model} (Acc: {best_acc:.4f})")
        checkpoint = torch.load('./output/'+args.dataset+'/'+best_model, map_location='cpu')
        model.load_state_dict(checkpoint, strict=False)
    else:
        print("   🆕 No checkpoint found, starting from scratch")
    
    return best_acc
