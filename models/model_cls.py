# coding=utf-8
import torch
import torch.nn as nn
import timm

from .classifier import CosineClassifier, LinearClassifier
from .adapter import CLSAdapter, LoRALayer


class DINOSmallCLSFinetune(nn.Module):
    """DINO-Small with CLS Token Fine-tuning"""
    def __init__(self, num_classes=47, img_size=256, freeze_backbone=True, use_lora=False, use_cosine=False):
        super(DINOSmallCLSFinetune, self).__init__()
        
        print("🎯 Initializing DINO-Small with CLS Token Fine-tuning...")
        
        # DINO-Small 모델 로드 (backbone만)
        self.dino_model = timm.create_model('vit_small_patch14_dinov2', 
                                           pretrained=True, 
                                           num_classes=0,  # 분류 헤드 제거
                                           img_size=img_size)
        
        # Backbone freeze (선택적)
        if freeze_backbone:
            for param in self.dino_model.parameters():
                param.requires_grad = False
            print("   ✅ Backbone frozen")
        else:
            print("   🔄 Backbone trainable")
        
        # LoRA 적용 (선택적)
        self.use_lora = use_lora
        if use_lora:
            self._apply_lora_to_blocks()
            print("   🔧 LoRA applied to transformer blocks")
        
        # CLS Adapter 추가
        self.cls_adapter = CLSAdapter(dim=384, hidden_dim=128)
        
        # Domain-specific 분류기 (Cosine or Linear)
        if use_cosine:
            self.classifier = CosineClassifier(dim=384, num_classes=num_classes)
            print("   🎯 Cosine Classifier loaded")
        else:
            self.classifier = LinearClassifier(dim=384, num_classes=num_classes)
            print("   🎯 Linear Classifier loaded")
        
        # Contrastive learning을 위한 projection head
        self.projection = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        print(f"   ✅ CLS Adapter + Classifier loaded: {num_classes} classes")
    
    def _apply_lora_to_blocks(self):
        """Transformer blocks에 LoRA 적용 (단순화된 버전)"""
        # LoRA를 CLS token 처리에만 적용
        self.cls_lora = LoRALayer(384, 384)
        print("   🔧 LoRA applied to CLS token processing only")
        
    def get_cls_token(self, x):
        """DINO에서 CLS token 추출"""
        # DINO의 forward_features 대신 직접 attention 블록 통과
        x = self.dino_model.patch_embed(x)
        x = self.dino_model._pos_embed(x)
        
        # CLS token 추가
        cls_token = self.dino_model.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Transformer blocks 통과
        for block in self.dino_model.blocks:
            x = block(x)
        
        # CLS token 추출 (첫 번째 토큰)
        cls_token = x[:, 0]  # [B, 384]
        
        # LoRA 적용 (선택적)
        if self.use_lora:
            cls_token = cls_token + self.cls_lora(cls_token)
        
        return cls_token
    
    def forward(self, x_s=None, x_t=None, ad_net=None, cp_mask=None, optimal_flag=1):
        """CLS token 중심 forward"""
        
        if x_t is not None:  # Training mode (source + target)
            # Source domain CLS token
            cls_s = self.get_cls_token(x_s)
            cls_s_adapted = self.cls_adapter(cls_s)
            logits_s = self.classifier(cls_s_adapted)
            
            # Target domain CLS token
            cls_t = self.get_cls_token(x_t)
            cls_t_adapted = self.cls_adapter(cls_t)
            logits_t = self.classifier(cls_t_adapted)
            
            # Contrastive learning을 위한 projection
            proj_s = self.projection(cls_s_adapted)
            proj_t = self.projection(cls_t_adapted)
            
            # Dummy values for compatibility
            loss_ad_local = torch.tensor(0.0).to(x_s.device)
            loss_rec = torch.tensor(0.0).to(x_s.device)
            combined_features_s = cls_s_adapted.unsqueeze(1)  # [B, 1, 384]
            combined_features_t = cls_t_adapted.unsqueeze(1)  # [B, 1, 384]
            current_cp_mask = torch.eye(325).to(x_s.device)
            
            return (logits_s, logits_t, loss_ad_local, loss_rec, 
                   combined_features_s, combined_features_t, current_cp_mask, proj_s, proj_t)
                   
        else:  # Inference mode (only source)
            cls = self.get_cls_token(x_s)
            cls_adapted = self.cls_adapter(cls)
            logits = self.classifier(cls_adapted)
            current_cp_mask = torch.eye(325).to(x_s.device)
            
            return logits, None, None, current_cp_mask
