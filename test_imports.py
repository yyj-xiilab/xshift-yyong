# coding=utf-8
"""Test script to verify all imports work correctly"""

def test_imports():
    """Test all module imports"""
    try:
        # Test config
        from config import get_args
        print("✅ config.py imported successfully")
        
        # Test models
        from models.model_cls import DINOSmallCLSFinetune
        from models.classifier import CosineClassifier, LinearClassifier
        from models.adapter import CLSAdapter, LoRALayer
        print("✅ models module imported successfully")
        
        # Test dataset
        from dataset.loader import FilenameBasedDataset, setup_data_loaders
        from dataset.utils import extract_label_from_filename, extract_label_from_path
        print("✅ dataset module imported successfully")
        
        # Test losses
        from losses.contrastive import contrastive_loss
        print("✅ losses module imported successfully")
        
        # Test trainer
        from trainer.train import train_cls_finetune
        from trainer.validate import valid_cls_finetune, calculate_train_accuracy
        print("✅ trainer module imported successfully")
        
        # Test utils
        from utils.metrics import simple_accuracy, AverageMeter
        from utils.save import save_model, count_parameters, load_best_checkpoint
        from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
        from utils.logger import setup_logging
        print("✅ utils module imported successfully")
        
        print("\n🎉 All imports successful! The modular structure is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False


if __name__ == "__main__":
    test_imports() 