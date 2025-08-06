#!/bin/bash

# ==============================================
# 🎯 CLS Token Fine-tuning: Command Collection
# ==============================================

echo "🎯 CLS Token Fine-tuning Commands"
echo "=================================="

# ==============================================
# 🚀 기본 실행 명령어들
# ==============================================

echo ""
echo "🚀 기본 실행 명령어들"
echo "-------------------"

# 1. 완전한 설정 (권장)
echo "# 1. 완전한 설정 (LoRA + Cosine + 모든 기능)"
echo "python main.py --name cls_lora_cosine --dataset leader_filename_47class --use_lora --use_cosine --lora_rank 8 --lora_alpha 16"
echo ""

# 2. 기본 설정
echo "# 2. 기본 설정 (Linear Classifier, LoRA 없음)"
echo "python main.py --name cls_basic --dataset leader_filename_47class"
echo ""

# 3. LoRA만 사용
echo "# 3. LoRA만 사용 (Linear Classifier)"
echo "python main.py --name cls_lora_only --dataset leader_filename_47class --use_lora"
echo ""

# 4. Cosine만 사용
echo "# 4. Cosine만 사용 (LoRA 없음)"
echo "python main.py --name cls_cosine_only --dataset leader_filename_47class --use_cosine"
echo ""

# ==============================================
# 🔧 하이퍼파라미터 튜닝 명령어들
# ==============================================

echo ""
echo "🔧 하이퍼파라미터 튜닝 명령어들"
echo "------------------------------"

# 학습률 조정
echo "# 학습률 조정"
echo "python main.py --name cls_lora_cosine_lr1e2 --dataset leader_filename_47class --use_lora --use_cosine --learning_rate 1e-2"
echo "python main.py --name cls_lora_cosine_lr5e2 --dataset leader_filename_47class --use_lora --use_cosine --learning_rate 5e-2"
echo ""

# 배치 크기 조정
echo "# 배치 크기 조정"
echo "python main.py --name cls_lora_cosine_bs8 --dataset leader_filename_47class --use_lora --use_cosine --train_batch_size 8 --eval_batch_size 16"
echo "python main.py --name cls_lora_cosine_bs16 --dataset leader_filename_47class --use_lora --use_cosine --train_batch_size 16 --eval_batch_size 32"
echo ""

# 학습 스텝 수 조정
echo "# 학습 스텝 수 조정"
echo "python main.py --name cls_lora_cosine_10k --dataset leader_filename_47class --use_lora --use_cosine --num_steps 10000"
echo "python main.py --name cls_lora_cosine_20k --dataset leader_filename_47class --use_lora --use_cosine --num_steps 20000"
echo ""

# LoRA 파라미터 조정
echo "# LoRA 파라미터 조정"
echo "python main.py --name cls_lora_cosine_rank16 --dataset leader_filename_47class --use_lora --use_cosine --lora_rank 16 --lora_alpha 32"
echo "python main.py --name cls_lora_cosine_rank32 --dataset leader_filename_47class --use_lora --use_cosine --lora_rank 32 --lora_alpha 64"
echo ""

# ==============================================
# 🖥️ GPU 설정 명령어들
# ==============================================

echo ""
echo "🖥️ GPU 설정 명령어들"
echo "-------------------"

# 특정 GPU 사용
echo "# 특정 GPU 사용"
echo "CUDA_VISIBLE_DEVICES=0 python main.py --name cls_lora_cosine --dataset leader_filename_47class --use_lora --use_cosine"
echo "CUDA_VISIBLE_DEVICES=1 python main.py --name cls_lora_cosine --dataset leader_filename_47class --use_lora --use_cosine"
echo ""

# CPU만 사용 (테스트용)
echo "# CPU만 사용 (테스트용)"
echo "python main.py --name cls_lora_cosine --dataset leader_filename_47class --use_lora --use_cosine --no_cuda"
echo ""

# ==============================================
# 📊 모니터링 명령어들
# ==============================================

echo ""
echo "📊 모니터링 명령어들"
echo "-------------------"

# TensorBoard 로그 확인
echo "# TensorBoard 로그 확인"
echo "tensorboard --logdir ./runs"
echo ""

# 실시간 로그 확인
echo "# 실시간 로그 확인"
echo "python main.py --name cls_lora_cosine --dataset leader_filename_47class --use_lora --use_cosine 2>&1 | tee training.log"
echo ""

# ==============================================
# 🧪 실험 조합 명령어들
# ==============================================

echo ""
echo "🧪 실험 조합 명령어들"
echo "-------------------"

# 실험 1: 기본 vs LoRA
echo "# 실험 1: 기본 vs LoRA"
echo "python main.py --name exp1_basic --dataset leader_filename_47class"
echo "python main.py --name exp1_lora --dataset leader_filename_47class --use_lora"
echo ""

# 실험 2: Linear vs Cosine
echo "# 실험 2: Linear vs Cosine"
echo "python main.py --name exp2_linear --dataset leader_filename_47class --use_lora"
echo "python main.py --name exp2_cosine --dataset leader_filename_47class --use_lora --use_cosine"
echo ""

# 실험 3: LoRA Rank 비교
echo "# 실험 3: LoRA Rank 비교"
echo "python main.py --name exp3_rank8 --dataset leader_filename_47class --use_lora --use_cosine --lora_rank 8 --lora_alpha 16"
echo "python main.py --name exp3_rank16 --dataset leader_filename_47class --use_lora --use_cosine --lora_rank 16 --lora_alpha 32"
echo "python main.py --name exp3_rank32 --dataset leader_filename_47class --use_lora --use_cosine --lora_rank 32 --lora_alpha 64"
echo ""

# ==============================================
# 🚀 빠른 실행 함수들
# ==============================================

echo ""
echo "🚀 빠른 실행 함수들"
echo "-------------------"

echo "# 권장 실행 (복사해서 바로 실행)"
echo "python main.py --name cls_lora_cosine --dataset leader_filename_47class --use_lora --use_cosine --lora_rank 8 --lora_alpha 16"
echo ""

echo "# 테스트 실행 (빠른 확인용)"
echo "python main.py --name test_run --dataset leader_filename_47class --use_lora --use_cosine --num_steps 1000 --eval_every 200"
echo ""

echo "# 완전한 실험 (긴 시간)"
echo "python main.py --name full_experiment --dataset leader_filename_47class --use_lora --use_cosine --lora_rank 8 --lora_alpha 16 --num_steps 10000 --eval_every 500"
echo ""

# ==============================================
# 📝 사용법 안내
# ==============================================

echo ""
echo "📝 사용법 안내"
echo "-------------"
echo "1. 권장 실행: 위의 '권장 실행' 명령어를 복사해서 실행"
echo "2. 모니터링: tensorboard --logdir ./runs"
echo "3. 로그 확인: tail -f training.log"
echo "4. 체크포인트: ./output/leader_filename_47class/ 폴더에서 확인"
echo ""

echo "🎉 명령어 모음집 완성!"
echo "원하는 명령어를 복사해서 실행하세요." 