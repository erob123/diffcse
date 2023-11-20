#!/bin/bash

LR=7e-6
MASK=0.30
LAMBDA=0.005

python train.py \
    --model_name_or_path voidism/diffcse-bert-base-uncased-sts \
    --generator_name distilbert-base-uncased \
    --train_file "cui/full_text.txt" \
    --output_dir cui/models/alexa \
    --num_train_epochs 2 \
    --per_device_train_batch_size 64 \
    --learning_rate $LR \
    --max_seq_length 32 \
    --evaluation_strategy no \
    --load_best_model_at_end \
    --eval_steps 125 \
    --pooler_type cls_before_pooler \
    --mlp_only_train \
    --overwrite_output_dir \
    --logging_first_step \
    --logging_dir cui/logs/alexa \
    --temp 0.05 \
    --do_train \
    --do_eval \
    --batchnorm \
    --lambda_weight $LAMBDA \
    --fp16 --masking_ratio $MASK

#     # --metric_for_best_model stsb_spearman \
