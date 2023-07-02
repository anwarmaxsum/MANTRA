#export CUDA_VISIBLE_DEVICES=2


# python -u run_dualmode3k.py \
#   --is_training 1 \
#   --root_path ./dataset/ETT-small/ \
#   --data_path ETTm2.csv \
#   --model_id E3k_ETTm2_96_96 \
#   --model B6autoformer \
#   --slow_model AutoformerS1 \
#   --data ETTm2 \
#   --features S \
#   --seq_len 96 \
#   --label_len 48 \
#   --pred_len 96 \
#   --e_layers 2 \
#   --d_layers 1 \
#   --n_learner 3 \
#   --urt_head 1\
#   --learning_rate 0.0001 \
#   --factor 1 \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --des 'Exp' \
#   --train_epochs 20 \
#   --itr 3 \
#   --fix_seed 2023 \
#   --dropout 0.1 \
#   --d_model 256 \
#   --checkpoints ./checkpoints0/


python -u run_dualmode3k.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id E3k_ETTm2_UniVar_96_192 \
  --model B6autoformer \
  --slow_model AutoformerS1 \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --n_learner 3 \
  --urt_head 1 \
  --learning_rate 0.001 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 3 \
  --fix_seed 2023 \
  --dropout 0.01 \
  --anomaly 1.0 \
  --d_model 256 \
  --checkpoints ./checkpoints1/




python -u run_dualmode3k.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id E3k_ETTm2_UniVar_96_336 \
  --model B6autoformer \
  --slow_model AutoformerS1 \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 2 \
  --d_layers 1 \
  --n_learner 3 \
  --urt_head 1 \
  --learning_rate 0.001 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 3 \
  --fix_seed 2023 \
  --dropout 0.01 \
  --anomaly 1.0 \
  --d_model 256 \
  --checkpoints ./checkpoints2/


python -u run_dualmode3k.py \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm2.csv \
  --model_id E3k_ETTm2_UniVar_96_720 \
  --model B6autoformer \
  --slow_model AutoformerS1 \
  --data ETTm2 \
  --features S \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 2 \
  --d_layers 1 \
  --n_learner 3 \
  --urt_head 1 \
  --learning_rate 0.001 \
  --factor 1 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --train_epochs 20 \
  --itr 3 \
  --fix_seed 2023 \
  --dropout 0.01 \
  --anomaly 1.0 \
  --d_model 256 \
  --checkpoints ./checkpoints2/



