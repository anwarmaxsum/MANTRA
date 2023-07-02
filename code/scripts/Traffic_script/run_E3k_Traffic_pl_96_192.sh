
python -u run_dualmode3k.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id E3k_Traffic_96_96 \
  --model B6autoformer \
  --slow_model AutoformerS1 \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_head 1\
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --itr 3 \
  --fix_seed 2023 \
  --dropout 0.1 \
  --d_model 256 \
  --checkpoints ./checkpoints0/



python -u run_dualmode3k.py \
  --is_training 1 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id E3k_Traffic_96_192 \
  --model B6autoformer \
  --slow_model AutoformerS1 \
  --data custom \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --n_learner 3 \
  --urt_head 1\
  --learning_rate 0.0001 \
  --train_epochs 20 \
  --itr 3 \
  --fix_seed 2023 \
  --dropout 0.1 \
  --d_model 256 \
  --checkpoints ./checkpoints1/