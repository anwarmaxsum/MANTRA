python -u run_dualmodb.py \
  --is_training 1 \
  --root_path ./dataset/illness/ \
  --data_path national_illness.csv \
  --model_id ili_MantraB_36_36 \
  --model Mantra \
  --slow_model MantraB \
  --data custom \
  --features M \
  --seq_len 36 \
  --label_len 18 \
  --pred_len 36 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --n_learner 10 \
  --urt_heads 7 \
  --itr 3 \
  --fix_seed 2021,2022,2023 \
  --checkpoints ./checkpoints1/