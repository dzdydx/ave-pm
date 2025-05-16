
nohup python supv_main.py \
--gpu 0 \
--lr 0.0007 \
--clip_gradient 0.5 \
--snapshot_pref "./Exps/" \
--n_epoch 200 \
--b 64 \
--test_batch_size 16 \
--print_freq 1 \
> log.log 2>&1 &
