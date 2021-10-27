python train_mmm.py \
	--processed_data_dir="processed_data_original_80/" \
	--window=90 \
	--batch_size=32 \
	--epochs=10000 \
	--device=1 \
	--entity=rilab-motion \
	--exp_name="slerp80_qnorm" \
	--save_interval=25 \
	--learning_rate=0.0001 \
	--optim_beta1=0.9 \
	--optim_beta2=0.99 \
	--loss_cond_weight=2.0 \
	--loss_pos_weight=0.05 \
	--loss_rot_weight=1.0 \
	--from_idx=9 \
	--target_idx=88 \
	--interpolation='slerp'
