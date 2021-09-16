python train_mmm.py \
	--processed_data_dir="processed_data_original/" \
	--batch_size=1024 \
	--epochs=10000 \
	--device=3 \
	--entity=rilab-motion \
	--exp_name="preserve_link_pp" \
	--save_interval=100 \
	--learning_rate=0.0001 \
	--optim_beta1=0.9 \
	--optim_beta2=0.99 \
	--loss_pos_weight=0.03 \
	--loss_rot_weight=1.0 \
	--from_idx=9 \
	--target_idx=40
	# --preserve_link_train
