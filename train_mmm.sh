python train_mmm.py \
	--processed_data_dir="processed_data_original/" \
	--batch_size=32 \
	--epochs=10000 \
	--device=3 \
	--entity=rilab-motion \
	--exp_name="noised_0_only_model" \
	--save_interval=50 \
	--learning_rate=0.0001 \
	--optim_beta1=0.9 \
	--optim_beta2=0.99 \
	--loss_pos_weight=0.03 \
	--loss_rot_weight=1.0 \
	--from_idx=9 \
	--target_idx=40
	# --preserve_link_train
