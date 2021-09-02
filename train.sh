python train_baseline.py \
	--processed_data_dir="processed_data_all/" \
	--batch_size=64 \
	--epochs=3000 \
	--device=3 \
	--entity=rilab-motion \
	--exp_name="CMIP_BASE(max horizon)" \
	--save_interval=100 \
	--learning_rate=0.0001 \
	--optim_beta1=0.9 \
	--optim_beta2=0.99 \
	--loss_root_weight=0.01 \
	--loss_quat_weight=1.0 \
	--loss_contact_weight=0.2 \
	--loss_global_pos_weight=0.01 \
	--from_idx=9 \
	--target_idx=49
