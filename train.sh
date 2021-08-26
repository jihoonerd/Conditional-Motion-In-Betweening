python train.py \
	--processed_data_dir="processed_data_walk/" \
	--batch_size=16 \
	--epochs=5000 \
	--device=3 \
	--entity=rilab-motion \
	--exp_name="W: BERT WO LERP (16)" \
	--save_interval=200 \
	--generator_learning_rate=0.0001 \
	--discriminator_learning_rate=0.00001 \
	--cr_learning_rate=0.0001 \
	--optim_beta1=0.9 \
	--optim_beta2=0.99 \
	--loss_root_weight=0.01 \
	--loss_quat_weight=1.0 \
	--loss_contact_weight=0.2 \
	--loss_global_pos_weight=0.01
	# --loss_discriminator_weight=1.0 \
	# --loss_generator_weight=0.5 \
	# --loss_code_weight=0.5 \
	# --infogan_disc_code=4 \
	# --infogan_cont_code=0 \
	# --loss_crh_weight=0.15