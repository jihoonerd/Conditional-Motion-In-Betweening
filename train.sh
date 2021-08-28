python train_tfgan.py \
	--processed_data_dir="processed_data_walk/" \
	--batch_size=64 \
	--epochs=10000 \
	--device=0 \
	--entity=rilab-motion \
<<<<<<< HEAD
	--exp_name="A:TFLSGAN128_0.25" \
	--save_interval=400 \
=======
	--exp_name="A:TFG_CONVD(32, 0.25)" \
	--save_interval=200 \
>>>>>>> d46f173db9181046d387a3119a2cbbccbf605d73
	--generator_learning_rate=0.001 \
	--discriminator_learning_rate=0.0001 \
	--optim_beta1=0.9 \
	--optim_beta2=0.99 \
	--loss_generator_weight=1.0 \
	--loss_discriminator_weight=1.0
	# --cr_learning_rate=0.0001 \
	# --loss_root_weight=0.01 \
	# --loss_quat_weight=1.0 \
	# --loss_contact_weight=0.2 \
	# --loss_global_pos_weight=0.01
<<<<<<< HEAD
=======
	--loss_discriminator_weight=1.0 \
	--loss_generator_weight=1.0
>>>>>>> d46f173db9181046d387a3119a2cbbccbf605d73
	# --loss_code_weight=0.5 \
	# --infogan_disc_code=4 \
	# --infogan_cont_code=0 \
	# --loss_crh_weight=0.15