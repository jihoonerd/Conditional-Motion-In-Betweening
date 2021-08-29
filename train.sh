python train_tfseq2seq.py \
	--processed_data_dir="processed_data_all/" \
	--batch_size=64 \
	--epochs=10000 \
	--device=0 \
	--entity=rilab-motion \
	--exp_name="AE_TF_SEQ2SEQ_btnck_512" \
	--save_interval=200 \
	--generator_learning_rate=0.001 \
	--discriminator_learning_rate=0.0001 \
	--optim_beta1=0.5 \
	--optim_beta2=0.99 \
	--loss_recon_weight=1.0 \
	--loss_fk_weight=0.1 \
	--bottleneck_dim=512

	# --loss_generator_weight=1.0 \
	# --loss_discriminator_weight=1.0
	# --cr_learning_rate=0.0001 \
	# --loss_root_weight=0.01 \
	# --loss_quat_weight=1.0 \
	# --loss_contact_weight=0.2 \
	# --loss_global_pos_weight=0.01
	# --loss_code_weight=0.5 \
	# --infogan_disc_code=4 \
	# --infogan_cont_code=0 \
	# --loss_crh_weight=0.15