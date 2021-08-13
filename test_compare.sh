python3 test_compare_plots.py \
	--pretrained_weights="runs/train/hard-injecting-wdj2/weights/train-17.pt" \
	--infogan_disc_code=3 \
	--infogan_cont_code=2 \
	--exp_name=D2C1 \
	# --control_latent='cond' \
	--processed_data_dir="processed_data_walk_dance_jump/"
