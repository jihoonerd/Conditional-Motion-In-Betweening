import subprocess


for epoch in range(1000, 1200, 10):
    print(f"Epochs: {epoch}")
    subprocess.run(["python", "test_benchmark.py", 
                    "--project", "runs/train", 
                    "--exp_name", "slerp30_qnorm_final_bc",
                    "--weight", str(epoch),
                    "--processed_data_dir", "processed_data_original_bc/"])

