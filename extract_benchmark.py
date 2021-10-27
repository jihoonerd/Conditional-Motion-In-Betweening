import subprocess


if __name__ == "__main__":

    for i in range(300, 7201, 25):
        print(i)
        subprocess.run(["python", "test_benchmark.py", "--weight", str(i)])  # doesn't capture output