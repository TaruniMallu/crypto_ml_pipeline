# src/models/predict_all.py

import os
import subprocess

def run_prediction(coin, input_file):
    cmd = [
        "python",
        "src/models/predict_model.py",
        "--coin", coin,
        "--input", input_file
    ]

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = process.communicate()

    if process.returncode != 0:
        print(f"[ERROR] ❌ Failed for {coin}")
        print(err.decode())
    else:
        print(f"[OK] ✔ Done: {coin}")


def main():
    directory = "data/new"
    files = [f for f in os.listdir(directory) if f.endswith("_4h_new.csv")]

    print(f"[INFO] Found {len(files)} files. Starting predictions...\n")

    for file in files:
        coin = file.replace("_4h_new.csv", "")
        input_path = f"{directory}/{file}"

        run_prediction(coin, input_path)

    print("\n[INFO] 🎉 All predictions completed!")


if __name__ == "__main__":
    main()
