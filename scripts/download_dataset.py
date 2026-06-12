import os
import sys
import json
import argparse
import subprocess


def configure_kaggle(username, key):
    os.makedirs(os.path.expanduser("~/.config/kaggle"), exist_ok=True)
    creds_path = os.path.expanduser("~/.config/kaggle/kaggle.json")
    with open(creds_path, "w") as f:
        json.dump({"username": username, "key": key}, f)
    os.chmod(creds_path, 0o600)
    print("Kaggle configurado")


def download_dataset(output_dir="data/raw"):
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "kaggle", "datasets", "download",
        "-d", "muskanverma24/pothole-detection-dataset-yolov11-optimized",
        "-p", output_dir,
        "--unzip",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error al descargar:", result.stderr)
        sys.exit(1)
    print(f"Dataset descargado en {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", required=True, help="Usuario de Kaggle")
    parser.add_argument("--key", required=True, help="API key de Kaggle")
    parser.add_argument("--output", default="data/raw", help="Carpeta de destino")
    args = parser.parse_args()

    configure_kaggle(args.username, args.key)
    download_dataset(args.output)
