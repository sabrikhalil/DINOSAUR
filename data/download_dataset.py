import os
import tarfile
import requests
from tqdm import tqdm  # progress bar

VOC_URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
DATA_DIR = os.path.join(os.path.dirname(__file__), "PASCAL_VOC2012")

def download_and_extract_voc():
    if os.path.exists(DATA_DIR):
        print("Pascal VOC 2012 already downloaded and extracted.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    tar_path = os.path.join(DATA_DIR, "VOCtrainval.tar")
    
    print("Downloading Pascal VOC 2012...")
    response = requests.get(VOC_URL, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 KB
    progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
    
    with open(tar_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))
    progress_bar.close()
    
    print("Download complete. Extracting...")
    
    with tarfile.open(tar_path) as tar:
        tar.extractall(path=os.path.dirname(DATA_DIR))
    os.remove(tar_path)
    print("Extraction complete. Data available at:", DATA_DIR)

if __name__ == "__main__":
    download_and_extract_voc()
