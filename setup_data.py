# Module: Automated Data Setup and Extraction

import os
import sys
import zipfile
import urllib.request
from pathlib import Path

def download_data(url, dest_path):
    """Downloads a file from a direct URL with a progress percentage."""
    print(f"Downloading from {url}...")
    try:
        def hook(count, block_size, total_size):
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r...{min(percent, 100)}%")
            sys.stdout.flush()
            
        urllib.request.urlretrieve(url, dest_path, reporthook=hook)
        print("\nDownload complete!")
    except Exception as e:
        print(f"\nFailed to download: {e}")
        sys.exit(1)

def extract_data(zip_path, extract_to):
    """Extracts the zip file into the target directory."""
    print(f"Extracting {zip_path} into {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Extraction complete!")
    except zipfile.BadZipFile:
        print("Error: The file provided is not a valid zip file.")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred during extraction: {e}")
        sys.exit(1)

def main():
    print("=== AneRBC Dataset Setup Utility ===\n")
    
    # Define our target directories
    data_dir = Path("data")
    target_extract_dir = data_dir / "AneRBC_raw_data"
    temp_zip_path = data_dir / "temp_dataset.zip"
    
    # Ensure the root data directory exists
    data_dir.mkdir(exist_ok=True)
    
    print("How would you like to provide the dataset?")
    print("  [1] Download via direct URL link")
    print("  [2] Provide the path to a locally downloaded ZIP file")
    
    choice = input("Enter 1 or 2: ").strip()
    
    if choice == '1':
        url = input("\nEnter the direct download URL: ").strip()
        download_data(url, temp_zip_path)
        extract_data(temp_zip_path, target_extract_dir)
        # Clean up the temp zip file to save space
        os.remove(temp_zip_path)
        
    elif choice == '2':
        local_zip = input("\nEnter the absolute or relative path to your ZIP file: ").strip()
        if not os.path.exists(local_zip):
            print(f"Error: Could not find file at '{local_zip}'")
            sys.exit(1)
        extract_data(local_zip, target_extract_dir)
        
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

    print(f"\nSuccess! Dataset is ready at: {target_extract_dir.absolute()}")
    print("You can now run 'python main.py' to start the pipeline.")

if __name__ == "__main__":
    main()