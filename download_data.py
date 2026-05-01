import os
import subprocess
import sys

def check_kaggle_installed():
    try:
        import kaggle
    except ImportError:
        print("Kaggle library not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])

def download_dataset():
    print("Preparing to download CIC-DDoS2019...")
    try:
        # Import inside function to ensure it's available after potential install
        import kaggle
        
        # Ensure data directory exists
        os.makedirs("data", exist_ok=True)
        
        print("Authenticating with Kaggle API...")
        kaggle.api.authenticate()
        
        print("Downloading and extracting dataset into 'data/' directory. This may take a while...")
        kaggle.api.dataset_download_cli('dhoogla/cicddos2019', path='data/', unzip=True)
        
        print("Download and extraction complete! Your data/ folder is ready.")
        
    except OSError as e:
        if "Could not find kaggle.json" in str(e):
            print("\n" + "="*50)
            print("KAGGLE API CREDENTIALS REQUIRED")
            print("="*50)
            print("To automate the download, you need a Kaggle API token.")
            print("1. Create a free account at https://www.kaggle.com")
            print("2. Go to your Account settings (Settings -> API section)")
            print("3. Click 'Create New Token'. This downloads 'kaggle.json'")
            print("4. Place 'kaggle.json' in your user directory:")
            print(f"   {os.path.join(os.path.expanduser('~'), '.kaggle', 'kaggle.json')}")
            print("   (You may need to create the .kaggle folder)")
            print("5. Run this script again.")
            print("="*50)
        else:
            print(f"An OS error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_kaggle_installed()
    download_dataset()
