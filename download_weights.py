import os
import gdown

# Google Drive shareable file URL
url = "https://drive.google.com/uc?id=1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD"
output_path = "weights/best.pt"

os.makedirs("weights", exist_ok=True)
print("📦 Downloading best.pt from Google Drive...")
gdown.download(url, output_path, quiet=False)
print("✅ best.pt downloaded successfully.")
