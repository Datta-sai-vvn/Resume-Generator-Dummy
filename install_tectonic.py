import urllib.request
import zipfile
import os
import ssl

def install_tectonic():
    url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.15.0/tectonic-0.15.0-x86_64-pc-windows-msvc.zip"
    zip_path = "tectonic.zip"
    
    # Bypass SSL errors if any (uncommon but safe for this specific retrieval)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    
    print(f"Downloading {url}...")
    with urllib.request.urlopen(url, context=ctx) as response, open(zip_path, 'wb') as out_file:
        out_file.write(response.read())
        
    print("Download complete. Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
        
    print("Extraction complete. Checking binary...")
    if os.path.exists("tectonic.exe"):
        print("tectonic.exe found!")
    else:
        print("Error: tectonic.exe not found after extraction.")

if __name__ == "__main__":
    install_tectonic()
