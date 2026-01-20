import sys
import os
import shutil
from pathlib import Path
import subprocess

# Add current dir to sys.path
sys.path.append(os.getcwd())

APP_DIR = Path(os.getcwd())

def compile_latex_local(tex_path: Path, workdir: Path) -> tuple[bool, str, Path | None]:
    # Check for local tectonic.exe in APP_DIR first
    tectonic_cmd = "tectonic"
    if (APP_DIR / "tectonic.exe").exists():
        tectonic_cmd = str(APP_DIR / "tectonic.exe")

    cmd = [
        tectonic_cmd,
        "-X", "compile",
        tex_path.name
    ]
    
    try:
        p = subprocess.run(
            cmd,
            cwd=str(workdir),
            capture_output=True,
            text=True,
        )
        log = (p.stdout or "") + "\n" + (p.stderr or "")
        pdf_path = workdir / (tex_path.stem + ".pdf")
        success = (p.returncode == 0) and pdf_path.exists()
        return success, log, (pdf_path if success else None)
    except Exception as e:
        return False, f"Execution failed: {str(e)}", None

def test_custom_template():
    workdir = Path("test_custom_run")
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(exist_ok=True)
    
    # Copy the actual template
    src_template = Path("templates/mle_base.tex")
    dest_template = workdir / "test_mle.tex"
    shutil.copy2(src_template, dest_template)

    print(f"Testing compilation of {src_template}...")
    success, log, pdf_path = compile_latex_local(dest_template, workdir)
    
    if success:
        print(f"Success! PDF saved to {pdf_path}")
        print(f"PDF Size: {pdf_path.stat().st_size} bytes")
    else:
        print("Compilation Failed")
        try:
            print(log)
        except:
            print("Log printing failed")

if __name__ == "__main__":
    test_custom_template()
