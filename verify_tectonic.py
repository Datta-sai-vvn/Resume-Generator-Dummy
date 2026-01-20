import sys
import os
from pathlib import Path

# Add current dir to sys.path so we can import app
sys.path.append(os.getcwd())

from app import compile_latex

def test_compile():
    workdir = Path("test_run_tectonic")
    workdir.mkdir(exist_ok=True)
    
    tex_file = workdir / "test.tex"
    tex_file.write_text(r"""
\documentclass{article}
\begin{document}
Hello World from Tectonic!
\end{document}
""", encoding="utf-8")

    print(f"Testing Tectonic compilation of {tex_file}...")
    success, log, pdf_path = compile_latex(tex_file, workdir)
    
    if success:
        print(f"Success! PDF saved to {pdf_path}")
        print(f"PDF Size: {pdf_path.stat().st_size} bytes")
    else:
        print("Compilation Failed")
        try:
            print(log)
        except Exception:
            print("Could not print log due to encoding error")

if __name__ == "__main__":
    test_compile()
