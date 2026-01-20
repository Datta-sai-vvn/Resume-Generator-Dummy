import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())
from app import compile_latex

def test_fix():
    workdir = Path("test_run_fix")
    workdir.mkdir(exist_ok=True)
    
    # Create a minimal test tex file mimicking the start of the template
    tex_content = r"""
\documentclass[letterpaper,11pt]{article}
\usepackage{latexsym}
\usepackage[empty]{fullpage}
\usepackage{titlesec}
\usepackage{marvosym}
\usepackage[usenames,dvipsnames]{color}
\usepackage{verbatim}
\usepackage{enumitem}
\usepackage[hidelinks]{hyperref}
\usepackage{fancyhdr}
\usepackage[english]{babel}
\usepackage{tabularx}
% \input{glyphtounicode} % REMOVED for Tectonic

\begin{document}
Hello World!
\end{document}
"""
    tex_file = workdir / "test_fix.tex"
    tex_file.write_text(tex_content, encoding="utf-8")

    print(f"Testing compilation of patched {tex_file}...")
    success, log, pdf_path = compile_latex(tex_file, workdir)
    
    if success:
        print(f"Success! PDF saved to {pdf_path}")
    else:
        print("Compilation Failed")
        try:
            print(log)
        except:
            print("Log printing failed")

if __name__ == "__main__":
    test_fix()
