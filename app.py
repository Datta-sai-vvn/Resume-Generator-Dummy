import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


# =========================
# Setup
# =========================
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

APP_DIR = Path(__file__).parent
PROMPTS_DIR = APP_DIR / "prompts"
TEMPLATES_DIR = APP_DIR / "templates"
RUNS_DIR = APP_DIR / "runs"
RESUMES_DIR = APP_DIR / "resumes"


# Role -> prompt file
ROLE_TO_PROMPT = {
    "MLE": PROMPTS_DIR / "mle.txt",
    "SDE": PROMPTS_DIR / "sde.txt",
    "AV":  PROMPTS_DIR / "av.txt",
}

# Role -> base LaTeX resume
ROLE_TO_BASE_TEX = {
    "MLE": TEMPLATES_DIR / "mle_base.tex",
    "SDE": TEMPLATES_DIR / "sde_base.tex",
    "AV":  TEMPLATES_DIR / "av_base.tex",
}

# Optional: shared LaTeX class/style files your templates may depend on
# Put these files inside templates/ if you use them.
EXTRA_LATEX_FILES = [
    "resume.cls",
    "custom.sty",
]

# AUTOGEN block markers (must match your base .tex and prompt output)
AUTOGEN_BLOCKS = {
    "summary": ("%==== AUTOGEN_SUMMARY_START", "%==== AUTOGEN_SUMMARY_END"),
    "skills": ("%==== AUTOGEN_SKILLS_START", "%==== AUTOGEN_SKILLS_END"),
    "proj": ("%==== AUTOGEN_PROJECTS_START", "%==== AUTOGEN_PROJECTS_END"),
}


# =========================
# Helpers: IO / LaTeX / Merge
# =========================
def save_final_pdf(pdf_path: Path, role: str, run_ts: str) -> Path:
    """
    Copies the final PDF into resumes/<ROLE>/ with a clean filename.
    """
    role_dir = RESUMES_DIR / role
    role_dir.mkdir(parents=True, exist_ok=True)

    # You can customize this naming
    filename = f"Varun_{role}_{run_ts}.pdf"
    dest_path = role_dir / filename

    shutil.copy2(pdf_path, dest_path)
    return dest_path

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def ensure_dirs():
    RUNS_DIR.mkdir(exist_ok=True, parents=True)



def sanitize_model_output_to_tex(raw: str) -> str:
    """
    Removes common wrappers that break parsing/compilation (```latex ...```).
    """
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:latex)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def extract_block(text: str, start: str, end: str) -> str:
    """
    Extracts a block including the start/end marker lines.
    """
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    m = pattern.search(text)
    if not m:
        raise ValueError(f"Missing block markers in model output: {start} ... {end}")
    return m.group(0).strip()


def replace_block(base_tex: str, start: str, end: str, new_block: str) -> str:
    """
    Replaces the block in base_tex delimited by start/end markers with new_block.
    Uses a function replacement so LaTeX backslashes are not treated as regex escapes.
    """
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    if not pattern.search(base_tex):
        raise ValueError(f"Missing block markers in base template: {start} ... {end}")

    replacement = new_block.strip()
    return pattern.sub(lambda _: replacement, base_tex, count=1)



def merge_autogen_blocks(base_tex: str, model_out: str) -> tuple[str, dict]:
    """
    Extracts 4 AUTOGEN blocks from model output and merges them into base_tex.
    Returns (merged_tex, extracted_blocks_dict).
    """
    extracted = {}
    for key, (start, end) in AUTOGEN_BLOCKS.items():
        extracted[key] = extract_block(model_out, start, end)

    merged = base_tex
    for key, (start, end) in AUTOGEN_BLOCKS.items():
        merged = replace_block(merged, start, end, extracted[key])

    return merged, extracted


def copy_extra_latex_files(run_dir: Path):
    """
    Copies any shared .cls/.sty files into the run directory so latexmk can find them.
    """
    for fname in EXTRA_LATEX_FILES:
        src = TEMPLATES_DIR / fname
        if src.exists():
            shutil.copy2(src, run_dir / fname)


# =========================
# Tectonic Compilation
# =========================
def compile_latex(tex_path: Path, workdir: Path) -> tuple[bool, str, Path | None]:
    """
    Compiles using Tectonic via subprocess. Returns (success, log, pdf_path)
    Tectonic automatically downloads packages on first run.
    """
    # Tectonic command: tectonic -X compile <file>
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
    except FileNotFoundError:
        return False, "tectonic not found in PATH. Ensure it is installed (pip install tectonic).", None
    except Exception as e:
        return False, f"Tectonic execution failed: {str(e)}", None


# =========================
# LLM Call
# =========================
def call_gpt(prompt: str, base_tex: str, jd_text: str, role: str) -> str:
    """
    IMPORTANT: Model must return ONLY the 4 AUTOGEN blocks (patches), not a full document.
    """
    system = (
        "You are an expert ATS resume editor. "
        "Return ONLY the 3 LaTeX AUTOGEN blocks with their markers. "
        "No markdown, no explanations, no other LaTeX."
    )


    user = f"""
ROLE: {role}

<<MASTER_PROMPT>>
{prompt}
<</MASTER_PROMPT>>

<<BASE_LATEX>>
{base_tex}
<</BASE_LATEX>>

<<JOB_DESCRIPTION>>
{jd_text}
<</JOB_DESCRIPTION>>

HARD OUTPUT RULES:
- Output ONLY these 3 marker-delimited blocks (and include the marker lines):
  1) %==== AUTOGEN_SUMMARY_START ... %==== AUTOGEN_SUMMARY_END
  2) %==== AUTOGEN_SKILLS_START ... %==== AUTOGEN_SKILLS_END
  3) %==== AUTOGEN_PROJECTS_START ... %==== AUTOGEN_PROJECTS_END
- Do NOT output the full LaTeX document.
- Do NOT output any other text.
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.output_text


# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="Resume Fine-Tuner", layout="wide")
ensure_dirs()

st.title("Resume Fine-Tuner (LaTeX patches → merge → PDF)")

with st.sidebar:
    st.header("Settings")
    role = st.selectbox("Job Type", ["MLE", "SDE", "AV"])
    st.caption("Model: gpt-4.1-mini (change in code if needed)")
    run_name = st.text_input("Run name (optional)", value="")
    show_debug = st.checkbox("Show debug outputs", value=True)
    generate_btn = st.button("Generate Resume", type="primary")

col1, col2 = st.columns(2)

base_tex_path = ROLE_TO_BASE_TEX[role]
prompt_path = ROLE_TO_PROMPT[role]

with col1:
    st.subheader("Inputs")
    jd_text = st.text_area("Paste Job Description", height=360)
    prompt_preview = st.checkbox("Show selected prompt", value=False)
    base_preview = st.checkbox("Show selected base LaTeX", value=False)

with col2:
    st.subheader("Template / Prompt Files")
    st.write("Prompt file:", str(prompt_path.relative_to(APP_DIR)))
    st.write("Base LaTeX:", str(base_tex_path.relative_to(APP_DIR)))

if prompt_preview and prompt_path.exists():
    st.code(read_text(prompt_path), language="text")

if base_preview and base_tex_path.exists():
    st.code(read_text(base_tex_path), language="latex")

if generate_btn:
    if not jd_text.strip():
        st.error("Paste a Job Description first.")
        st.stop()

    if not prompt_path.exists():
        st.error(f"Missing prompt file: {prompt_path}")
        st.stop()

    if not base_tex_path.exists():
        st.error(f"Missing base LaTeX file: {base_tex_path}")
        st.stop()

    base_tex = read_text(base_tex_path)
    prompt = read_text(prompt_path)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_run = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_name.strip()) if run_name.strip() else ""
    run_dir = RUNS_DIR / f"{ts}_{role}{('_' + safe_run) if safe_run else ''}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save inputs for reproducibility
    (run_dir / "jd.txt").write_text(jd_text, encoding="utf-8")
    (run_dir / "prompt_used.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "base_used.tex").write_text(base_tex, encoding="utf-8")

    # Copy shared latex files if used
    copy_extra_latex_files(run_dir)

    with st.spinner("Calling GPT (patch generation)..."):
        raw_out = call_gpt(prompt=prompt, base_tex=base_tex, jd_text=jd_text, role=role)

    (run_dir / "model_raw_output.txt").write_text(raw_out or "", encoding="utf-8")

    patch_out = sanitize_model_output_to_tex(raw_out)
    (run_dir / "model_patch_output.txt").write_text(patch_out, encoding="utf-8")

    # Merge patches into the full base template
    try:
        final_tex, extracted_blocks = merge_autogen_blocks(base_tex, patch_out)
    except ValueError as e:
        st.error(f"Patch merge failed: {e}")
        if show_debug:
            st.write("Model patch output:")
            st.code(patch_out, language="latex")
        st.stop()

    # Write final merged resume tex
    out_tex_path = run_dir / "resume.tex"
    out_tex_path.write_text(final_tex, encoding="utf-8")

    with st.spinner("Compiling LaTeX to PDF..."):
        ok, log, pdf_path = compile_latex(out_tex_path, run_dir)

    st.subheader("Result")

    if ok and pdf_path:
        st.success("PDF generated successfully ✅")

        # Save a clean copy into resumes/<ROLE>/
        saved_pdf_path = save_final_pdf(
            pdf_path=pdf_path,
            role=role,
            run_ts=ts,
        )

        # Download from runs/
        pdf_bytes = pdf_path.read_bytes()
        st.download_button(
            label="Download PDF",
            data=pdf_bytes,
            file_name=f"{role}_resume.pdf",
            mime="application/pdf",
        )

        st.download_button(
            label="Download LaTeX (.tex)",
            data=final_tex.encode("utf-8"),
            file_name=f"{role}_resume.tex",
            mime="text/plain",
        )

        st.write(f"Run artifacts saved in: {run_dir}")
        st.write(f"Final PDF saved to: {saved_pdf_path}")




    else:
        st.error("LaTeX compilation failed ❌")
        st.write("Compile log:")
        st.code(log, language="text")

    if show_debug:
        st.subheader("Debug Outputs")

        st.write("Model patch output (4 blocks):")
        st.code(patch_out, language="latex")

        st.write("Extracted blocks:")
        for k in ["summary", "skills", "exp", "proj"]:
            st.caption(k)
            st.code(extracted_blocks.get(k, ""), language="latex")

        st.write("Final merged LaTeX (compiled input):")
        st.code(final_tex, language="latex")
