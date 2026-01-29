import os
import re

import zipfile
import io
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
import pypdf
import platform
import tarfile
import urllib.request
import ssl
import stat


# =========================
# Setup
# =========================
load_dotenv()

# Get API Key (Local .env or Streamlit Secrets)
api_key = os.getenv("OPENAI_API_KEY")
if not api_key and "OPENAI_API_KEY" in st.secrets:
    api_key = st.secrets["OPENAI_API_KEY"]

if not api_key:
    st.error("Missing OpenAI API Key! Please set OPENAI_API_KEY in .env (local) or Streamlit Secrets (cloud).")
    st.stop()

client = OpenAI(api_key=api_key)

APP_DIR = Path(__file__).parent
PROMPTS_DIR = APP_DIR / "prompts"
TEMPLATES_DIR = APP_DIR / "templates"
RUNS_DIR = APP_DIR / "runs"
RESUMES_DIR = APP_DIR / "resumes"

def setup_tectonic_binary():
    """
    Ensures Tectonic binary is present for the current OS.
    Streamlit Cloud (Linux) needs the binary downloaded manually since pip fails.
    """
    system = platform.system().lower()
    binary_name = "tectonic.exe" if system == "windows" else "tectonic"
    binary_path = APP_DIR / binary_name

    if binary_path.exists():
        return # Already installed

    print(f"[{system}] Tectonic binary not found. Downloading...")
    
    # URLs for 0.15.0
    if system == "windows":
        url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.15.0/tectonic-0.15.0-x86_64-pc-windows-msvc.zip"
    else: # Linux (Streamlit Cloud is usually x86_64 Linux)
        url = "https://github.com/tectonic-typesetting/tectonic/releases/download/tectonic@0.15.0/tectonic-0.15.0-x86_64-unknown-linux-gnu.tar.gz"

    # Download
    print(f"Downloading from {url}...")
    try:
        # Unsafe context for speed/ease in this script (safe for public github release)
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        
        with urllib.request.urlopen(url, context=ctx) as response:
            data = response.read()
            
        # Extract
        if url.endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                zf.extractall(APP_DIR)
        else: # tar.gz
            with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
                tar.extractall(APP_DIR)
                
        # chmod +x for Linux
        if system != "windows" and binary_path.exists():
            st = os.stat(binary_path)
            os.chmod(binary_path, st.st_mode | stat.S_IEXEC)
            
        print("Tectonic binary installed successfully.")
        
    except Exception as e:
        print(f"Failed to install Tectonic: {e}")

# Run setup immediately
setup_tectonic_binary()


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
    
    # --- RESCUE MISSING MARKERS ---
    # Sometimes the model forgets the markers but returns valid LaTeX.
    # We try to inject them if missing.
    raw = rescue_missing_markers(raw)
    
    return raw.strip()


def rescue_missing_markers(text: str) -> str:
    """
    If the text incorrectly implements headers without markers, we wrap them.
    We look for standard LaTeX sections generated by the model.
    """
    # 1. Summary
    # Model usually outputs just text for summary, or \section{Professional Summary}
    # This is hard to guess if strictly just text. 
    # But usually it's the FIRST block.
    
    # Let's verify if markers exist
    has_sum = "%==== AUTOGEN_SUMMARY_START" in text
    has_proj = "%==== AUTOGEN_PROJECTS_START" in text
    has_skill = "%==== AUTOGEN_SKILLS_START" in text
    
    if has_sum and has_proj and has_skill:
        return text # All good
        
    # Heuristic Rescue
    # We assume order: Summary -> Projects -> Skills (or similar)
    # But actually prompts ask for Summary, Projects, Skills (order varies in prompt vs template)
    # In 'mle.txt', we ask for: 1) Summary 2) Projects 3) Skills
    

    # State tracking
    has_sum = "%==== AUTOGEN_SUMMARY_START" in text
    has_proj = "%==== AUTOGEN_PROJECTS_START" in text
    has_skill = "%==== AUTOGEN_SKILLS_START" in text
    
    if has_sum and has_proj and has_skill:
        return text # All good

    print("System Alert: Markers missing in model output. Attempting rescue...")
    
    # We will try to segment the text by looking for section headers (even if slightly wrong)
    # Common headers the model might generate:
    # Summary, Projects, Skills
    
    # Normalize to find split points
    # We look for \section{...} OR just standalone lines like "Projects" if they look like headers.
    # But LaTeX output usually implies \section.
    
    # Strategy:
    # 1. Find "Skills" section start.
    # 2. Find "Projects" section start.
    # 3. Everything before Projects is Summary.
    
    # Regex for headers
    # Matches: \section{Projects}, \section*{Projects}, or just **Projects**
    re_proj = re.compile(r'\\section\*?\{.*(Projects|Experience).*\}', re.IGNORECASE) 
    re_skill = re.compile(r'\\section\*?\{.*(Skills|Technical).*\}', re.IGNORECASE)
    
    # We iterate lines to find boundaries
    lines = text.split('\n')
    
    summary_lines = []
    proj_lines = []
    skill_lines = []
    
    current_mode = "summary" # optimize for Summary first
    
    for line in lines:
        row = line.strip()
        
        # Check for Skills change
        if re_skill.search(row):
            current_mode = "skills"
            continue # skip the header line itself (template has it)
            
        # Check for Projects change
        if re_proj.search(row):
            current_mode = "proj"
            continue # skip header
            
        # Append to current bucket
        if current_mode == "summary":
            summary_lines.append(line)
        elif current_mode == "proj":
            proj_lines.append(line)
        elif current_mode == "skills":
            skill_lines.append(line)
            
    # Now reconstruct with markers
    # We only wrap if we actually found something, otherwise empty block to avoid crashes
    
    res = ""
    
    # Summary
    res += "%==== AUTOGEN_SUMMARY_START\n"
    res += "\n".join(summary_lines).strip()
    res += "\n%==== AUTOGEN_SUMMARY_END\n\n"
    
    # Projects
    res += "%==== AUTOGEN_PROJECTS_START\n"
    res += "\n".join(proj_lines).strip()
    res += "\n%==== AUTOGEN_PROJECTS_END\n\n"

    # Skills
    res += "%==== AUTOGEN_SKILLS_START\n"
    res += "\n".join(skill_lines).strip()
    res += "\n%==== AUTOGEN_SKILLS_END\n"
    
    return res




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
    # Check for local binary in APP_DIR (Auto-downloaded by setup_tectonic_binary)
    # Windows: tectonic.exe, Linux: tectonic
    tectonic_cmd = "tectonic"
    
    local_bin_name = "tectonic.exe" if platform.system().lower() == "windows" else "tectonic"
    local_bin = APP_DIR / local_bin_name
    
    if local_bin.exists():
        tectonic_cmd = str(local_bin.resolve())

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
# New Helpers: Batch / Extraction
# =========================
def extract_company_name(jd_text: str) -> str:
    """
    Extracts company name from JD text using a cheap model call.
    Returns filesystem-safe string.
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Extract the Company Name from the Job Description. Return ONLY the company name. If unknown, return 'Unknown_Company'. Do not include Inc, LLC, etc unless necessary for distinction. Keep it short and filesystem safe (no special chars)."},
                {"role": "user", "content": jd_text[:2000]} # Send first 2k chars to save tokens
            ],
            temperature=0.0
        )
        name = resp.choices[0].message.content.strip()
        # Sanitize for filesystem
        name = re.sub(r"[^a-zA-Z0-9 _-]", "", name)
        name = name.strip().replace(" ", "_")
        return name or "Unknown_Company"
    except Exception:
        return "Unknown_Company"




def validate_project_bullets(block_text: str) -> tuple[bool, str]:
    """
    Validates Project bullets to ensure NO LINE wrapping.
    Hard limit: 88 characters per bullet.
    Also validates Header length to prevent stack spilling (approx 100 chars raw latex).
    """
    lines = block_text.split('\n')
    for line in lines:
        raw = line.strip()
        if not raw or raw.startswith('%') or raw.startswith(r'\section'):
            continue
            
        # Check Header Line (Project Title + Stack)
        # Format: \textbf{Name: Desc} | \textit{Stack} \\
        if raw.startswith(r'\textbf'):
             # Loose check for visual length. 
             # Raw latex is longer than visual, but 105 is a safe upper bound for single line.
             if len(raw) > 115: 
                 return False, f"Project Header too long ({len(raw)} chars). Prune Stack or Title to fit 1 line."

        # Only check actual item bullets
        if raw.startswith(r'\item'):
            # Text content is after \item
            content = raw.replace(r'\item', '').strip()
            if len(content) > 88:
                 return False, f"Project bullet too long ({len(content)} > 88 chars): '{content[:30]}...'"
    
    return True, "Valid"


def validate_skills_block(block_text: str) -> tuple[bool, str]:
    """
    Validates the Skills block for the new Single Paragraph format.
    Constraint: 
    - Max: 800 chars (approx 6-7 lines, safety buffer).
    - Min: 250 chars (approx 2-3 lines, to ensure density).
    """
    lines = block_text.split('\n')
    valid_lines = [l for l in lines if l.strip() and not l.strip().startswith('%') and not l.strip().startswith(r'\section')]
    
    if not valid_lines:
        return True, "Empty (pass)"

    # In paragraph mode, we expect usually 1 long line, or maybe minimal wrapping.
    # We validate the TOTAL content length mostly.
    total_len = sum(len(l) for l in valid_lines)
    
    if total_len > 900:
        return False, f"Skills too long ({total_len} > 900 chars). Prune low-priority items."
        
    if total_len < 250:
         return False, f"Skills too short ({total_len} < 250 chars). User wants 3-4 full lines. Add more JD keywords."

    return True, "Valid"


def enforce_skills_length(block_text: str, max_len: int = 900) -> str:
    """
    Simple truncation for paragraph mode.
    If total length > 900, plain cut at the last pipe/comma.
    """
    # Join all lines (except comments) to treat as one block
    lines = block_text.split('\n')
    content_lines = []
    headers = []
    
    for line in lines:
        raw = line.strip()
        if raw.startswith('%') or raw.startswith(r'\section'):
            headers.append(line)
        else:
            content_lines.append(line)
            
    full_text = " ".join(content_lines)
    
    if len(full_text) <= max_len:
        return block_text # No change needed
        
    # Truncate
    # Find a cut point (pipe or comma) near max_len
    cut_candidate = full_text[:max_len]
    last_pipe = cut_candidate.rfind('|')
    last_comma = cut_candidate.rfind(',')
    
    cut_idx = max(last_pipe, last_comma)
    if cut_idx == -1:
        cut_idx = max_len # Hard cut if no delimiters found
        
    final_text = full_text[:cut_idx]
    
    return "\n".join(headers + [final_text])



def redistribute_project_stacks(tex_content: str) -> str:
    """
    Parses the LaTeX content to:
    1. Identify Project Headers (Title | Stack).
    2. Limit Stack to top 4 technologies.
    3. Move 'overflow' technologies to the Skills section.
    """
    # 1. Extract Projects Block
    if "proj" not in AUTOGEN_BLOCKS: return tex_content
    p_start, p_end = AUTOGEN_BLOCKS["proj"]
    proj_block = extract_block(tex_content, p_start, p_end)
    if not proj_block: 
        return tex_content
        
    overflow_items = []

    def replace_stack(match):
        prefix = match.group(1) # \textbf{...} | \textit{
        stack_content = match.group(2)
        suffix = match.group(3) # }
        
        # Split by comma
        items = [s.strip() for s in stack_content.split(',')]
        if len(items) > 4:
            keep = items[:4]
            move = items[4:]
            overflow_items.extend(move)
            return prefix + ", ".join(keep) + suffix
        return match.group(0)

    # Regex to find the Header line: \textbf{...} | \textit{...}
    # We use non-greedy matching.
    # Note: LaTeX might have nested braces, but stack usually simple text.
    new_proj_block = re.sub(
        r'(\\textbf\{.+?\}\s*\|\s*\\textit\{)(.+?)(\})', 
        replace_stack, 
        proj_block,
        flags=re.MULTILINE
    )
    
    # Update Projects in main text
    # (Since we just have the block, we replace it)
    tex_content = replace_block(tex_content, p_start, p_end, new_proj_block)
    
    if not overflow_items:
        return tex_content

    # 2. Extract Skills Block to append overflow
    if "skills" not in AUTOGEN_BLOCKS: return tex_content
    s_start, s_end = AUTOGEN_BLOCKS["skills"]
    skills_block = extract_block(tex_content, s_start, s_end)
    
    # Identify existing skills to avoid dupes
    # Skills are | separated or comma separated handled loosely
    existing_text = skills_block.replace('\n', ' ')
    # Normalize
    existing_set = set()
    for token in re.split(r'[|,\,]', existing_text):
        clean = token.strip()
        if clean:
            existing_set.add(clean.lower())
            
    # Filter overflow
    to_add = []
    for item in overflow_items:
        if item.lower() not in existing_set:
            to_add.append(item)
            existing_set.add(item.lower()) # mark added
            
    if not to_add:
        return tex_content
        
    # Append to Skills
    # If skills block ends with newline/space, strip it slightly
    # We want to append " | Item1 | Item2"
    
    # Simple append strategy:
    # If skills block is non-empty, add " | " then items
    clean_skills = skills_block.strip()
    if clean_skills:
        # Check if it ends with a separator
        if clean_skills[-1] in ['|', ',']:
            new_skills_block = clean_skills + " " + " | ".join(to_add)
        else:
            new_skills_block = clean_skills + " | " + " | ".join(to_add)
    else:
        new_skills_block = " | ".join(to_add)
        
    # Replace Skills
    tex_content = replace_block(tex_content, s_start, s_end, new_skills_block)
    
    return tex_content


def run_resume_pipeline(
    jd_text: str,
    role: str,
    prompt: str,
    base_tex: str,
    run_dir: Path
) -> dict:
    """
    Core pipeline: Inputs -> LLM -> Patch -> Merge -> Compile -> PDF.
    Now includes AUTO-RETRY limit for Skills validation.
    """
    # 1. Save inputs
    (run_dir / "jd.txt").write_text(jd_text, encoding="utf-8")
    (run_dir / "prompt_used.txt").write_text(prompt, encoding="utf-8")
    (run_dir / "base_used.tex").write_text(base_tex, encoding="utf-8")

    # 2. Copy shared latex files
    copy_extra_latex_files(run_dir)

    # 3. Call GPT (With Retry Loop)
    max_retries = 3
    final_raw_out = ""
    final_patch_out = ""
    
    current_prompt = prompt
    
    for attempt in range(max_retries):
        try:
            # Generate
            raw_out = call_gpt(prompt=current_prompt, base_tex=base_tex, jd_text=jd_text, role=role)
            patch_out = sanitize_model_output_to_tex(raw_out)
            
            # --- NEW: Redistribute Stack Overflow to Skills ---
            patch_out = redistribute_project_stacks(patch_out)
            # --------------------------------------------------
            
            # Extract Skills to validate
            # We use the global AUTOGEN_BLOCKS definition
            s_start, s_end = AUTOGEN_BLOCKS["skills"]
            skills_block = extract_block(patch_out, s_start, s_end)
            
            # Validate
            is_valid_skills, msg_skills = validate_skills_block(skills_block)
            
            # Extract Projects to validate
            p_start, p_end = AUTOGEN_BLOCKS["proj"]
            proj_block = extract_block(patch_out, p_start, p_end)
            is_valid_proj, msg_proj = validate_project_bullets(proj_block)
            
            if is_valid_skills and is_valid_proj:
                final_raw_out = raw_out
                final_patch_out = patch_out
                break # Success
            else:
                # Validation failed
                final_msg = ""
                if not is_valid_skills: final_msg += f"Skills Error: {msg_skills}. "
                if not is_valid_proj: final_msg += f"Projects Error: {msg_proj}. "
                
                print(f"Attempt {attempt+1} failed validation: {final_msg}")
                msg = final_msg # For retry prompt
                final_raw_out = raw_out # Keep last result just in case
                final_patch_out = patch_out
                
                # Append instruction to prompt for next turn
                if attempt < max_retries - 1:
                    current_prompt += f"\n\nSYSTEM ALERT: Previous output FAILED validation. {msg}. You MUST prune skills to be shorter. REGENERATE."
        
        except Exception as e:
            # If extraction failed or other error, retry
            print(f"Attempt {attempt+1} crashed: {e}")
            final_raw_out = raw_out if 'raw_out' in locals() else ""
            final_patch_out = patch_out if 'patch_out' in locals() else ""
            if attempt < max_retries - 1:
                current_prompt += f"\n\nSYSTEM ALERT: Previous output was malformed. {str(e)}. REGENERATE strictly following the FORMAT."

    # Loop finished.
    # Checks if we have a valid output? If not, perform HARD FALLBACK truncation.
    # Is final_patch_out valid?
    # We need to re-extract to check because loop variable scope logic above
    # might leave us with an invalid 'final_patch_out'.
    
    # Let's try to extract and force-fix the skills block in 'final_patch_out'
    try:
        s_start, s_end = AUTOGEN_BLOCKS["skills"]
        # Extract
        bad_skills = extract_block(final_patch_out, s_start, s_end)
        # Check validity one last time
        is_ok, _ = validate_skills_block(bad_skills)
        
        if not is_ok:
            # FORCE TRUNCATE
            fixed_skills = enforce_skills_length(bad_skills, max_len=900)
            # Replace in the patch
            final_patch_out = replace_block(final_patch_out, s_start, s_end, fixed_skills)
            
    except Exception as e:
        print(f"Fallback truncation failed: {e}")
        # Proceed with what we have
    
            
    (run_dir / "model_raw_output.txt").write_text(final_raw_out or "", encoding="utf-8")
    (run_dir / "model_patch_output.txt").write_text(final_patch_out, encoding="utf-8")

    # 5. Merge
    try:
        final_tex, extracted_blocks = merge_autogen_blocks(base_tex, final_patch_out)
    except ValueError as e:
        return {
            "success": False,
            "error": f"Patch merge failed: {e}",
            "patch_out": final_patch_out,
            "raw_out": final_raw_out
        }

    # 6. Write final tex
    out_tex_path = run_dir / "resume.tex"
    out_tex_path.write_text(final_tex, encoding="utf-8")

    # 7. Compile
    ok, log, pdf_path = compile_latex(out_tex_path, run_dir)

    # 8. POST-PROCESS: Enforce Single Page (Hard Truncation)
    page_count = 0
    if ok and pdf_path and pdf_path.exists():
        try:
            reader = pypdf.PdfReader(str(pdf_path))
            page_count = len(reader.pages)
            if page_count > 1:
                # Force truncate to 1 page
                writer = pypdf.PdfWriter()
                writer.add_page(reader.pages[0])
                temp_output = run_dir / "resume_truncated.pdf"
                with open(temp_output, "wb") as f:
                    writer.write(f)
                
                # Replace original with truncated
                shutil.move(str(temp_output), str(pdf_path))
                page_count = 1
        except Exception as e:
            print(f"Truncation error: {e}")
            # If reading fails, rely on original page_count or just proceed
            pass

    return {
        "success": ok,
        "log": log,
        "pdf_path": pdf_path,
        "page_count": page_count,
        "final_tex": final_tex,
        "extracted_blocks": extracted_blocks,
        "patch_out": final_patch_out,
        "raw_out": final_raw_out
    }


# =========================
# Streamlit UI
# =========================
# =========================
# UI Logic
# =========================
st.set_page_config(page_title="Resume Fine-Tuner", layout="wide")
ensure_dirs()

st.title("Resume Fine-Tuner")

with st.sidebar:
    st.header("Settings")
    mode = st.radio("Mode", ["Single JD", "Batch JD (1-15)"])
    st.markdown("---")
    
    role = st.selectbox("Job Type", ["MLE", "SDE", "AV"])
    st.caption("Model: gpt-4.1-mini")
    
    # Common settings
    prompt_path = ROLE_TO_PROMPT[role]
    base_tex_path = ROLE_TO_BASE_TEX[role]

    if mode == "Single JD":
        run_name = st.text_input("Run name (optional)", value="")
        show_debug = st.checkbox("Show debug outputs", value=True)
        generate_btn = st.button("Generate Resume", type="primary")

# -------------------------
# SINGLE JD MODE
# -------------------------
if mode == "Single JD":
    col1, col2 = st.columns(2)
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
        if not prompt_path.exists() or not base_tex_path.exists():
            st.error("Missing template/prompt files.")
            st.stop()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_run = re.sub(r"[^a-zA-Z0-9_-]+", "_", run_name.strip()) if run_name.strip() else ""
        run_dir = RUNS_DIR / f"{ts}_{role}{('_' + safe_run) if safe_run else ''}"
        run_dir.mkdir(parents=True, exist_ok=True)

        with st.spinner("Processing..."):
            res = run_resume_pipeline(
                jd_text=jd_text,
                role=role,
                prompt=read_text(prompt_path),
                base_tex=read_text(base_tex_path),
                run_dir=run_dir
            )

        st.subheader("Result")
        if res["success"] and res["pdf_path"]:
            st.success("PDF generated successfully ✅")
            
            # Save copy
            saved_pdf_path = save_final_pdf(res["pdf_path"], role, ts)
            
            # Downloads
            pdf_bytes = res["pdf_path"].read_bytes()
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name=f"{role}_resume.pdf",
                mime="application/pdf",
            )
            st.download_button(
                label="Download LaTeX (.tex)",
                data=res["final_tex"].encode("utf-8"),
                file_name=f"{role}_resume.tex",
                mime="text/plain",
            )
            st.write(f"Final PDF saved to: {saved_pdf_path}")
        
        else:
            st.error("Generation failed ❌")
            if "error" in res:
                st.error(res["error"])
            st.write("Log:")
            st.code(res.get("log", ""), language="text")

        if show_debug:
            st.subheader("Debug Outputs")
            if "patch_out" in res:
                st.write("Model patch output:")
                st.code(res["patch_out"], language="latex")
            if "extracted_blocks" in res:
                st.write("Extracted blocks:")
                for k in ["summary", "skills", "proj"]:
                    st.caption(k)
                    st.code(res["extracted_blocks"].get(k, ""), language="latex")
            if "final_tex" in res:
                st.write("Final merged LaTeX:")
                st.code(res["final_tex"], language="latex")


# -------------------------
# BATCH JD MODE
# -------------------------
else: # Batch JD
    st.subheader("Batch Processing Mode")
    st.info("Enter multiple Job Descriptions below (1 Box = 1 JD). NO parsing/splitting will be done.")

    # State management for dynamic boxes
    if "batch_jd_count" not in st.session_state:
        st.session_state.batch_jd_count = 1

    def add_jd_box():
        if st.session_state.batch_jd_count < 15:
            st.session_state.batch_jd_count += 1

    # Dynamic Inputs
    jd_texts = []
    for i in range(st.session_state.batch_jd_count):
        val = st.text_area(f"Job Description #{i+1}", height=200, key=f"batch_jd_{i}")
        jd_texts.append(val)

    # Add Button
    if st.session_state.batch_jd_count < 15:
        st.button("Add JD", on_click=add_jd_box)
    else:
        st.caption("Maximum 15 JDs reached.")

    st.markdown("---")

    # Filter processed inputs
    valid_jds = [txt for txt in jd_texts if txt.strip()]

    if st.button(f"Generate Batch ({len(valid_jds)} JDs)", type="primary"):
        if not valid_jds:
            st.error("Please fill at least one JD box.")
            st.stop()
            
        # Container for ZIP creation
        zip_buffer = io.BytesIO()
        
        # Progress tracking
        prog_bar = st.progress(0.0)
        status_box = st.empty()
        
        success_count = 0
        failures = []
        
        prompt_content = read_text(prompt_path)
        base_tex_content = read_text(base_tex_path)
        
        ts_batch = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            
            for i, current_jd_text in enumerate(valid_jds):
                # 1. Update status
                display_num = i + 1
                status_box.write(f"Processing JD #{display_num}/{len(valid_jds)}...")
                prog_bar.progress(i / len(valid_jds))
                
                try:
                    # 2. Extract company name
                    company_name = extract_company_name(current_jd_text)
                    
                    # 3. Create run sub-folder
                    # Ensure unique folder even if same company appears twice
                    run_id = f"{ts_batch}_batch_{i}_{company_name}"
                    run_dir = RUNS_DIR / run_id
                    run_dir.mkdir(parents=True, exist_ok=True)
                    
                    # 4. Run pipeline
                    res = run_resume_pipeline(
                        jd_text=current_jd_text,
                        role=role,
                        prompt=prompt_content,
                        base_tex=base_tex_content,
                        run_dir=run_dir
                    )
                    
                    if res["success"] and res["pdf_path"]:
                        # 5. Add to ZIP
                        # Structure: <Company_Name>/Sai Resume.pdf
                        # Handle duplicate company names in ZIP by appending index if needed?
                        # User spec: "Folder name = Company Name". If duplicates, they might overwrite or merge.
                        # We will assume unique company JDs or allow overwrite if zipfile handles it (it creates duplicates usually).
                        # To be safe and clean, strictly follow "Company_Name/Sai Resume.pdf"
                        # If user puts same company twice, they get two folders? No, same folder. 
                        # We'll just stick to spec.
                        archive_name = f"{company_name}/Sai Resume.pdf"
                        zf.write(res["pdf_path"], arcname=archive_name)
                        success_count += 1
                    else:
                        fail_reason = res.get("error", "Compilation failed")
                        # Include log snippet for debugging
                        log_snippet = res.get("log", "")[:300].replace("\n", " ")
                        failures.append(f"JD #{display_num} ({company_name}): {fail_reason} | Log: {log_snippet}...")

                except Exception as e:
                    failures.append(f"JD #{display_num}: Unexpected error {str(e)}")
                    
        prog_bar.progress(1.0)
        status_box.success("Batch processing complete!")
        
        # Summary
        st.write(f"**Processed:** {len(valid_jds)} | **Success:** {success_count} | **Failed:** {len(failures)}")
        
        if failures:
            st.error("Failures:")
            for f in failures:
                st.write(f"- {f}")
                
        if success_count > 0:
            st.download_button(
                label="Download All Resumes (ZIP)",
                data=zip_buffer.getvalue(),
                file_name=f"Batch_Resumes_{ts_batch}.zip",
                mime="application/zip"
            )
