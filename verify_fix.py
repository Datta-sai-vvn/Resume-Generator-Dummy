
import re

AUTOGEN_BLOCKS = {
    "summary": ("%==== AUTOGEN_SUMMARY_START", "%==== AUTOGEN_SUMMARY_END"),
    "skills": ("%==== AUTOGEN_SKILLS_START", "%==== AUTOGEN_SKILLS_END"),
    "proj": ("%==== AUTOGEN_PROJECTS_START", "%==== AUTOGEN_PROJECTS_END"),
}

def extract_block(text: str, start: str, end: str) -> str:
    print(f"Searching for {start} ... {end}")
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    m = pattern.search(text)
    if not m:
        return None
    return m.group(0).strip()

def test_extraction(mock_output):
    print(f"--- Testing Output ---\n{mock_output}\n----------------------")
    missing = []
    for key, (start, end) in AUTOGEN_BLOCKS.items():
        if not extract_block(mock_output, start, end):
            missing.append(key)
    
    if missing:
        print(f"FAILED: Missing blocks: {missing}")
    else:
        print("SUCCESS: All blocks found.")

# Case 1: Perfect Output
perfect = """
%==== AUTOGEN_SUMMARY_START
Summary Content
%==== AUTOGEN_SUMMARY_END

%==== AUTOGEN_PROJECTS_START
Projects Content
%==== AUTOGEN_PROJECTS_END

%==== AUTOGEN_SKILLS_START
Skills Content
%==== AUTOGEN_SKILLS_END
"""

# Case 2: Missing Markers (Common Failure)
bad = """
\section{Summary}
Summary Content

\section{Projects}
Projects Content

\section{Skills}
Skills Content
"""


from app import rescue_missing_markers

# ... (Previous code)

print("Running Tests...")
print("--- Standard Extraction Tests ---")
test_extraction(perfect)

print("\n--- Rescue Logic Test ---")
rescued_bad = rescue_missing_markers(bad)
print(f"Rescued Output:\n{rescued_bad}")
test_extraction(rescued_bad)

