"""
Altermagnet Screener — Streamlit app (with live terminal view)
------------------------------------------------------------------
Upload VASP structure files, and this app drives `amcheck` interactively
(answering its spin prompts) across every u/d spin combination for
magnetic elements, flagging any structure where at least one combination
reports `Altermagnet? True`.

This version streams the raw amcheck subprocess output to an always-visible
terminal panel in real time, so you can see exactly what amcheck is doing
and diagnose parsing issues if the automated responses don't match.
"""

import itertools
import os
import re
import shutil
import subprocess
import tempfile
import zipfile

import streamlit as st

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

SPECIAL_ELEMENTS = {"Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Mo", "Ru"}
MAX_N = 26
ATOM_COUNT_HARD_LIMIT = 24

ELEMENT_RE = re.compile(r"Orbit of (\w+) atoms at positions:")
ATOM_RE = re.compile(r"\d+ \(\d+\) \[\s*[-?\d.eE]+\s+[-?\d.eE]+\s+[-?\d.eE]+\s*\]")
ALTERMAGNET_RE = re.compile(r"Altermagnet\?\s*(True|False)")
SPIN_PROMPT = "Type spin (u, U, d, D, n, N, nn or NN) for each of them"
PRIMITIVE_CELL_PROMPT = "Do you want to use it instead? (Y/n)"

SUBPROCESS_TIMEOUT_SECONDS = 120  # per amcheck invocation, safety net


# --------------------------------------------------------------------------
# amcheck availability
# --------------------------------------------------------------------------

def check_amcheck_installed() -> bool:
    try:
        subprocess.run(
            ["amcheck", "--help"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
            timeout=15,
        )
        return True
    except Exception:
        return False


# --------------------------------------------------------------------------
# u/d sequence generation
# --------------------------------------------------------------------------

def flip(seq: str) -> str:
    return "".join("d" if c == "u" else "u" for c in seq)


def generate_combinations(n: int) -> list:
    if n % 2 != 0:
        return []
    result, seen = [], set()

    def backtrack(path, u_count, d_count):
        if len(path) == n:
            seq = "".join(path)
            flipped = flip(seq)
            if seq not in seen and flipped not in seen:
                seen.add(seq)
                result.append(" ".join(seq))
            return
        if u_count < n // 2:
            backtrack(path + ["u"], u_count + 1, d_count)
        if d_count < n // 2:
            backtrack(path + ["d"], u_count, d_count + 1)

    backtrack([], 0, 0)
    result.sort()
    return result


@st.cache_data(show_spinner=False)
def build_sequence_map(max_n: int) -> dict:
    return {n: generate_combinations(n) for n in range(2, max_n + 1, 2)}


# --------------------------------------------------------------------------
# Core subprocess runner with live streaming
# --------------------------------------------------------------------------

class AmcheckRunResult:
    def __init__(self):
        self.lines = []
        self.returncode = None
        self.timed_out = False
        self.exception = None


def _run_amcheck_process(vasp_file_path, respond_fn, stream_cb=None):
    """
    Runs `amcheck <file>`, reading stdout line by line. For each line,
    calls respond_fn(line) -> Optional[str]; if it returns a string,
    that string + newline is written to the process's stdin.
    stream_cb(line) is called for every line, purely for live display.
    Returns an AmcheckRunResult.
    """
    result = AmcheckRunResult()
    cmd = ["amcheck", vasp_file_path]

    if stream_cb:
        stream_cb(f"$ {' '.join(cmd)}")

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except FileNotFoundError as e:
        result.exception = f"amcheck executable not found: {e}"
        if stream_cb:
            stream_cb(f"[ERROR] {result.exception}")
        return result
    except Exception as e:
        result.exception = str(e)
        if stream_cb:
            stream_cb(f"[ERROR] {result.exception}")
        return result

    try:
        for line in proc.stdout:
            line = line.rstrip("\n")
            result.lines.append(line)
            if stream_cb:
                stream_cb(line)

            if PRIMITIVE_CELL_PROMPT in line:
                proc.stdin.write("Y\n")
                proc.stdin.flush()
                if stream_cb:
                    stream_cb("  > sent: Y")

            response = respond_fn(line)
            if response is not None:
                proc.stdin.write(response + "\n")
                proc.stdin.flush()
                result.lines.append(f"Sending response: {response}")
                if stream_cb:
                    stream_cb(f"  > sent: {response}")

        proc.stdin.close()
        proc.wait(timeout=SUBPROCESS_TIMEOUT_SECONDS)
        result.returncode = proc.returncode
    except subprocess.TimeoutExpired:
        proc.kill()
        result.timed_out = True
        if stream_cb:
            stream_cb("[ERROR] amcheck timed out and was killed")
    except Exception as e:
        result.exception = str(e)
        if stream_cb:
            stream_cb(f"[ERROR] {e}")

    if stream_cb:
        stream_cb(f"[process exited with code {result.returncode}]")

    return result


def run_nn_amcheck(vasp_file_path: str, stream_cb=None):
    """Discovery pass: answers every spin prompt with a default response
    (alternating u/d for special elements, 'nn' otherwise), purely to
    find out which elements are present and how many atoms each has."""
    element = None
    atom_count = 0
    element_array, atom_count_array = [], []

    def respond(line):
        nonlocal element, atom_count
        m = ELEMENT_RE.search(line)
        if m:
            element = m.group(1)
            atom_count = 0

        if ATOM_RE.search(line):
            atom_count += 1

        if SPIN_PROMPT in line:
            element_array.append(element)
            atom_count_array.append(atom_count)
            if element in SPECIAL_ELEMENTS:
                options = ["u", "d"]
                resp = [options[i % 2] for i in range(atom_count)]
            else:
                resp = ["nn"]
            return " ".join(resp)
        return None

    run_result = _run_amcheck_process(vasp_file_path, respond, stream_cb)
    return element_array, atom_count_array, run_result


def run_amcheck(vasp_file_path: str, input_array: list, stream_cb=None):
    """Runs amcheck once, answering each spin prompt with the pre-chosen
    combination in input_array (one entry per element, in order)."""
    index_holder = {"i": 0}
    altermagnet_result = {"value": None}

    def respond(line):
        m = ALTERMAGNET_RE.search(line)
        if m:
            altermagnet_result["value"] = m.group(1)

        if SPIN_PROMPT in line:
            i = index_holder["i"]
            if i >= len(input_array):
                return "nn"  # safety fallback, shouldn't normally happen
            resp = input_array[i]
            index_holder["i"] += 1
            return resp
        return None

    run_result = _run_amcheck_process(vasp_file_path, respond, stream_cb)
    return altermagnet_result["value"], run_result


def generate_input_combinations(vasp_file_path, arr, stream_cb=None, progress_cb=None):
    total = 0
    true_count = 0
    errors = 0
    combos = list(itertools.product(*arr)) if arr else [()]

    for combo in combos:
        total += 1
        output, run_result = run_amcheck(vasp_file_path, list(combo), stream_cb)
        if output == "True":
            true_count += 1
        if run_result.exception or run_result.timed_out or (
            run_result.returncode not in (0, None) and output is None
        ):
            errors += 1
        if progress_cb:
            progress_cb(total, len(combos), true_count, errors)

    return total, true_count, errors


def process_file(vasp_file_path, true_files_dir, sequence_map, stream_cb=None, progress_cb=None):
    if stream_cb:
        stream_cb(f"===== Discovery pass for {os.path.basename(vasp_file_path)} =====")

    element_array, atom_count_array, nn_run_result = run_nn_amcheck(vasp_file_path, stream_cb)

    if not element_array:
        reason = (
            "amcheck produced no recognizable element/atom output. "
            "Check the terminal panel above — either the file format wasn't "
            "recognized, or amcheck's prompts don't match what this app expects."
        )
        if nn_run_result.exception:
            reason += f" Exception: {nn_run_result.exception}"
        if nn_run_result.timed_out:
            reason += " (process timed out)"
        return {
            "File": os.path.basename(vasp_file_path),
            "Status": "Error",
            "Reason": reason,
            "Combinations tried": 0,
            "Altermagnet? True count": 0,
            "Flagged": False,
        }

    element_input_array = []
    for element, atom_count in zip(element_array, atom_count_array):
        if element in SPECIAL_ELEMENTS:
            if atom_count > ATOM_COUNT_HARD_LIMIT:
                msg = "Atom Count is Greater than 24. Try more combinations."
                return {
                    "File": os.path.basename(vasp_file_path),
                    "Status": "Skipped",
                    "Reason": msg,
                    "Combinations tried": 0,
                    "Altermagnet? True count": 0,
                    "Flagged": False,
                }
            element_input_array.append(sequence_map.get(atom_count, []))
        else:
            element_input_array.append(["nn"])

    if stream_cb:
        stream_cb(
            f"===== Running {len(list(itertools.product(*element_input_array))) if element_input_array else 1} "
            f"combination(s) for {os.path.basename(vasp_file_path)} ====="
        )

    total, true_count, errors = generate_input_combinations(
        vasp_file_path, element_input_array, stream_cb, progress_cb
    )

    result = {
        "File": os.path.basename(vasp_file_path),
        "Status": "Done" if errors == 0 else f"Done ({errors} run errors)",
        "Reason": "",
        "Combinations tried": total,
        "Altermagnet? True count": true_count,
        "Flagged": true_count > 0,
    }

    if true_count > 0:
        dest = os.path.join(true_files_dir, os.path.basename(vasp_file_path))
        shutil.copy(vasp_file_path, dest)

    return result


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------

st.set_page_config(page_title="Altermagnet Screener", page_icon="🧲", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    .terminal-box {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
        font-size: 0.8rem;
        padding: 1rem;
        border-radius: 8px;
        height: 340px;
        overflow-y: auto;
        white-space: pre-wrap;
        border: 1px solid #30363d;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧲 Altermagnet Screener")
st.caption(
    "Upload VASP structure files. This app runs `amcheck` across every u/d spin "
    "combination for magnetic elements and flags any structure with at least one "
    "combination where **Altermagnet? True**."
)

amcheck_ok = check_amcheck_installed()
if not amcheck_ok:
    st.error(
        "⚠️ `amcheck` isn't available on this server. Add `amcheck` to your "
        "`requirements.txt` and redeploy."
    )

if "results" not in st.session_state:
    st.session_state.results = []
if "full_log" not in st.session_state:
    st.session_state.full_log = []
if "true_files_dir" not in st.session_state:
    st.session_state.true_files_dir = None

st.divider()

uploaded_files = st.file_uploader(
    "Drop VASP structure files here",
    accept_multiple_files=True,
    help="You can select or drag in multiple files at once.",
)

run_col, clear_col = st.columns([1, 1])
run_clicked = run_col.button("▶ Run analysis", type="primary", disabled=not amcheck_ok)
clear_clicked = clear_col.button("🗑 Clear results")

if clear_clicked:
    st.session_state.results = []
    st.session_state.full_log = []
    st.session_state.true_files_dir = None
    st.rerun()

if run_clicked:
    if not uploaded_files:
        st.warning("Please upload at least one file first.")
        st.stop()

    st.session_state.results = []
    st.session_state.full_log = []

    work_dir = tempfile.mkdtemp(prefix="amcheck_uploads_")
    output_dir = tempfile.mkdtemp(prefix="amcheck_output_")
    true_files_dir = os.path.join(output_dir, "trueFiles")
    os.makedirs(true_files_dir, exist_ok=True)

    file_paths = []
    for uf in uploaded_files:
        dest = os.path.join(work_dir, uf.name)
        with open(dest, "wb") as fh:
            fh.write(uf.getbuffer())
        file_paths.append(dest)

    sequence_map = build_sequence_map(MAX_N)

    st.subheader("Progress")
    overall_progress = st.progress(0.0, text="Starting...")
    file_progress = st.progress(0.0, text="")

    st.subheader("🖥 Live terminal output")
    terminal_lines = []
    terminal_box = st.empty()
    terminal_box.markdown('<div class="terminal-box"></div>', unsafe_allow_html=True)

    results_box = st.empty()

    def stream_cb(line):
        terminal_lines.append(line)
        display_text = "\n".join(terminal_lines[-500:])
        # simple HTML-escape so amcheck output can't break the box
        display_text = (
            display_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        terminal_box.markdown(
            f'<div class="terminal-box">{display_text}</div>', unsafe_allow_html=True
        )

    all_log_lines = []

    for i, path in enumerate(file_paths):
        overall_progress.progress(
            i / len(file_paths),
            text=f"Processing file {i + 1}/{len(file_paths)}: {os.path.basename(path)}",
        )

        def progress_cb(done, total, trues, errors, _path=path):
            pct = done / total if total else 1.0
            file_progress.progress(
                pct,
                text=f"{os.path.basename(_path)}: combination {done}/{total} "
                     f"({trues} True so far, {errors} errors)",
            )

        result = process_file(path, true_files_dir, sequence_map, stream_cb, progress_cb)
        st.session_state.results.append(result)
        all_log_lines.extend(terminal_lines)
        results_box.dataframe(st.session_state.results, use_container_width=True)

    overall_progress.progress(1.0, text="Done!")
    file_progress.progress(1.0, text="")

    st.session_state.full_log = terminal_lines
    st.session_state.true_files_dir = true_files_dir
    shutil.rmtree(work_dir, ignore_errors=True)

# --------------------------------------------------------------------------
# Results display
# --------------------------------------------------------------------------

if st.session_state.results:
    st.divider()
    st.subheader("Results")

    n_files = len(st.session_state.results)
    n_flagged = sum(1 for r in st.session_state.results if r.get("Flagged"))
    m1, m2, m3 = st.columns(3)
    m1.metric("Files processed", n_files)
    m2.metric("Flagged as altermagnetic", n_flagged)
    m3.metric("Flag rate", f"{(n_flagged / n_files * 100):.0f}%" if n_files else "0%")

    st.dataframe(st.session_state.results, use_container_width=True)

    true_files_dir = st.session_state.true_files_dir
    dl_col1, dl_col2 = st.columns(2)

    if true_files_dir and os.path.isdir(true_files_dir) and os.listdir(true_files_dir):
        zip_path = os.path.join(tempfile.gettempdir(), "trueFiles.zip")
        with zipfile.ZipFile(zip_path, "w") as zf:
            for fname in os.listdir(true_files_dir):
                zf.write(os.path.join(true_files_dir, fname), arcname=fname)
        with open(zip_path, "rb") as zf:
            dl_col1.download_button(
                "⬇ Download flagged structures (trueFiles.zip)",
                data=zf.read(),
                file_name="trueFiles.zip",
                mime="application/zip",
            )
    else:
        dl_col1.info("No structures were flagged as altermagnetic.")

    if st.session_state.full_log:
        dl_col2.download_button(
            "⬇ Download full run log",
            data="\n".join(st.session_state.full_log),
            file_name="amcheck_log.txt",
            mime="text/plain",
        )
else:
    st.info("Upload files above and click **Run analysis** to begin.")
