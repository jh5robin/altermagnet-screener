"""
Altermagnet Screener — Streamlit app (Materials Project integration)
------------------------------------------------------------------
Search structures directly from the Materials Project database, or upload
your own VASP files, then this app drives `amcheck` (in parallel) across
every u/d spin combination for magnetic elements, flagging any structure
where at least one combination reports `Altermagnet? True`.

Performance notes vs. the original version:
  * amcheck availability check is cached (st.cache_resource) instead of
    re-spawning a subprocess on every Streamlit rerun.
  * amcheck runs for different u/d combinations are executed concurrently
    with a thread pool (subprocess.Popen releases the GIL while waiting on
    I/O, so this gives a near-linear speedup with worker count).
  * Live terminal output is throttled (batched by line-count + time) and
    kept in a bounded deque, instead of re-joining/re-escaping the full
    scrollback on every single printed line.
"""

import collections
import itertools
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
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

SUBPROCESS_TIMEOUT_SECONDS = 120

TERMINAL_MAX_LINES = 500
TERMINAL_FLUSH_EVERY_N_LINES = 15
TERMINAL_FLUSH_EVERY_SECONDS = 0.35

DEFAULT_MAX_WORKERS = 4


# --------------------------------------------------------------------------
# amcheck availability (cached — was re-run on every Streamlit rerun before)
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
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
# Materials Project search
# --------------------------------------------------------------------------

def mp_search(api_key: str, mode: str, query: str):
    """Returns a list of MPDataDoc objects (each has .material_id,
    .formula_pretty, .structure, .symmetry, .energy_above_hull, .nsites)."""
    from mp_api.client import MPRester

    fields = ["material_id", "formula_pretty", "structure", "symmetry",
              "energy_above_hull", "nsites"]

    with MPRester(api_key) as mpr:
        if mode == "Material ID":
            ids = [s.strip() for s in query.split(",") if s.strip()]
            docs = mpr.materials.summary.search(material_ids=ids, fields=fields)
        elif mode == "Chemical system (e.g. Fe-O)":
            docs = mpr.materials.summary.search(chemsys=query.strip(), fields=fields)
        else:  # Formula
            docs = mpr.materials.summary.search(formula=query.strip(), fields=fields)
    return docs


def structure_to_poscar_text(structure) -> str:
    from pymatgen.io.vasp import Poscar
    return Poscar(structure).get_string()


# --------------------------------------------------------------------------
# Throttled terminal buffer (thread-safe)
# --------------------------------------------------------------------------

class ThrottledTerminal:
    """Buffers lines from (possibly concurrent) amcheck runs and flushes to
    the UI at most every N lines or every T seconds, instead of on every
    single printed line. This is what keeps the UI responsive when a run
    produces thousands of lines of output."""

    def __init__(self, render_cb):
        self._lines = collections.deque(maxlen=TERMINAL_MAX_LINES)
        self._render_cb = render_cb
        self._lock = threading.Lock()
        self._since_flush = 0
        self._last_flush = 0.0
        self.all_lines = []  # full log kept for the downloadable file

    def write(self, line: str):
        with self._lock:
            self._lines.append(line)
            self.all_lines.append(line)
            self._since_flush += 1
            now = time.monotonic()
            should_flush = (
                self._since_flush >= TERMINAL_FLUSH_EVERY_N_LINES
                or (now - self._last_flush) >= TERMINAL_FLUSH_EVERY_SECONDS
            )
            if should_flush:
                snapshot = list(self._lines)
                self._since_flush = 0
                self._last_flush = now
            else:
                snapshot = None
        if snapshot is not None:
            self._render_cb(snapshot)

    def flush_final(self):
        with self._lock:
            snapshot = list(self._lines)
        self._render_cb(snapshot)


# --------------------------------------------------------------------------
# Core subprocess runner
# --------------------------------------------------------------------------

class AmcheckRunResult:
    def __init__(self):
        self.lines = []
        self.returncode = None
        self.timed_out = False
        self.exception = None


def _run_amcheck_process(vasp_file_path, respond_fn, stream_cb=None):
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
    index_holder = {"i": 0}
    altermagnet_result = {"value": None}

    def respond(line):
        m = ALTERMAGNET_RE.search(line)
        if m:
            altermagnet_result["value"] = m.group(1)

        if SPIN_PROMPT in line:
            i = index_holder["i"]
            if i >= len(input_array):
                return "nn"
            resp = input_array[i]
            index_holder["i"] += 1
            return resp
        return None

    run_result = _run_amcheck_process(vasp_file_path, respond, stream_cb)
    return altermagnet_result["value"], run_result


def generate_input_combinations(vasp_file_path, arr, max_workers, stream_cb=None, progress_cb=None):
    """Runs every u/d combination concurrently via a thread pool.

    subprocess.Popen blocks on I/O (waiting for the child process), which
    releases the GIL, so a thread pool gives real parallel speedup here even
    though Python threads don't parallelize pure-CPU work.
    """
    combos = list(itertools.product(*arr)) if arr else [()]
    total_combos = len(combos)

    counters = {"done": 0, "true_count": 0, "errors": 0}
    counters_lock = threading.Lock()

    def _run_one(combo):
        output, run_result = run_amcheck(vasp_file_path, list(combo), stream_cb)
        is_error = bool(
            run_result.exception or run_result.timed_out or (
                run_result.returncode not in (0, None) and output is None
            )
        )
        return output, is_error

    workers = max(1, min(max_workers, total_combos))

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_one, combo): combo for combo in combos}
        for future in as_completed(futures):
            output, is_error = future.result()
            with counters_lock:
                counters["done"] += 1
                if output == "True":
                    counters["true_count"] += 1
                if is_error:
                    counters["errors"] += 1
                snapshot = dict(counters)
            if progress_cb:
                progress_cb(snapshot["done"], total_combos, snapshot["true_count"], snapshot["errors"])

    return total_combos, counters["true_count"], counters["errors"]


def process_file(vasp_file_path, true_files_dir, sequence_map, max_workers, stream_cb=None, progress_cb=None):
    if stream_cb:
        stream_cb(f"===== Discovery pass for {os.path.basename(vasp_file_path)} =====")

    element_array, atom_count_array, nn_run_result = run_nn_amcheck(vasp_file_path, stream_cb)

    if not element_array:
        reason = (
            "amcheck produced no recognizable element/atom output. "
            "Check the terminal panel above."
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
        n_combos = len(list(itertools.product(*element_input_array))) if element_input_array else 1
        stream_cb(f"===== Running {n_combos} combination(s) for {os.path.basename(vasp_file_path)} "
                   f"(up to {max_workers} in parallel) =====")

    total, true_count, errors = generate_input_combinations(
        vasp_file_path, element_input_array, max_workers, stream_cb, progress_cb
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
    .block-container {padding-top: 2rem; max-width: 1200px;}
    div[data-testid="stMetricValue"] {font-size: 1.7rem; font-weight: 700;}
    div[data-testid="stMetric"] {
        background: rgba(148, 163, 184, 0.08);
        border: 1px solid rgba(148, 163, 184, 0.18);
        border-radius: 12px;
        padding: 0.75rem 1rem;
    }

    .am-hero {
        background: linear-gradient(120deg, #4338ca 0%, #7c3aed 45%, #db2777 100%);
        border-radius: 16px;
        padding: 1.6rem 1.8rem;
        margin-bottom: 1.4rem;
        color: white;
    }
    .am-hero h1 {
        margin: 0 0 0.35rem 0;
        font-size: 1.9rem;
        color: white;
    }
    .am-hero p {
        margin: 0;
        opacity: 0.92;
        font-size: 0.98rem;
    }

    .terminal-box {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'SF Mono', 'Consolas', 'Monaco', monospace;
        font-size: 0.78rem;
        padding: 1rem;
        border-radius: 10px;
        height: 340px;
        overflow-y: auto;
        white-space: pre-wrap;
        border: 1px solid #30363d;
        line-height: 1.45;
    }

    .badge {
        display: inline-block;
        padding: 0.15rem 0.6rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .badge-flagged {background: rgba(219, 39, 119, 0.15); color: #db2777; border: 1px solid rgba(219,39,119,0.35);}
    .badge-clean {background: rgba(16, 185, 129, 0.12); color: #059669; border: 1px solid rgba(16,185,129,0.3);}
    .badge-error {background: rgba(239, 68, 68, 0.12); color: #dc2626; border: 1px solid rgba(239,68,68,0.3);}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="am-hero">
        <h1>🧲 Altermagnet Screener</h1>
        <p>Search a structure from the Materials Project, or upload your own VASP files.
        This app runs <code>amcheck</code> in parallel across every u/d spin combination
        for magnetic elements and flags any structure with at least one combination
        where <b>Altermagnet? True</b>.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

amcheck_ok = check_amcheck_installed()
if not amcheck_ok:
    st.error(
        "⚠️ `amcheck` isn't available on this server. Add `amcheck` to your "
        "`requirements.txt` and redeploy."
    )

# --- session state init -----------------------------------------------------
for key, default in [
    ("results", []),
    ("full_log", []),
    ("true_files_dir", None),
    ("mp_docs", {}),
    ("mp_queue", []),  # list of {"material_id", "formula", "poscar_text"}
]:
    if key not in st.session_state:
        st.session_state[key] = default

# --------------------------------------------------------------------------
# Sidebar: performance settings
# --------------------------------------------------------------------------

with st.sidebar:
    st.subheader("⚙️ Run settings")
    max_workers = st.slider(
        "Parallel amcheck workers",
        min_value=1, max_value=16, value=DEFAULT_MAX_WORKERS,
        help="How many u/d spin combinations to run at once. Higher is faster "
             "but uses more CPU/RAM on the server — start around 4–8.",
    )
    st.caption(
        "Each amcheck run is its own subprocess, so running several combinations "
        "concurrently is usually the single biggest speed-up available."
    )

st.divider()

# --------------------------------------------------------------------------
# Input: tabs for Materials Project search vs. file upload
# --------------------------------------------------------------------------

tab_mp, tab_upload = st.tabs(["🔍 Search Materials Project", "📁 Upload files"])

with tab_mp:
    with st.expander("Where do I get an API key?", expanded=False):
        st.markdown(
            "1. Go to **[next-gen.materialsproject.org](https://next-gen.materialsproject.org)** "
            "and log in (free account).\n"
            "2. Open your **Dashboard** — your API key is shown there.\n"
            "3. Paste it below. It's only kept in this browser session, never stored on disk."
        )

    api_key = st.text_input("Materials Project API key", type="password", key="mp_api_key")

    # Wrapped in a form so typing the query doesn't trigger a rerun on every
    # keystroke — only "Search" submits.
    with st.form("mp_search_form"):
        search_col1, search_col2 = st.columns([1, 2])
        with search_col1:
            search_mode = st.radio(
                "Search by", ["Formula", "Chemical system (e.g. Fe-O)", "Material ID"],
            )
        with search_col2:
            placeholder = {
                "Formula": "e.g. Fe2O3",
                "Chemical system (e.g. Fe-O)": "e.g. Fe-O",
                "Material ID": "e.g. mp-19770 (comma-separate for several)",
            }[search_mode]
            query = st.text_input("Query", placeholder=placeholder)
        search_submitted = st.form_submit_button("Search", disabled=not amcheck_ok)

    if search_submitted:
        if not api_key or not query:
            st.warning("Enter both an API key and a query.")
        else:
            try:
                with st.spinner("Searching Materials Project..."):
                    docs = mp_search(api_key, search_mode, query)
                if not docs:
                    st.warning("No matching structures found.")
                else:
                    st.session_state.mp_docs = {d.material_id: d for d in docs}
                    st.success(f"Found {len(docs)} structure(s).")
            except Exception as e:
                st.error(f"Search failed: {e}")

    if st.session_state.mp_docs:
        table_rows = []
        for mid, d in st.session_state.mp_docs.items():
            table_rows.append({
                "Material ID": str(mid),
                "Formula": d.formula_pretty,
                "Spacegroup": getattr(d.symmetry, "symbol", "—") if d.symmetry else "—",
                "Energy above hull (eV/atom)": round(d.energy_above_hull, 4) if d.energy_above_hull is not None else None,
                "Sites": d.nsites,
            })
        st.dataframe(table_rows, use_container_width=True, hide_index=True)

        options = [f"{mid} — {d.formula_pretty}" for mid, d in st.session_state.mp_docs.items()]
        selected = st.multiselect("Select structure(s) to add to the analysis queue", options)

        if st.button("➕ Add selected to queue", disabled=not selected):
            for opt in selected:
                mid = opt.split(" — ")[0]
                doc = st.session_state.mp_docs[mid]
                poscar_text = structure_to_poscar_text(doc.structure)
                already_queued = any(q["material_id"] == mid for q in st.session_state.mp_queue)
                if not already_queued:
                    st.session_state.mp_queue.append({
                        "material_id": mid,
                        "formula": doc.formula_pretty,
                        "poscar_text": poscar_text,
                    })
            st.rerun()

    if st.session_state.mp_queue:
        st.write("**Queued from Materials Project:**")
        for i, item in enumerate(st.session_state.mp_queue):
            c1, c2 = st.columns([5, 1])
            c1.write(f"`{item['material_id']}` — {item['formula']}")
            if c2.button("Remove", key=f"remove_mp_{i}"):
                st.session_state.mp_queue.pop(i)
                st.rerun()

with tab_upload:
    uploaded_files = st.file_uploader(
        "Drop VASP structure files here",
        accept_multiple_files=True,
        help="You can select or drag in multiple files at once.",
    )

st.divider()

run_col, clear_col = st.columns([1, 1])
run_clicked = run_col.button("▶ Run analysis", type="primary", disabled=not amcheck_ok)
clear_clicked = clear_col.button("🗑 Clear results")

if clear_clicked:
    st.session_state.results = []
    st.session_state.full_log = []
    st.session_state.true_files_dir = None
    st.rerun()

if run_clicked:
    uploaded_files = uploaded_files or []
    if not uploaded_files and not st.session_state.mp_queue:
        st.warning("Please upload at least one file or add a structure from Materials Project first.")
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

    for item in st.session_state.mp_queue:
        safe_name = f"{item['material_id']}_{item['formula']}.vasp".replace("/", "-")
        dest = os.path.join(work_dir, safe_name)
        with open(dest, "w") as fh:
            fh.write(item["poscar_text"])
        file_paths.append(dest)

    sequence_map = build_sequence_map(MAX_N)

    st.subheader("Progress")
    overall_progress = st.progress(0.0, text="Starting...")
    file_progress = st.progress(0.0, text="")

    st.subheader("🖥 Live terminal output")
    terminal_box = st.empty()
    terminal_box.markdown('<div class="terminal-box"></div>', unsafe_allow_html=True)

    results_box = st.empty()

    def render_terminal(snapshot_lines):
        display_text = "\n".join(snapshot_lines)
        display_text = (
            display_text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        terminal_box.markdown(
            f'<div class="terminal-box">{display_text}</div>', unsafe_allow_html=True
        )

    terminal = ThrottledTerminal(render_terminal)
    stream_cb = terminal.write

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

        result = process_file(path, true_files_dir, sequence_map, max_workers, stream_cb, progress_cb)
        st.session_state.results.append(result)
        results_box.dataframe(st.session_state.results, use_container_width=True, hide_index=True)

    terminal.flush_final()
    overall_progress.progress(1.0, text="Done!")
    file_progress.progress(1.0, text="")

    st.session_state.full_log = terminal.all_lines
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
    n_errors = sum(1 for r in st.session_state.results if "error" in str(r.get("Status", "")).lower())

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Files processed", n_files)
    m2.metric("🧲 Flagged as altermagnetic", n_flagged)
    m3.metric("Flag rate", f"{(n_flagged / n_files * 100):.0f}%" if n_files else "0%")
    m4.metric("⚠️ Errors", n_errors)

    def badge_for(row):
        if row.get("Flagged"):
            return '<span class="badge badge-flagged">🧲 Flagged</span>'
        if "error" in str(row.get("Status", "")).lower() or row.get("Status") == "Error":
            return '<span class="badge badge-error">Error</span>'
        return '<span class="badge badge-clean">Clean</span>'

    df = pd.DataFrame(st.session_state.results)
    if not df.empty:
        df.insert(1, "", [badge_for(r) for r in st.session_state.results])
        st.markdown(
            df.to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )

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
    st.info("Search Materials Project or upload files above, then click **Run analysis**.")
