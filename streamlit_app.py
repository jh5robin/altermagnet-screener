import subprocess

import streamlit as st

st.set_page_config(page_title="Altermagnet Screener — Setup Check", page_icon="🧲")

st.title("🧲 Setup Check")
st.write(
    "This tiny app just confirms that `amcheck` installed correctly on the server. "
    "Once this shows a green checkmark, we'll swap in the full app."
)


def check_amcheck_installed() -> tuple[bool, str]:
    try:
        result = subprocess.run(
            ["amcheck", "--help"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0, (result.stdout + result.stderr)
    except FileNotFoundError as e:
        return False, f"amcheck command not found: {e}"
    except Exception as e:
        return False, str(e)


ok, output = check_amcheck_installed()

if ok:
    st.success("✅ amcheck is installed and working!")
else:
    st.error("❌ amcheck is NOT available on this server.")

with st.expander("Show raw output / error"):
    st.code(output or "(no output)")

st.divider()
st.subheader("Python package check")

import importlib

for pkg in ["ase", "spglib", "diophantine"]:
    try:
        importlib.import_module(pkg)
        st.write(f"✅ `{pkg}` imports fine")
    except Exception as e:
        st.write(f"❌ `{pkg}` failed to import: {e}")
