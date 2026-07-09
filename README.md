# Altermagnet Screener

A Streamlit-based graphical interface for screening altermagnetic materials
using amcheck. Search structures directly from the Materials Project or upload
your own VASP files for automated screening.

![Application Screenshot](images/Capture.PNG)

## 🚀 Quick Start

```bash
git clone https://github.com/USERNAME/altermagnet-screener.git

cd altermagnet-screener

python -m venv venv

# Windows
venv\Scripts\activate

# Linux/macOS
source venv/bin/activate

pip install -r requirements.txt

streamlit run streamlit_app.py
```

## 1.Features

- Streamlit graphical interface
- Upload one or multiple POSCAR/VASP files
- Search Materials Project directly
- Automatic spin configuration generation
- Parallel screening using amcheck
- Live terminal output
- Download flagged structures
- Download complete execution log

## 2.Requirements

- Python 3.10 or newer
- pip
- Git
- amcheck

# 3.Installation
  ## Clone repository
  ```bash
git clone https://github.com/USERNAME/altermagnet-screener.git

cd altermagnet-screener
```
 ## Create Virtual Environment
 ### Windows
 ```bash
 python -m venv venv
venv\Scripts\activate
```

### Linux/MacOS
```bash
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies
```bash
pip install --upgrade pip

pip install -r requirements.txt
```
## Running the Application
```bash
streamlit run streamlit_app.py
```
### Application opens automatically in
```bash
http://localhost:8501
````
# 4.Using the Application
## Option 1
```bash
Upload Files

↓

Run Analysis

↓

Download Results
```

## Option 2(Material Project Search)
```bash
Enter API Key

↓

Search Formula

↓

Add to Queue

↓

Run Analysis
```
