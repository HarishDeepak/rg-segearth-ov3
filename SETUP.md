# Setup Guide — rg-segearth-ov3

End-to-end guide for teammates: Kaggle API setup, local editing in VS Code, pushing notebooks to Kaggle, swapping datasets, and tuning prompts/configs.

---

## 1. Prerequisites

- **Python 3.10+** with `pip`
- **Git**
- **VS Code** (recommended) or any editor
- A **Kaggle account** — needs to be added as a collaborator on the datasets/kernels (ask Harish)
- **PowerShell 7+** on Windows (already default on Win 11) — needed for `kaggle_push.ps1`

Install the Kaggle CLI:

```bash
pip install kaggle
```

---

## 2. One-time Kaggle API setup

### Option A — interactive login (recommended)

```bash
kaggle auth login
```

Follow the prompts. This writes your token to `~/.kaggle/credentials.json` (Windows: `C:\Users\<you>\.kaggle\credentials.json`).

### Option B — manual token file

1. Go to kaggle.com → Your Profile → Settings → API → **Create New Token**
2. Download `kaggle.json`, place it at `~/.kaggle/kaggle.json`
3. On Linux/Mac: `chmod 600 ~/.kaggle/kaggle.json`

### Verify it works

```bash
kaggle datasets list
```

Should return a list without auth errors.

> **Note:** `kaggle_push.ps1` reads `credentials.json` (Option A path) to set `KAGGLE_API_TOKEN` explicitly, since the Kaggle CLI v2 is inconsistent about which file it reads. If you only have `kaggle.json`, either re-run `kaggle auth login` or update `kaggle_push.ps1` line 19 to point to your file path.

---

## 3. Clone this repo

```bash
git clone https://github.com/HarishDeepak/rg-segearth-ov3
cd rg-segearth-ov3
```

The **SegEarth-OV-3 model fork** lives at a separate repo (`HarishDeepak/SegEarth-OV-3`). Kaggle notebooks clone it at runtime — you don't need it locally unless you're modifying the model architecture itself.

---

## 4. Repo layout

```
rg-segearth-ov3/
├── notebooks/                  # Source notebooks — edit these
│   ├── NB01_verify_data.ipynb
│   ├── NB02_potsdam_eval.ipynb
│   ├── NB03_hessen_infer.ipynb
│   ├── NB04_demo.ipynb
│   ├── NB05_ptsam_train.ipynb
│   └── push/                   # Kaggle push staging dirs — one per notebook
│       ├── nb01/
│       │   ├── kernel-metadata.json   ← dataset slugs, GPU settings live here
│       │   └── NB01_verify_data.ipynb ← auto-copied by kaggle_push.ps1, don't edit
│       ├── nb02/ ...
│       └── nb05/
├── configs/
│   ├── cfg_potsdam.py          # Model config for Potsdam eval
│   ├── cfg_hessen.py           # Model config for Hessen inference
│   ├── cls_potsdam.txt         # Class prompts for Potsdam (6 classes)
│   ├── cls_hessen.txt          # Class prompts for Hessen (7 classes)
│   └── cfg_*.py / cls_*.txt   # Other datasets (upstream, not used here)
├── src/segearth_utils/         # Our thin utilities (constants, eval helpers)
├── sam3/                       # SAM3 model code (from upstream fork, frozen)
├── segearthov3_segmentor.py    # Main inference wrapper
├── kaggle_push.ps1             # One-command push script
└── CLAUDE.md                   # Claude Code instructions (not for humans)
```

**Rule:** only edit files in `notebooks/` and `configs/`. Never edit anything inside `sam3/` (frozen model).

---

## 5. The push workflow

### Step 1 — edit the notebook locally

Open `notebooks/NB0X_<name>.ipynb` in VS Code (or Jupyter). Make your changes.

### Step 2 — push to Kaggle

```powershell
.\kaggle_push.ps1 nb01   # replace nb01 with nb02, nb03, nb04, or nb05
```

What this does internally:

1. Copies `notebooks/NB0X_*.ipynb` → `notebooks/push/nb0X/` (Kaggle requires the notebook inside the push folder)
2. Runs `kaggle kernels push -p notebooks/push/nb0X/` which reads `kernel-metadata.json` for GPU, datasets, etc.
3. Prints the exact next commands to check status and pull outputs

### Step 3 — check if it finished

```bash
kaggle kernels status harish77718/nb0X-<name>
```

Status cycles: `queued` → `running` → `complete` (or `error`).

### Step 4 — pull outputs

```bash
kaggle kernels output harish77718/nb0X-<name> -p results\
```

Downloads any saved files (PNGs, CSVs, logs) into your local `results/` folder.

### Kernel slugs

| Notebook | Kaggle slug                       |
| -------- | --------------------------------- |
| NB01     | `harish77718/nb01-verify-data`  |
| NB02     | `harish77718/nb02-potsdam-eval` |
| NB03     | `harish77718/nb03-hessen-infer` |
| NB04     | `harish77718/nb04-demo`         |
| NB05     | `harish77718/nb05-ptsam-train`  |

---

## 6. Changing datasets

Datasets attached to each notebook are listed in `notebooks/push/nb0X/kernel-metadata.json` under `dataset_sources`.

**Example** (`notebooks/push/nb02/kernel-metadata.json`):

```json
{
  "dataset_sources": [
    "dummyirl/sam3-weights",
    "dummyirl/6isprs"
  ]
}
```

To swap or add a dataset:

1. Find the Kaggle dataset slug (e.g. `username/dataset-name` from its Kaggle URL)
2. Edit the `dataset_sources` array in the relevant `kernel-metadata.json`
3. On Kaggle the dataset mounts at `/kaggle/input/<dataset-name>/`e.g. `dummyirl/6isprs` → `/kaggle/input/6isprs/`but `harish77718/darmstadt-dop20-presliced` → `/kaggle/input/darmstadt-dop20-presliced/`
4. Update any hardcoded paths inside the notebook to match
5. Re-push: `.\kaggle_push.ps1 nb0X`

### Current dataset slugs

| Slug                                      | What it is                            | Mount path                                                          |
| ----------------------------------------- | ------------------------------------- | ------------------------------------------------------------------- |
| `dummyirl/sam3-weights`                 | SAM3 checkpoint (`sam3.pt`)         | `/kaggle/input/sam3-weights/sam3.pt`                              |
| `dummyirl/6isprs`                       | Potsdam ISPRS tiles (RGB + labels)    | `/kaggle/input/6isprs/6ISPRS/`                                    |
| `harish77718/darmstadt-dop20-presliced` | Hessen DOP20 pre-sliced 256px patches | `/kaggle/input/darmstadt-dop20-presliced/darmstadt_dop20/images/` |

---

## 7. Changing class prompts

Prompts are plain text files — one line per class, comma-separated synonyms. SegEarth-OV-3 scores each pixel against all terms and picks the best match.

**Potsdam (6 classes):** `configs/cls_potsdam.txt`

```
impervious surface, road, pavement, paved ground
building, rooftop, structure
low vegetation, grass, shrub, lawn, meadow
tree, forest, canopy
car, vehicle
clutter, background
```

**Hessen (7 classes):** `configs/cls_hessen.txt`

```
nadir aerial view of impervious paved surface, asphalt, concrete ground, parking lot
overhead low-resolution view of building, rooftop, flat roof casting shadow, residential structure
aerial top-down view of dense tree canopy, forest patch, treetop, woodland
overhead low-resolution view of automotive vehicle, parked car, automobile seen from above
nadir view of low vegetation, grassland, lawn, meadow, green ground cover
aerial view of railway tracks, rail line, train tracks seen from above
overhead low-resolution view of paved road, urban street, road surface, carriageway
```

**To change prompts:**

1. Edit the relevant `configs/cls_*.txt`
2. `git add configs/cls_potsdam.txt && git commit -m "..." && git push`
3. The notebook clones the fork at runtime (`git clone https://github.com/HarishDeepak/SegEarth-OV-3`) — but our `configs/` is actually read from **this repo**, not the fork. The notebook copies `configs/` at startup, so pushing here is enough.
4. Re-push the notebook: `.\kaggle_push.ps1 nb0X`

> **Line order matters** — the line index must match the label index in the dataset annotation. Don't reorder lines for Potsdam.

---

## 8. Changing model config

Key config files:

| File                       | Controls                                               |
| -------------------------- | ------------------------------------------------------ |
| `configs/cfg_potsdam.py` | Potsdam eval: crop size, stride, threshold, class file |
| `configs/cfg_hessen.py`  | Hessen inference: smaller crop for 20cm GSD imagery    |

Key knobs inside those files:

```python
model = dict(
    classname_path='./configs/cls_potsdam.txt',  # which class prompt file
    prob_thd=0.1,           # min score to assign any class (lower = more coverage)
    confidence_threshold=0.1,
    slide_crop=1024,        # crop size fed to SAM3 (256 for Hessen, 1024 for Potsdam)
    slide_stride=1024,      # step between crops (128 for Hessen = 50% overlap)
)
```

**Hessen uses `slide_crop=256 / stride=128`** — this compensates for the 4× resolution gap (20cm vs 5cm GSD). SAM3 resizes any crop to 1008×1008 internally, so a 256px crop of a 20cm tile gets processed at the same pixel density as Potsdam.

After editing a config, commit + push the repo, then re-push the notebook.

---

## 9. Adding a new notebook

1. Create `notebooks/NB0X_<name>.ipynb`
2. Create `notebooks/push/nb0X/kernel-metadata.json`:

```json
{
  "id": "harish77718/nb0X-<name>",
  "title": "NB0X <Name>",
  "code_file": "NB0X_<name>.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": true,
  "enable_gpu": true,
  "enable_tpu": false,
  "machine_shape": "NvidiaTeslaT4",
  "enable_internet": true,
  "dataset_sources": [
    "dummyirl/sam3-weights"
  ],
  "competition_sources": [],
  "kernel_sources": []
}
```

3. Add an entry in `kaggle_push.ps1`:
   - Add `"nb0X"` to `[ValidateSet(...)]`
   - Add `"nb0X" = @{ file = "NB0X_<name>.ipynb"; slug = "harish77718/nb0X-<name>" }` to `$map`
4. First push creates the kernel: `.\kaggle_push.ps1 nb0X`

---

## 10. GitHub sync

This repo (`HarishDeepak/rg-segearth-ov3`) is the source of truth for:

- Notebooks (`notebooks/`)
- Class prompts (`configs/cls_*.txt`)
- Model configs (`configs/cfg_potsdam.py`, `configs/cfg_hessen.py`)
- Utilities (`src/segearth_utils/`)

Push here before running on Kaggle. The notebooks pull configs from this repo at runtime via `git clone`.

The **SegEarth-OV-3 fork** (`HarishDeepak/SegEarth-OV-3`) is separate — only touch it if modifying the model architecture or segmentor code (`segearthov3_segmentor.py`). It is cloned by Kaggle notebooks from GitHub at runtime using `enable_internet: true`.

```bash
# Normal workflow
git add configs/cls_potsdam.txt notebooks/NB02_potsdam_eval.ipynb
git commit -m "update Potsdam prompts"
git push
.\kaggle_push.ps1 nb02
```

---

## Quick reference

```powershell
# Push notebook to Kaggle
.\kaggle_push.ps1 nb02

# Check if it finished
kaggle kernels status harish77718/nb02-potsdam-eval

# Pull outputs locally
kaggle kernels output harish77718/nb02-potsdam-eval -p results\

# View Kaggle kernel logs (if it errored)
kaggle kernels output harish77718/nb02-potsdam-eval -p results\ --log
```
