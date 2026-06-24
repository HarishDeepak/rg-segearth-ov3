# SegEarth-OV-3 — Review & Improvement Notes

Based on: full repo at https://github.com/dummy-irl/SegEarth-OV-3 + `segearth03-bfd0b0.ipynb`

---

## Issues found

### Critical

**1. No quantitative eval in the notebook — but everything needed is already in the repo**  
`demo.py` only saves PNGs. However `eval.py` + `configs/cfg_potsdam.py` + `configs/cls_potsdam.txt` form a complete Potsdam eval pipeline — it just hasn't been wired into the Kaggle notebook. Adding a cell that stages Potsdam data at `/kaggle/working/PotsdamEval` and calls `eval.py --config configs/cfg_potsdam.py` would produce mIoU/mAcc numbers with no new code needed.

**2. Dead code: `seg_pred` computed twice in `predict()`**  
```python
# segearthov3_segmentor.py, inside the `if self.num_cls != self.num_queries:` block
seg_pred = seg_logits.argmax(0, keepdim=True)   # ← computed then immediately overwritten
...
seg_pred = torch.argmax(seg_logits, dim=0)       # ← this is the one actually used
```
When `num_cls == num_queries` (the normal case), the first assignment never runs. When `num_cls != num_queries`, the first assignment runs but is then overwritten without ever being read. No functional bug but it signals that the multi-label path was never tested.

**3. Abstract methods left empty**  
`_forward`, `inference`, `encode_decode`, `extract_feat`, `loss` in `SegEarthOV3Segmentation` are all stub `pass` methods. MMSeg's `Runner` calls some of these during eval — `eval.py` will likely fail at runtime if those code paths are hit.

### Medium

**4. MMCV_MAX version patch is fragile**  
The notebook patches `mmsegmentation/__init__.py` to change `MMCV_MAX = '2.2.0'` → `'2.3.0'`. This works because they actually installed mmcv 2.2.0 but mmseg's upper-bound check rejects it. The real fix is to pin compatible versions from the start and not patch source. If mmseg is updated this patch will break or change location.

**5. requirements.txt doesn't pin torch/mmcv/mmseg**  
The critical version constraints (torch==2.4.0, mmcv==2.2.0, mmseg==1.2.2) live only in `!pip install ...` lines in the notebook, not in requirements.txt. A developer installing from requirements.txt gets different (likely incompatible) versions. These should be in requirements.txt.

**6. `config_local` wildcard import hides dependencies**  
`from config_local import *` in every file makes it impossible to know what constants are needed without reading config_local. Should be explicit imports: `from config_local import SAM3_CHECKPOINT, INPUT_FOLDER, ...`

**7. 10-class prompts vs Frankfurt DOP20 (no GT labels)**  
Current prompts: building, road, construction area, parking lot, grassland, car, tree, water, container, clutter. The team is running on Frankfurt DOP20 which has no labeled GT — so this is purely qualitative. If they want numbers vs Potsdam GT, they'd need to switch the prompt set to 6 ISPRS classes and point `INPUT_FOLDER` at Potsdam imagery.

**8. `prob_thd=0.1` hard-coded in `demo.py`**  
Threshold is set without justification. For dense urban scenes with no empty regions, a 0.1 threshold may suppress valid predictions. Should be tunable per dataset.

**9. No seed for reproducibility**  
No `torch.manual_seed` / `np.random.seed`. Results may differ slightly between runs if any sampling is involved inside SAM3.

### Minor

**10. PAMR is in the repo but never used**  
`pamr.py` exists and is mature code (Apache 2.0, TU Darmstadt). Applying PAMR on final logits would sharpen boundaries around trees and buildings. Currently dormant.

**11. `%cd` in notebook cell is brittle**  
`%cd /kaggle/working` and `%cd SegEarth-OV-3` change the notebook's working directory persistently. If cells are re-run out of order, paths break. Better to use `subprocess` with `cwd=` or `os.chdir()` with a guard.

**12. GT loading in `demo.py` only works for Potsdam `_RGB` naming**  
The condition `if "_RGB" in img_path.name` silently skips GT loading for any other naming convention. Frankfurt DOP20 filenames (`dop20_*.jpg`) would never trigger this.

**13. No progress bar / ETA**  
The image loop prints one line per image with time, but no total ETA. `tqdm` would help for long runs.

---

## Improvement ideas

### High value

**A. Add a metrics cell** (most important for Praktikum deliverable)  
After `demo.py` runs, add a cell that loads predictions + GT labels, remaps classes to Potsdam 6-class scheme, and computes per-class IoU and mean F1. Can reuse our `metrics.py` from `rg_geoprompt` or write standalone.

**B. Add `config_local_template.py` to the repo**  
```python
# config_local_template.py — copy to config_local.py and fill in
SAM3_CHECKPOINT = "/kaggle/input/sam3-weights/sam3_checkpoint.pth"
INPUT_FOLDER = "/kaggle/input/potsdam-dataset/images"
RUN_SINGLE_IMAGE = False
TARGET_IMAGE = "example.tif"
```

**C. Pin all deps in requirements.txt**  
Move the notebook's hardcoded `pip install` versions into requirements.txt so the setup is reproducible from a single file.

**D. Enable PAMR refinement**  
Add an optional `use_pamr=True` flag. Apply PAMR on `seg_logits` before argmax. Likely free boundary quality improvement. ~5 lines of code.

### Medium value

**E. Potsdam-tuned prompt set**  
Switch to the 6 ISPRS class prompts and tune `prob_thd` per class on the val tile (6_15). Currently the 10-class prompt set has classes (water, container, construction area) that don't exist in Potsdam — these waste model capacity and could introduce confusion.

**F. Replace MMCV_MAX patch with correct version pinning**  
Install `mmcv==2.1.0` (which is within mmseg 1.2.2's declared range) instead of 2.2.0 + patch. Or use `mim install mmcv` which resolves the compatible version automatically.

**G. `config_local` → explicit imports**  
Replace `from config_local import *` with explicit `from config_local import SAM3_CHECKPOINT, INPUT_FOLDER, RUN_SINGLE_IMAGE, TARGET_IMAGE` in each file.

### Low value / nice to have

**H. tqdm progress bar in the image loop**  
**I. Abstract method stubs** — add `raise NotImplementedError` instead of silent `pass`  
**J. Single seed cell** — `torch.manual_seed(42); np.random.seed(42)`

---

## What NOT to change

- The SAM3 integration (`sam3/` directory) — that's the core innovation, touching it is risky
- The sliding window logic — it's correct and handles boundary cases properly
- The three-source score combination — the `use_transformer_decoder / use_sem_seg / use_presence_score` flags are a thoughtful design choice

---

## For the improved notebook (`segearth04`)

Priority order of changes to incorporate:
1. Add `config_local_template.py` creation cell
2. Fix `requirements.txt` pin in the install cells (no actual patch needed)
3. Use correct mmcv version to avoid the MMCV_MAX patch
4. Add Potsdam 6-class prompt set as an option
5. Add metrics cell after demo.py
6. Add `torch.manual_seed(42)`
7. Optionally enable PAMR
