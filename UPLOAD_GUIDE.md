# PhysIQ — Kaggle Submission Upload Guide

## Bundle Measurement

| Metric | Value |
|--------|-------|
| Bundle size | **2.6 MB** |
| File count | 34 files (33 + BUNDLE_LAYOUT.md) |
| Submission path | **PATH A — direct zip upload** |

---

## PATH A — Under 100 MB: Zip and Upload Directly

### Step 1 — Zip the bundle (PowerShell)

```powershell
cd "C:\Users\terry\downloads\empirical_research\kaggle_agi_bench\final_submission"
Compress-Archive -Path submission_bundle\* -DestinationPath physiq_submission.zip
```

This creates `physiq_submission.zip` (~2.6 MB) in `final_submission/`.

### Step 2 — Upload to Kaggle writeup page

1. Open your Kaggle competition writeup page
2. Scroll to the **FILES** section
3. Click **"Upload Files"**
4. Select `physiq_submission.zip`
5. Done ✓

---

## PATH B — Over 100 MB (not needed here, for reference)

### Option B1 — GitHub (recommended)

```bash
cd submission_bundle
git init
git add .
git commit -m "PhysIQ benchmark — initial submission"
git remote add origin https://github.com/USERNAME/physiq-benchmark.git
git push -u origin main
```

Then on the Kaggle writeup page:
- Click **"Add a link"** under **PROJECT LINKS**
- Paste the GitHub URL
- Title: `PhysIQ — Source Code & Results`

### Option B2 — Kaggle Dataset

1. Go to kaggle.com/datasets → **New Dataset**
2. Drag the `submission_bundle/` folder into the uploader
3. Publish the dataset
4. On the writeup page → **"Add a link"** → **Kaggle Datasets** tab → select it

> **Recommendation:** B1 (GitHub) — judges can browse code directly without downloading.

---

## Next 3 Commands

```powershell
# 1. Navigate to the submission folder
cd "C:\Users\terry\downloads\empirical_research\kaggle_agi_bench\final_submission"

# 2. Create the zip
Compress-Archive -Path submission_bundle\* -DestinationPath physiq_submission.zip

# 3. Verify the zip size before uploading
(Get-Item physiq_submission.zip).Length / 1MB
```
