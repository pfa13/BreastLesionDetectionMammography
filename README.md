# Breast Lesion Detection (DMID)

## Dataset

Download the dataset manually (not included due to size):
https://figshare.com/articles/dataset/_b_Digital_mammography_Dataset_for_Breast_Cancer_Diagnosis_Research_DMID_b_DMID_rar/24522883

---

## Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

---

## Preprocessing

Run from the project root:

```bash
python -m src.utils.preprocess --input "PATH/TO/DATASET.zip"
```

This will:

* Extract the dataset into `data/raw/`
* Create train/val/test splits
* Generate COCO annotations in `data/annotations/`

---

## Output

```bash
data/
├── raw/
├── splits/
└── annotations/
```

---

## Notes

* Dataset is not tracked in git (`.gitignore`)
* Annotations are converted to COCO format from center + radius
