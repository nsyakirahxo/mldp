# Phishing URL Detector — Streamlit App

## 🚀 Quick Start (local development)

1. Create and activate a Python virtual environment (recommended):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install required packages from `requirements.txt` (recommended). If you don't have a `requirements.txt`, you can create one with `pip freeze > requirements.txt` and then install.

```powershell
pip install -r requirements.txt
```

3. If you trained the model already, make sure these files are next to `app.py`:

```
app.py
phishing_model.pkl   # exported trained model
scaler.pkl           # exported scaler used at training
model_metadata.json  # feature names and metadata
```

4. Run the Streamlit app:

```powershell
streamlit run app.py
```

Open http://localhost:8501 in your browser if the page doesn't open automatically.

---

## 📋 Important Note — Feature Extraction

The app currently uses a placeholder/dummy feature extraction implementation. The model in `phishing_model.pkl` was trained on ~46 features (see `model_metadata.json`). For correct predictions on arbitrary URLs you must implement `extract_features_from_url()` in `app.py` so it returns the same features (same order and scaling) used during training. Key example features include:

- URL length and token counts
- Count of special characters ("@", "-", ".", etc.)
- Number of subdomains
- Presence of IP address in domain
- Use of URL shorteners
- Domain/SSL-related indicators

The dataset in this repo (`PhiUSIIL_Phishing_URL_Dataset.csv`) contains pre-extracted features that match the training process — use it as a reference for building a compatible extractor.

---

## 🧰 Files & Where They Come From

- `model.ipynb` — Notebook used to train the model and export `phishing_model.pkl`, `scaler.pkl`, and `model_metadata.json`.
- `model_metadata.json` — Contains feature names and other metadata; use it when building the feature extractor.
- `PhiUSIIL_Phishing_URL_Dataset.csv` — Training dataset with pre-extracted features.

If you don't have `phishing_model.pkl` and `scaler.pkl` in this folder, open `model.ipynb`, run the cells, and run the export cell (section 10.4 in the notebook) to produce them.

---

## 🔧 Troubleshooting

- **"Model files not found"**: ensure `phishing_model.pkl`, `scaler.pkl`, and `model_metadata.json` are present and named exactly.
- **Prediction mismatches**: verify your feature extractor returns the same number and order of features listed in `model_metadata.json` and apply the same scaling.
- **Streamlit not installed**: run `pip install streamlit` or reinstall with `pip install --upgrade streamlit`.

---

## 📚 Resources

- Streamlit docs: https://docs.streamlit.io/
- Scikit-learn model persistence: https://scikit-learn.org/stable/model_persistence.html
- PHIUSIIL dataset: https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

