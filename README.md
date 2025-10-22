# 🧩 Embeddings Comparison: Word2Vec, FastText, GloVe, spaCy, SBERT, and Raw BERT

This project provides a **side-by-side interactive comparison** of the most popular open-source embedding methods — from traditional word vectors to state-of-the-art transformer-based sentence embeddings.

---

## 🚀 Features

### 🧱 Core Embedding Models
| Type | Model | Library |
|------|--------|----------|
| Word-Level | Word2Vec | Gensim |
| Word-Level | FastText | Gensim |
| Word-Level (Pretrained) | GloVe (100d) | Gensim Data API |
| Sentence-Level | spaCy Medium (static) | spaCy |
| Sentence-Level | SBERT (all-MiniLM-L6-v2) | SentenceTransformers |
| Sentence-Level | BERT (bert-base-uncased) | Hugging Face Transformers |

---

## 🧰 Functionality

### 🔤 Tokenization
- Sentence splitting and NLTK tokenization
- Stopword removal using NLTK
- Option to customize or extend token filters

### 🧮 Embedding & Visualization
- Train your own **Word2Vec** and **FastText** on sample corpus
- Load **pretrained GloVe** vectors
- Generate **sentence embeddings** with spaCy / SBERT / BERT
- Display **sample vectors** (first few dimensions)
- Compute **cosine similarity tables** across all models
- Generate **cosine similarity heatmaps** for sentence embeddings
- Perform **semantic search** and **KMeans clustering** with silhouette scores

### 🧭 Visual Exploration
- **3D PCA word plots** with cosine-similarity edges
- **3D and 2D analogy visualizations** (e.g., _king - man + woman ≈ queen_)
- **2D UMAP visualizations** for both word and sentence embeddings
- **3D sentence embeddings comparison (SBERT vs BERT)**

---

## 🧠 Key Insights
- SBERT produces smoother sentence clusters and better semantic separation than vanilla BERT.
- Word2Vec and FastText need large corpora to form meaningful relations.
- GloVe pretrained vectors offer the most robust word-level similarities out of the box.
- UMAP 2D plots give visually interpretable clusters of semantically related entities.

---

## 🛠️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m nltk.downloader punkt stopwords
python -m spacy download en_core_web_md
```

### 2. Run the notebook
```bash
jupyter notebook Embeddings_Comparison.ipynb
```

---

## 📊 Notebook Sections

| Section | Description |
|----------|-------------|
| 1–3 | Setup, Data, Tokenization |
| 4–5 | Train Word2Vec / FastText + Load GloVe / spaCy / SBERT / BERT |
| 6 | Utility functions (cosine, embedding wrappers) |
| 7 | Display sample embeddings |
| 7b | Sentence-level embeddings + cosine heatmaps |
| 8 | Word-level cosine similarities |
| 9 | Analogies (`king - man + woman ≈ queen`) |
| 10 | Sentence semantic search + clustering |
| 11 | 3D PCA word plots |
| 11b | 3D PCA sentence plots (SBERT vs BERT) |
| 12 | 2D UMAP visualizations |
| 13 | 3D and 2D analogy arrows |

---

## 🧩 Example Outputs

**Analogy Results:**
```
GloVe ->
[('queen', 0.77), ('monarch', 0.68), ('throne', 0.67)]
```

**Sentence Search (SBERT):**
```
Query: "capital of a country"
Top matches:
1. Rome is the capital of Italy.
2. Paris is the capital of France.
```

**3D Plot Preview:**
- Word clusters (`king`, `queen`, `palace`)
- Fruit clusters (`apple`, `banana`)
- Geographic pairs (`paris` ↔ `france`, `rome` ↔ `italy`)

---

## 🧭 Tips
- For serious training, use larger corpora (≥100K tokens).
- Tune hyperparameters:
  - `min_count ≥ 5`
  - `window ≈ 5–10`
  - `negative = 10`
  - `epochs = 10–30`
- SBERT is ideal for semantic search and clustering tasks.

---

## 📜 License
MIT License © 2025 — freely usable for research, education, and experimentation.
