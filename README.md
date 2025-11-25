# Arabic Medical Question Classification (Automated Triage)

## 📋 Project Overview
This project implements an end-to-end Natural Language Processing (NLP) pipeline designed to classify Arabic medical user queries into specific medical specialties (e.g., Internal Medicine, Dentistry, Ophthalmology).

The system addresses the complexities of **Arabic medical text**, which often includes code-switching (mixing English medical terms with Arabic), dialectal variations, and noisy input. The solution explores a wide spectrum of approaches, ranging from classical statistical methods (TF-IDF/Naive Bayes) to Deep Learning (Bi-GRU/LSTM) and State-of-the-Art Transformers (AraBERT, AraGPT2).

## 🔒 Data Privacy Notice
**Important:** The dataset used for training and evaluating these models is sourced from **Altibbi**.
* This data is **private, proprietary, and confidential**.
* The raw CSV files referenced in the code (`altibbi_specialty_data.csv`) are **not** included in this repository.
* This repository serves as a showcase of the methodology, architecture, and NLP pipeline construction.

## 🛠️ Methodology & Pipeline

### 1. Exploratory Data Analysis (EDA) & Cleaning
* **Sanitization:** Removal of empty strings, duplicates, and URLs.
* **English/Arabic Code-Switching Analysis:** Calculated the ratio of English words per query to identify the prevalence of medical terminology in Latin script.
* **Data Cleaning:**
    * Regex-based filtering of alphanumeric noise.
    * Normalization of Arabic characters (unifying Alef forms, handling Teh Marbuta).

### 2. Hybrid Translation & Localization (En $\to$ Ar)
To unify the vector space, English medical terms were translated to Arabic using a multi-stage approach:
1.  **Identification:** Used **FastText Language Identification (LID)** to detect English phrases within Arabic text.
2.  **Ranking:** Extracted top keywords using TF-IDF to prioritize translation of information-dense terms.
3.  **Machine Translation:** Deployed **MarianMT (`Helsinki-NLP/opus-mt-en-ar`)** to translate detected phrases.
4.  **Dictionary Mapping:** Implemented a custom glossary for domain-specific entities (e.g., brand names like *Voltfast*, abbreviations like *MRI*, *WBC*).

### 3. Preprocessing & Stemming Comparison
The project benchmarked three stemming algorithms to determine the optimal normalization strategy:
* **ISRI Stemmer**
* **Porter Stemmer**
* **Snowball Stemmer**
* *Validation:* Each stemmer was tested using a Multinomial Naive Bayes baseline.

### 4. Semi-Supervised Label Correction
To improve data quality, a **Semi-Supervised Learning** approach was applied:
* A Logistic Regression model was trained on a high-confidence subset of data.
* The model predicted labels for the remaining dataset.
* Instances with **high-confidence mismatches** (>90%) between the original tag and the prediction were automatically corrected to reduce label noise.

### 5. Feature Engineering & Embeddings
Multiple text representation techniques were evaluated:
* **Bag of Words (BoW)** & **TF-IDF**
* **Word2Vec** (CBOW/Skip-gram)
* **FastText** (Handling out-of-vocabulary words via sub-word information)

## 🧠 Model Architectures

The project progressed through three tiers of model complexity:

### Tier 1: Classical Machine Learning
* **Models:** Logistic Regression, Linear SVC, Multinomial & Gaussian Naive Bayes.
* **Goal:** Establish strong baselines and evaluate embedding effectiveness.

### Tier 2: Deep Learning (Sequential Models)
Implemented using **TensorFlow/Keras**:
* **Bi-Directional GRU (Gated Recurrent Unit):** Tested with both Word2Vec and FastText embeddings.
* **Bi-Directional LSTM (Long Short-Term Memory):** Optimized for capturing long-range dependencies in patient narratives.
* **Architecture:** Embedding Layer (Frozen) $\to$ Bi-RNN $\to$ Dropout $\to$ Dense Layers.

### Tier 3: Transformers (Hugging Face / PyTorch)
Fine-tuned State-of-the-Art Arabic Language Models:
* **AraBERT (v02):** BERT-based architecture optimized for Arabic segments.
* **AraGPT2:** Generative Pre-trained Transformer adapted for classification tasks.
* **Training Details:**
    * Dynamic padding via `DataCollator`.
    * Mixed Precision Training (`torch.cuda.amp`) for efficiency.
    * AdamW Optimizer.

## 📊 Evaluation Metrics
All models were evaluated on a stratified test set using the following metrics:
* **Accuracy**
* **Weighted F1-Score** (Crucial for class imbalance)
* **Precision & Recall**

## 💻 Tech Stack

* **Languages:** Python 3.x
* **Data Manipulation:** Pandas, NumPy, Regex
* **NLP Tools:** NLTK, Gensim, FastText
* **Machine Learning:** Scikit-Learn
* **Deep Learning:** PyTorch (Transformers), TensorFlow/Keras (RNNs)
* **Translation:** Hugging Face Pipelines (MarianMT)
