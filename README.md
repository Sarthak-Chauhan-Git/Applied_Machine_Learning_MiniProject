# newsIQ: Multiclass News Article Classification Using Classical NLP 📰🧠

An Applied Machine Learning Mini Project (Experiment 10)
School of Computer Science, UPES Dehradun

## 📌 Overview

newsIQ is an end-to-end Natural Language Processing (NLP) system that automatically assigns a news article to one of eight categories: **Business, Education, Entertainment, Health, Politics, Science, Sports,** and **Technology**. The project compares seven classical machine-learning models on Bag-of-Words (BoW) and TF-IDF representations, tunes the strongest linear model with cross-validated grid search, evaluates it with multiple metrics, and serves the final pipeline through a Streamlit web app that shows the predicted category with a relative confidence score and the top-3 categories.

## Architecture

<!-- Replace with your architecture diagram, e.g. docs/architecture.png -->

`<img width="900" alt="newsIQ architecture" src="docs/architecture.png" />`

## ✨ Features

- Eight-class news classification trained on 30,000 articles
- Unified text-cleaning pipeline (lowercasing, URL/HTML/number/punctuation removal, stop-word and short-token filtering), reused identically at training and inference time
- Comparison of seven classical models on TF-IDF unigram + bigram features
- Leakage-safe design: vectorizers live inside scikit-learn pipelines, so vocabulary is learned only on training folds
- Hyperparameter tuning of Linear SVM via `GridSearchCV`
- Rich evaluation: accuracy, macro precision/recall/F1, weighted F1, 5-fold CV, confusion matrices, most informative words per class, and error analysis
- Trained pipeline saved as `news_classifier.joblib` for reuse
- **newsIQ** Streamlit frontend with headline and article inputs, color-coded category chips, relative confidence, and top-3 ranking

## 🧠 Problem Statement

News websites publish large volumes of articles every day, and assigning each one to a category by hand is slow and inconsistent. A reliable automatic classifier reduces editorial effort and makes archives searchable. This project studies how far classical, interpretable text-classification methods can go on this task, how BoW and TF-IDF features compare, and, importantly, how evaluation can be misleading when a dataset contains repeated article templates. We therefore report not only the scores but also a candid analysis of their limits.

## 📊 Performance

Stratified 80:20 train/test split plus 5-fold stratified cross-validation on the training set. All seven models use TF-IDF unigram + bigram features.

| Model               | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | 5-fold CV F1 |
| ------------------- | -------: | --------------: | -----------: | -------: | ----------: | -----------: |
| Logistic Regression |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| Linear SVM          |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| Multinomial NB      |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| Complement NB       |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| Decision Tree       |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| Random Forest       |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| KNN (k=5, cosine)   |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |

**Final tuned model:** Linear SVM (`C=0.1`, unigram features, `min_df=1`, `max_features=20000`), test accuracy **1.000** and macro F1 **1.000**.

### ⚠️ Interpreting these results

The perfect scores should **not** be read as real-world generalization performance. The source CSV contains many repeated article templates, and these rows are retained so that all eight categories stay represented after preprocessing. A random split therefore places near-identical text in both the training and test partitions, which makes the task close to memorization and produces optimistic scores. A production-grade evaluation should deduplicate the data or use a group-based split by source or article identity. This is the main limitation of the current study and the first item in our future work.

## 🗂️ Dataset

**Source:** [News Classification Dataset 2026](https://www.kaggle.com/datasets/mmumairkhattak/news-classification-dataset-2026-nlp-and-tc) (Kaggle)

| Property       |                                                    Value |
| -------------- | -------------------------------------------------------: |
| Articles       |                                                   30,000 |
| Classes        |                                                        8 |
| Text fields    |                     `Title`, `Article_Text`, `Full_Text` |
| Target field   |                                               `Category` |
| Other metadata | Source, author, region, date, keywords, length, language |

| Category      | Articles |
| ------------- | -------: |
| Business      |    3,814 |
| Education     |    3,756 |
| Entertainment |    3,761 |
| Health        |    3,803 |
| Politics      |    3,756 |
| Science       |    3,686 |
| Sports        |    3,690 |
| Technology    |    3,734 |

The classes are approximately balanced.

## ⚙️ System Architecture

The pipeline consists of:

1. **Data layer:** CSV ingestion with automatic target and text-column detection (low-cardinality target preferred by name; longest text column used as article body, with the title prepended when present)
2. **Preprocessing:** lowercase, remove URLs, HTML tags, digits and non-alphabetic symbols, whitespace tokenization, English stop-word removal, drop tokens shorter than three characters, drop empty articles
3. **Feature extraction:** CountVectorizer (BoW, unigrams), TF-IDF (unigrams), and TF-IDF (unigrams + bigrams with sublinear term frequency and a vocabulary limit)
4. **Model zoo:** Logistic Regression, Linear SVM, Multinomial NB, Complement NB, Decision Tree, Random Forest, KNN
5. **Tuning:** `GridSearchCV` over n-gram range, `min_df`, vocabulary size, and Linear SVM `C`
6. **Evaluation:** held-out test set, 5-fold stratified CV, confusion matrices, per-class metrics, informative-word analysis, and misclassification review
7. **Serving:** `news_classifier.joblib` loaded by a Streamlit app (`app.py`). Confidence is shown as a _relative_ score: a softmax over the SVM decision-function values, which is a ranking aid and not a calibrated probability

## 🛠️ Tech Stack

- **Language:** Python
- **ML / NLP:** scikit-learn (TF-IDF, CountVectorizer, Linear SVM, Logistic Regression, Naive Bayes, Tree and Forest models, KNN, GridSearchCV)
- **Data:** pandas · NumPy
- **Model persistence:** joblib
- **Experimentation:** Jupyter Notebook
- **Frontend:** Streamlit with custom CSS ("Newsprint & Ink" theme)

## 🧪 Evaluation Metrics

- **Accuracy:** proportion of correctly classified articles
- **Macro precision / recall / F1:** per-category averages that treat every category equally
- **Weighted F1:** F1 weighted by category frequency
- **5-fold CV F1:** stability estimate across five stratified training folds
- **Confusion matrix:** raw and normalized, showing category-level error patterns

## 🚀 Getting Started

```bash
git clone https://github.com/Sarthak-Chauhan-Git/Applied_Machine_Learning_MiniProject.git
cd Applied_Machine_Learning_MiniProject
pip install -r requirements.txt
```

**Run the experiments**

```bash
jupyter notebook Exp10_News_Classification_MiniProject.ipynb
```

Keep `news_classification_dataset.csv` beside the notebook for a reproducible real-dataset run, then select **Run All**.

**Run the web app**

```bash
streamlit run app.py
```

**Use the saved model in Python**

```python
predict_news("The central bank raised interest rates as inflation increased")
```

## 📁 Repository Contents

| File                                          | Purpose                                                            |
| --------------------------------------------- | ------------------------------------------------------------------ |
| `Exp10_News_Classification_MiniProject.ipynb` | Full implementation, experiments, plots, evaluation, and inference |
| `app.py`                                      | newsIQ Streamlit frontend                                          |
| `.streamlit/config.toml`                      | Streamlit theme configuration                                      |
| `news_classification_dataset.csv`             | Kaggle news dataset                                                |
| `data_dictionary.csv`                         | Dataset column descriptions                                        |
| `news_classifier.joblib`                      | Saved final model pipeline                                         |
| `requirements.txt`                            | Python dependencies                                                |

## 📈 Future Scope

- Deduplicate and group-split the data to obtain an honest estimate of generalization
- Evaluate on an independent, real-world news corpus (out-of-distribution test)
- Word2Vec, GloVe, fastText, LSTM/CNN, and transformer models (BERT, DistilBERT)
- Lemmatization, named-entity features, and domain-specific phrase features
- Probability calibration and confidence thresholds (e.g. abstain on low-confidence predictions)
- Multilingual and Hindi/Hinglish news classification
- Explainability with SHAP or LIME
- Deployment via FastAPI/Flask, plus automated tests, experiment tracking, model monitoring, and data-drift detection

## Contact

For any inquiries or feedback, please contact:

**Name:** Sarthak Chauhan
**System ID:** 590030655
**Education:** B.Tech, UPES, Dehradun
**Email:** sarthakchauhan5002@gmail.com
**GitHub:** https://github.com/Sarthak-Chauhan-Git/Applied_Machine_Learning_MiniProject

**Name:** Ravinder Singh
**System ID:** 590030651
**Education:** B.Tech, UPES, Dehradun
**Email:** ravindersingh522876@gmail.com
**GitHub:** https://github.com/Sarthak-Chauhan-Git/Applied_Machine_Learning_MiniProject

**Name:** Anshul Ghildiyal
**System ID:** 590031643
**Education:** B.Tech, UPES, Dehradun
**Email:** anshul.ghildiyal07@gmail.com
**GitHub:** https://github.com/Sarthak-Chauhan-Git/Applied_Machine_Learning_MiniProject

## Mentor

**Dr. Sahinur Rahman Laskar**
Assistant Professor (Senior Scale)
School of Computer Science, UPES, Dehradun, India
**Email:** sahinurlaskar.nits@gmail.com / sahinur.laskar@ddn.upes.ac.in

## Citation

```bibtex
@misc{newsiq2026,
  title={newsIQ: Multiclass News Article Classification Using Classical NLP and TF-IDF Features},
  author={Sarthak Chauhan and Ravinder Singh and Anshul Ghildiyal and Sahinur Rahman Laskar},
  year={2026},
  eprint={<arXiv-ID>},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/<arXiv-ID>}
}
```
