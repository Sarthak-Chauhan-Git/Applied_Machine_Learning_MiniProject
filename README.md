# Experiment 10 Mini Project

## News Article Multiclass Classification Using NLP

### Title Page

| Academic detail   | Information                                                                       |
| ----------------- | --------------------------------------------------------------------------------- |
| **Project title** | News Article Multiclass Classification Using NLP                                  |
| **Experiment**    | 10 – Mini Project                                                                 |
| **Subject**       | `Applied Machine Learning`                                                        |
| **Faculty**       | `Dr. Sahinur Rahman Laskar`                                                       |
| **Institution**   | `UPES`                                                                            |
| **Repository**    | `https://github.com/Sarthak-Chauhan-Git/Applied_Machine_Learning_MiniProject.git` |

### Team Members

| No. | Student name       | System ID / Roll No. |
| --: | ------------------ | -------------------- |
|   1 | `Sarthak Chauhan`  | `590030655`          |
|   2 | `Ravinder Singh`   | `590030651`          |
|   3 | `Aarti Somani`     | `590040559`          |
|   4 | `Anshul Ghildiyal` | `590031643`          |

---

## 1. Introduction

News websites publish large volumes of articles every day. Manually assigning each article to a category is slow and inconsistent. This project develops a multiclass Natural Language Processing (NLP) classifier that automatically assigns an article to one of eight categories: **Business, Education, Entertainment, Health, Politics, Science, Sports,** or **Technology**.

The project compares several classical machine-learning models using Bag-of-Words (BoW) and TF-IDF text representations. It also tunes a Linear Support Vector Machine (SVM), evaluates the final model using multiple metrics, analyses errors, and saves the trained pipeline for inference on new articles.

### Objectives

1. Collect and inspect a publicly available news dataset.
2. Clean and normalize article text.
3. Extract unigram and bigram features using BoW and TF-IDF.
4. Train and compare seven classical machine-learning models.
5. Tune the strongest linear model using cross-validated grid search.
6. Evaluate accuracy, macro precision, macro recall, macro F1, weighted F1, confusion matrix, and 5-fold cross-validation.
7. Demonstrate inference and save the trained classifier.

## 2. Dataset Description

### Source

The dataset is the [News Classification Dataset 2026](https://www.kaggle.com/datasets/mmumairkhattak/news-classification-dataset-2026-nlp-and-tc) from Kaggle. The local file is [`news_classification_dataset.csv`](./news_classification_dataset.csv).

### Dataset profile

| Property       |                                                    Value |
| -------------- | -------------------------------------------------------: |
| Articles       |                                                   30,000 |
| Classes        |                                                        8 |
| Text fields    |                     `Title`, `Article_Text`, `Full_Text` |
| Target field   |                                               `Category` |
| Other metadata | Source, author, region, date, keywords, length, language |

### Class distribution

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

The classes are approximately balanced. The notebook verifies the shape, column types, missing values, and class distribution before modelling.

## 3. Methodology

### 3.1 Data collection and column detection

The notebook searches the working directory and common notebook locations for a CSV whose filename contains `news` or `article`. If the real CSV is unavailable, it creates a clearly labelled synthetic corpus for demonstration only. The target is detected from low-cardinality object columns, with preference for names containing `category`, `label`, `class`, `topic`, `section`, or `type`. The longest text column is selected as the article body and a title column, when present, is prepended.

### 3.2 Text preprocessing

Each article is processed using the following steps:

1. Convert text to lowercase.
2. Remove URLs.
3. Remove HTML tags.
4. Remove numeric characters.
5. Remove punctuation and non-alphabetic symbols.
6. Tokenize by whitespace.
7. Remove English stop words.
8. Remove tokens shorter than three characters.

Empty articles are removed. Repeated text rows are retained because the source dataset contains repeated article templates and removing them would discard the class representation needed for stratified training and evaluation.

### 3.3 Feature extraction

The project compares:

- CountVectorizer / Bag-of-Words with unigrams.
- TF-IDF with unigrams.
- TF-IDF with unigrams and bigrams, sublinear term frequency, and a vocabulary limit.

All vectorizers are placed inside scikit-learn pipelines so vocabulary learning occurs only on the training folds and does not leak test information.

### 3.4 Models

The following seven models are trained using TF-IDF unigram–bigram features:

1. Logistic Regression
2. Linear SVM
3. Multinomial Naive Bayes
4. Complement Naive Bayes
5. Decision Tree
6. Random Forest
7. K-Nearest Neighbors

The baseline models use a fixed random seed where supported. The strongest baseline is tuned using `GridSearchCV` over TF-IDF n-gram range, minimum document frequency, vocabulary size, and Linear SVM regularization parameter `C`.

## 4. Experiments and Evaluation

The dataset is divided into training and test partitions using a stratified 80:20 split. Stratification preserves the category proportions. Baseline models are evaluated on the held-out test set and by 5-fold stratified cross-validation on the training set.

### Evaluation metrics

- **Accuracy:** proportion of correctly classified articles.
- **Macro precision:** average precision across categories, treating each category equally.
- **Macro recall:** average recall across categories.
- **Macro F1:** harmonic mean of macro precision and macro recall; useful for balanced multiclass comparison.
- **Weighted F1:** F1 weighted by category frequency.
- **5-fold CV F1:** stability estimate across five training folds.
- **Confusion matrix:** category-level error pattern.

### Results table

Run all cells in the notebook to regenerate the following table from the local dataset. The notebook prints the exact values and plots a comparison chart.

| Model               | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | 5-fold CV F1 |
| ------------------- | -------: | --------------: | -----------: | -------: | ----------: | -----------: |
| Logistic Regression |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| Linear SVM          |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| Multinomial NB      |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| Complement NB       |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| Decision Tree       |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| Random Forest       |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |
| KNN (k=5, cosine)   |    1.000 |           1.000 |        1.000 |    1.000 |       1.000 |        1.000 |

The executed real-dataset run selected Logistic Regression as the best baseline. Grid search selected Linear SVM with `C=0.1`, unigram features, `min_df=1`, and `max_features=20000`; its test accuracy and macro F1 were both **1.000**. The notebook also reports per-class precision/recall/F1, a raw and normalized confusion matrix, most informative words per class, and representative misclassified articles.

### Results interpretation and limitation

The perfect scores must be interpreted cautiously. The supplied CSV contains many repeated article templates, and repeated rows are retained so that all eight categories remain represented after preprocessing. Randomly splitting these rows allows near-identical text to occur in both training and test partitions, which can produce optimistic scores. A production evaluation should deduplicate or group-split by source/article identity before reporting generalization performance.

## 5. Execution Instructions

### Requirements

```bash
pip install -r requirements.txt
```

### Run the notebook

Open [`Exp10_News_Classification_MiniProject.ipynb`](./Exp10_News_Classification_MiniProject.ipynb) in Jupyter Notebook or VS Code and select **Run All**.

```bash
jupyter notebook Exp10_News_Classification_MiniProject.ipynb
```

The notebook automatically looks for `news_classification_dataset.csv`. It should be kept beside the notebook for a reproducible real-dataset run.

### Use the saved model

The final fitted pipeline is saved as [`news_classifier.joblib`](./news_classifier.joblib). The notebook exposes:

```python
predict_news("The central bank raised interest rates as inflation increased")
```

The function cleans the input using the same preprocessing logic and returns the predicted category.

## 6. Repository Contents

| File                                                                                           | Purpose                                                                |
| ---------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------- |
| [`Exp10_News_Classification_MiniProject.ipynb`](./Exp10_News_Classification_MiniProject.ipynb) | Complete implementation, experiments, plots, evaluation, and inference |
| [`news_classification_dataset.csv`](./news_classification_dataset.csv)                         | Kaggle news dataset                                                    |
| [`data_dictionary.csv`](./data_dictionary.csv)                                                 | Dataset column descriptions                                            |
| [`news_classifier.joblib`](./news_classifier.joblib)                                           | Saved final model pipeline                                             |
| [`requirements.txt`](./requirements.txt)                                                       | Python dependencies                                                    |
| [`Mini_project_guideline_Exp 10.docx`](./Mini_project_guideline_Exp%2010.docx)                 | Assignment guideline                                                   |

## 7. Conclusion

This project implements the complete NLP workflow required by Experiment 10: data collection, text cleaning, tokenization, feature extraction, model development, model comparison, cross-validation, detailed evaluation, and inference. TF-IDF with n-grams gives the classifiers informative sparse features, while Linear SVM and Logistic Regression provide strong baselines for high-dimensional text.

The final model should be selected using macro F1, cross-validation stability, confusion-matrix behaviour, and practical inference speed rather than accuracy alone. The dataset and model are intended for academic experimentation and should not be treated as a production news moderation or editorial system without further validation.

## 8. Future Scope

- Use Word2Vec, GloVe, fastText, LSTM/CNN, or transformer models such as BERT and DistilBERT.
- Add lemmatization, named-entity features, and domain-specific phrase features.
- Investigate class imbalance, probability calibration, and confidence thresholds.
- Test multilingual and Hindi/Hinglish news classification.
- Add explainability with SHAP or LIME.
- Deploy the saved pipeline through Streamlit, Flask, or FastAPI.
- Add automated tests, experiment tracking, model monitoring, and data-drift detection.

## 9. Presentation Outline

The report maps directly to the presentation requirements:

1. **Title slide:** project title, four team placeholders, system IDs, subject, faculty, and institution.
2. **Introduction:** motivation and objectives.
3. **Dataset description:** source, fields, size, classes, and distribution.
4. **Methodology:** preprocessing, BoW/TF-IDF, models, and tuning.
5. **Experiment results:** comparison table, CV scores, confusion matrix, and error analysis.
6. **Conclusion and future scope:** findings, limitations, and extensions.
