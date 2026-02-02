# Author Identification using Machine Learning (Java + Weka)

This project focuses on **author identification** — a text classification task where the goal is to determine the author of a given text based on writing style and linguistic patterns.

The project was developed in **Java** using **Weka** library as part of an academic machine learning assignment and demonstrates the practical use of classical ML techniques for text analysis.

---

## Project Overview

Given a collection of text samples written by multiple authors, the program:
- preprocesses textual data,
- extracts numerical features,
- trains machine learning models,
- and predicts the most likely author of unseen text samples.

The project shows the practical use of k-means classification algorithms.

---

## Programming languages & Tools

- **Java**
- **Weka Library**

---

## Features

- Text preprocessing and normalization
- Feature extraction for text classification
- Author identification using machine learning
- Training and evaluation of classification models
- Performance analysis using standard evaluation metrics

---

## Project Structure

ml-project-author-id/
- src/ # Java source code
- res/ # books used for creating datasets and testing
- data_fragments/ # Fragmented data (400 tokens per fragment)
- data_fragments_200/ # Fragmented data (200 tokens per fragment)
- data_fragments_600/ # Fragmented data (600 tokens per fragment)
- author_fragments.arff # Dataset: .arff file created while testing (from data_fragments)
- author_fragments_200.arff # Dataset: .arff file created while testing (from data_fragments_200)
- author_fragments_600.arff # Dataset: .arff file created while testing (from data_fragments_600)
- README.md

---

## How to Run

- To build a dataset:
    - Configure the parameters for building a dataset (class BuildDataset):
        -  value of field FRAG_TOKENS (number of tokens per fragment)
        -  value of field rawDir (raw directory)
        -  value of field fragsDir (fragmented data directory))
        -  argument passed to the method **new File(*filename*)** at line 34 (filename = name of .arff file to be created)
     
- To run author identification:
    - Configure the parameters for string-to-word vector (class AuthorID):
        - value of field MIN_TERM_FREQ
        - value of field NGRAM_MIN
        - value of field NGRAM_MAX
    - Specify the data source (class AuthorID):
        - argument passed to the method **new DataSource(*filename*)** at line 45 (filename = name of .arff file to be used)

---

## Evaluation

Evaluation results displayed when presenting the project will be added soon.

---

## What I learned

- Applying machine learning techniques to real-world textual data using Java
- Using the Weka library for classification tasks
- Importance of preprocessing and feature extraction
- Evaluating and comparing ML models based on performance metrics
- Structuring Java ML projects in a clean and maintainable way

---

## Future improvements

- Trying alternative algorithms
- Further parameter tuning and testing
- Improving scalability to fit bigger datasets
- Improving efficiency through saving stwv for some configurations

---
