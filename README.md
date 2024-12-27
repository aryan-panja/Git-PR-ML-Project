# GitHub PR Spam Detection

## Project Overview
This project aims to classify GitHub Pull Requests (PRs) as either **SPAM** or **NOT_SPAM** using machine learning techniques. It is designed to help maintainers of open-source repositories save time by automating the identification of low-value contributions.

---

## Submission Details
- **Course**: UML 501 Machine Learning  
- **Project Title**: GitHub PR Spam Detection  
- **Submitted by**:  
  - Aryan Panja (102217034)  
  - Prabhjot Singh (102217062)  
  - B.E. 3rd Year - COPC  
- **Submitted to**: Dr. Archana Singh  
- **Department**: Computer Science and Engineering  
- **Institute**: Thapar Institute of Engineering and Technology, Patiala  
- **Date**: November 2024  

---

## Problem Statement
With the growing popularity of GitHub, there has been a rise in low-effort contributions to open-source projects. These trivial pull requests often:
- Modify files without functional improvement.
- Add contributors' names without meaningful additions.
- Make minor changes to documentation without value.

Such contributions clutter the review process, wasting maintainers' time. This project automates the classification of PRs into:
- **SPAM**: Low-value or trivial contributions.
- **NOT_SPAM**: Meaningful contributions.

---

## Key Features
- **Data Sources**: PR data from popular repositories like React, TensorFlow, and Django.
- **Feature Engineering**: Includes TF-IDF for textual analysis and numerical features like lines added/removed.
- **Modeling**: Classification models including Random Forest, Logistic Regression, KNN, and Decision Tree.
- **Evaluation**: Metrics like Accuracy, Precision, Recall, and F1-Score to measure model performance.

---

## Installation
### Prerequisites
Install the dependencies listed in `requirements.txt`:
```bash
pip install -r requirements.txt
```
**Dependencies**:
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `numpy`

---

## Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/aryan-panja/Git-PR-ML-Project.git
   cd Git-PR-ML-Project
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the script to train and evaluate the model:
   ```bash
   python train_model.py
   ```
4. Evaluate the model using your dataset:
   ```bash
   python evaluate_model.py --input your_dataset.csv
   ```
5. Predict PR classifications:
   ```bash
   python predict.py --input test_pr_data.csv
   ```

---

## Model Performance
### Results Without Text Features (`title`)
| Metric       | Random Forest | Logistic Regression |
|--------------|---------------|---------------------|
| Accuracy     | 88.79%        | 89.65%              |
| Precision    | 88.23%        | 87.03%              |
| Recall       | 86.53%        | 90.38%              |
| F1-Score     | 87.37%        | 88.67%              |

### Results With Text Features (`title`)
| Metric       | Random Forest | Logistic Regression | KNN   | Decision Tree |
|--------------|---------------|---------------------|-------|---------------|
| Accuracy     | 96.55%        | 94.82%              | 96.55%| 92.52%        |
| Precision    | 92.85%        | 94.23%              | 92.85%| 89.15%        |
| Recall       | 100%          | 94.23%              | 100%  | 94.87%        |
| F1-Score     | 96.29%        | 94.23%              | 96.29%| 91.92%        |

### Confusion Matrix (Random Forest With `title` Feature)
![Confusion Matrix](images/confusion_matrix.png)

### Comparison with Benchmarks
| Model        | Precision     | Recall        | F1-Score     |
|--------------|---------------|---------------|--------------|
| Research DT  | 95.53%        | 99.01%        | 97.23%       |
| Ours DT      | 89.15%        | 94.87%        | 91.92%       |
| Research LR  | 75.17%        | 74.29%        | 68.35%       |
| Ours LR      | 94.23%        | 94.23%        | 94.23%       |

---

## Methodology
### Data Collection
PRs were collected from:
- **React**
- **TensorFlow**
- **Django**

### Data Preprocessing
- **Numerical Features**: Comments, files changed, lines added/removed.
- **Text Features**: TF-IDF vectorization of PR titles.
- **Normalization**: Applied using `StandardScaler`.

### Models
- **Random Forest**
- **Logistic Regression**
- **KNN**
- **Decision Tree**

---

## Impact
This project provides a scalable, real-world solution for:
- Reducing noise in open-source contributions.
- Encouraging high-quality contributions.
- Saving maintainers' time.

---

## References
1. A. Mohamed, et al. "Predicting Which Pull Requests Will Get Reopened in GitHub" (2018).  
   [IEEE Explore](https://ieeexplore.ieee.org/abstract/document/8719563)  
2. [GitHub Repository](https://github.com/aryan-panja/Git-PR-ML-Project)

---

For more details, please refer to the [documentation](docs/documentation.pdf).
