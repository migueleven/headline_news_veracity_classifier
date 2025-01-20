# Project: News Headlines Classification (Fake vs Real)

This project aims to build a supervised learning model to classify news headlines as **fake** or **real**. A Natural Language Processing (NLP) pipeline is implemented to preprocess the data, apply vectorization techniques, and train classification models.

---

## 1. Project Workflow

### **Main Steps**
1. **Data Loading and Exploration**:
   - Load the dataset containing two main columns:
     - `Headline`: The headline text.
     - `Veracity`: Binary label indicating whether the headline is fake (`0`) or real (`1`).
   - Explore the distribution of the labels to understand class balance.

2. **Text Preprocessing**:
   - Convert headlines to lowercase.
   - Remove special characters and punctuation.
   - *(Optional)* Remove stopwords and apply lemmatization.

3. **Text Vectorization**:
   - Use `CountVectorizer` to convert the headlines into a numerical representation based on word frequency.
   - Experiment with bigrams (`ngram_range=(1, 2)`) to capture word relationships.

4. **Training and Evaluating Base Models**:
   - Test multiple supervised classification models:
     - Naive Bayes
     - Logistic Regression
     - Random Forest
     - SVM
   - Evaluate each model on the test set using metrics such as accuracy and classification reports.

5. **Selecting the Best Model**:
   - The best-performing initial model was **Logistic Regression**, with an accuracy of ~92.92%.

6. **Hyperparameter Tuning**:
   - Perform hyperparameter tuning on the winning model (`Logistic Regression`) using `GridSearchCV`.
   - Explore combinations of `C`, `solver`, and `penalty`.

7. **Evaluating the Optimized Model**:
   - Recalculate performance metrics and visualize the confusion matrix.
   - The optimized model maintained similar performance to the default model.

---

## 2. Tools Used

- **Language**: Python 3
- **Libraries**:
  - `pandas`: For handling tabular data.
  - `numpy`: For mathematical operations.
  - `scikit-learn`: For preprocessing, vectorization, and model training.
  - `seaborn` and `matplotlib`: For metric visualization.
  - `nltk`: For text cleaning and tokenization.

---

## 3. Models and Configurations

### **Models Evaluated**
1. **Naive Bayes**:
   - A fast and efficient baseline model for sparse data.

2. **Logistic Regression**:
   - A robust linear model with regularization.
   - Initial winner with an accuracy of 92.92%.

3. **Random Forest**:
   - Tree-based model.
   - Competitive performance but did not surpass Logistic Regression.

4. **SVM**:
   - Maximizes decision boundaries.
   - Requires more training time.

### **Hyperparameters Tuned for Logistic Regression**
- `C`: [0.01, 0.1, 1, 10, 100]
- `penalty`: ['l1', 'l2']
- `solver`: ['liblinear', 'lbfgs']

---

## 4. Evaluation Metrics

- **Accuracy**: Proportion of correct predictions.
- **Classification Report**: Detailed metrics (precision, recall, F1-score) for each class.
- **Confusion Matrix**: Visualization of true positives, true negatives, false positives, and false negatives.

---

## 5. Final Results

- **Winning Model**: Logistic Regression (optimized).
- **Performance**:
  - Accuracy: 92.92%
  - Balanced precision and recall across classes.
- **Observations**:
  - Hyperparameter tuning did not significantly improve the performance over the default model.
  - The default model settings are near-optimal for this problem.

---

## 6. Next Steps

1. **Improving Preprocessing**:
   - Implement TF-IDF instead of `CountVectorizer`.
   - Experiment with advanced embeddings (Word2Vec, GloVe, or BERT).

2. **Collect More Data**:
   - A larger dataset could help improve performance and generalization.

3. **Try Advanced Models**:
   - Gradient boosting models like XGBoost or LightGBM.
   - Transformer-based models (e.g., BERT, RoBERTa) for deeper semantic analysis.

---

## 7. Contact

If you have any questions or suggestions, feel free to reach out:
- **Email**: miguelchamizo10@gmail.com
- **GitHub**: [Migueleven](https://github.com/migueleven)
- **Linkedin**: [Miguel Ángel Chamizo Sánchez](https://www.linkedin.com/in/miguelangelchamizosanchez/)
