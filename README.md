# ❤️ Heart Disease Prediction Project

A simple machine learning project that predicts the chances of heart disease.
It covers everything from cleaning the data to training models, checking results, and even building a simple Streamlit app for real-time predictions.

---

## 📂 Project Structure

```
Heart_Disease_Project/
│── data/                 # dataset
│── notebooks/            # step-by-step Jupyter notebooks
│── models/               # saved trained model (.pkl)
│── ui/                   # Streamlit app
│── results/              # evaluation metrics
│── README.md             # project documentation
│── requirements.txt      # required libraries
│── .gitignore            # ignored files
```

---

## 🚀 Features

* Cleans and prepares medical data for analysis
* Reduces data complexity using PCA and feature selection (RFE)
* Trains supervised models: Logistic Regression, Decision Tree, Random Forest, SVM
* Experiments with unsupervised models: K-Means, Hierarchical Clustering
* Optimizes models using GridSearchCV and RandomizedSearchCV
* Includes a Streamlit app for quick, real-time predictions

---

## ▶️ How to Run

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/Heart_Disease_Project.git
   cd Heart_Disease_Project
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app**

   ```bash
   streamlit run ui/app.py
   ```

4. **(Optional) Share the app online**
   Use Ngrok following the steps in `deployment/ngrok_setup.txt`.

---
