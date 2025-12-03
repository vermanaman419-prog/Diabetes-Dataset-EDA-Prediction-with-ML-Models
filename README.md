## 🩺 Diabetes Prediction — EDA, Modeling & Deployment on Hugging Face

A complete end-to-end machine learning project analyzing the PIMA Diabetes Dataset, performing EDA, building ML models, and deploying the best model using Gradio + Hugging Face Spaces.

This project walks through data exploration → model building → evaluation → deployment, focusing on both interpretability and real-world usability.

## 📌 Project Summary

The goal of this project is to predict whether a patient is likely to have diabetes based on key medical features such as glucose level, BMI, age, blood pressure, etc.

After performing detailed EDA and training multiple ML models, I deployed the best-performing model (XGBoost) while still analyzing and comparing the results with Random Forest, which was also a strong candidate.

## 🔍 Why XGBoost for Deployment?

During model evaluation:

## ✔️ Random Forest performed very well

- Strong baseline accuracy

- Robust to noise and missing patterns

- Good feature importance explainability

## ✔️ But XGBoost outperformed it slightly

- Better handling of class imbalance

- Higher accuracy & AUC

- Works extremely well on tabular data

- Faster inference for deployment

Since Hugging Face deployment benefits from models that predict quickly and consistently, XGBoost was chosen as the production model, while Random Forest remains a valuable benchmark model.

# 🚀 Live Deployed Model (XGBoost)

# 👉 Hugging Face App: https://huggingface.co/spaces/naman419/Diabetes-Dataset-EDA-Prediction-with-ML-Models
Predict diabetes using an interactive Gradio UI.

# 🧪 Workflow Overview
# 1️⃣ Data Preparation & Cleaning

- Treated zero-values in critical columns

- Handled missing or inconsistent data

- Standardized numerical features

- Prepared features for ML training

# 2️⃣ Exploratory Data Analysis (EDA)

Generated detailed visualizations including:

- 📊 Distribution plots
- 📈 Boxplots (outlier detection)
- 🔥 Correlation Heatmap
- 👥 Relationship plots between features

# Key insights identified:

- Glucose is the strongest predictor

- Higher BMI and Age correlate with diabetes

- Insulin levels are highly skewed and require transformation

# 3️⃣ Machine Learning Models Built
#✔ Random Forest (Benchmark Model)

- Strong accuracy and F1-score

- Great interpretability

- Feature importance ranking aligned with medical patterns

# ✔ XGBoost (Final Deployed Model)

- Best overall predictive performance

- Higher ROC-AUC

- Better generalization

- Chosen for deployment due to stability + accuracy



# 🧰 Tools & Technologies

- Programming	  :   Python (Pandas, NumPy, Scikit-Learn, XGBoost, Seaborn, Matplotlib)
- ML Models	    :   Random Forest, XGBoost
- Environment	  :   Google Colab
- Visualization	:   Seaborn, Matplotlib
- Deployment	  :   Gradio UI, Hugging Face Spaces
- Model Storage	:   joblib


# 📊 Key Takeaways

- EDA revealed strong relationships between glucose, BMI, age, and diabetes likelihood

- Random Forest provided excellent baseline accuracy and interpretability

- XGBoost delivered superior performance, making it the optimal choice for deployment

- End-to-end pipeline from raw data → EDA → ML → deployment using Hugging Face

# 👨‍💻 Author

Naman Verma
📍 Gurugram, India
📧 vermanaman419@gmail.com
🔗 LinkedIn : www.linkedin.com/in/naman419



---


