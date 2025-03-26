# 🎬 Movie Revenue Prediction 💰  

This project is a **Machine Learning-based application** that predicts a **movie's gross revenue** based on **budget, number of votes, runtime, and other important factors**.  

The project applies **Linear Regression, Polynomial Regression, Random Forest, and XGBoost models** to find the best predictions. We use **GridSearchCV** to fine-tune hyperparameters and deploy an **interactive Streamlit web app** for real-time predictions.  

---

## 🚀 Features  

✔ **Data Analysis & Visualization** – Correlation heatmaps, bar plots, scatter plots, etc.  
✔ **Machine Learning Models** – Linear Regression, Polynomial Regression, Random Forest, and XGBoost.  
✔ **Hyperparameter Optimization** – Uses `GridSearchCV` for the best model settings.  
✔ **Feature Importance Analysis** – Identifies the most important factors affecting revenue.  
✔ **Interactive Web App** – A **Streamlit-based UI** where users can enter movie details and get predictions.  
✔ **Model Persistence** – Saves the best-trained model for future use.  

---

## 📂 Project Structure  
```
Movie_Revenue_Prediction/
│── 📁 data/ # Contains raw dataset
│ ├── movies.csv # Raw movie dataset
│
│── 📁 models/ # Contains scripts for ML training
│ ├── analysis_and_charts.py # Data analysis & visualization (correlation heatmaps, graphs)
│ ├── model.py # Machine Learning model implementation
│ ├── trained_model.py # Script for model training & tuning
│
│── 📄 app.py # Streamlit web app for predictions
│── 📄 requirements.txt # Dependencies needed for the project
│── 📄 README.md # Documentation
```


---

## 📊 Machine Learning Models Used  

- **Linear Regression** 📉  
- **Polynomial Regression** 🔵  
- **Random Forest** 🌳  
- **XGBoost** ⚡ (Best performing model)  

We applied **hyperparameter tuning** using `GridSearchCV` to optimize performance.  

---

## ⚙️ Installation & Setup  

### **1️⃣ Clone the Repository**  

```bash
git clone https://github.com/Ra638/Movie_Revenue_Prediction.git  
cd Movie_Revenue_Prediction  
pip install -r requirements.txt  
python models/analysis_and_charts.py  
python models/trained_model.py  
python models/trained_model.py
```

🔥 Model Performance
---
```
Model                           MAE   MSE        R² Score  
---------------------------------------------------------  
Linear Regression               70M   1.77e+16   0.50  
Polynomial Regression (Degree 2) 49M   1.16e+16   0.67  
Polynomial Regression (Degree 3) 49M   1.16e+16   0.67  
Random Forest                   49M   1.09e+16   0.69  
XGBoost (Tuned)                 49M   9.69e+15   0.77  
```
✅ XGBoost performs the best with R² = 0.77!

🎨 Web App Preview
---
The project includes a Streamlit Web App where users can enter details and get predictions.

🛠 Future Improvements
---
🚀 Enhance Feature Engineering – Add more relevant features
📈 Try Deep Learning Models – Test Neural Networks for better accuracy
🌐 Deploy Online – Host the app on AWS/GCP/Heroku

🤝 Contributing
---
Want to improve this project? Feel free to fork and submit a Pull Request 🎯
