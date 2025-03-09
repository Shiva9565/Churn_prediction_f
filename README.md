# Churn-Prediction-ANN-

# Customer Churn Prediction

Welcome to the **Customer Churn Prediction** application! This project leverages **deep learning** and **machine learning** techniques to predict customer churn based on various financial and behavioral factors. Developed as part of my data science journey, this project showcases AI-driven predictive analytics for business insights.

![Churn Prediction Banner](images/banner.png)

## ✨ Features
- **📊 Customer Churn Prediction:** Predicts the probability of customer churn using a trained deep learning model.
- **🔄 Data Preprocessing:** Uses label encoding, one-hot encoding, and feature scaling for improved predictions.
- **🎛️ Interactive UI:** Built with **Streamlit** for an intuitive user experience.
- **📈 Model Deployment:** Utilizes **TensorFlow** for inference with a pre-trained model.
- **⚡ Real-Time Predictions:** Users can input their details and get immediate churn probability.

## 🛠️ Technologies Used
- **Python:** Core programming language for AI and ML tasks [https://www.python.org/].
- **Streamlit:** Web framework for interactive data apps [https://streamlit.io/].
- **TensorFlow/Keras:** Deep learning framework for model training and inference [https://www.tensorflow.org/].
- **scikit-learn:** Data preprocessing and feature engineering tools [https://scikit-learn.org/].
- **Pandas & NumPy:** Data handling and numerical computation [https://pandas.pydata.org/].

## 🚀 Getting Started
Clone the repository:
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Ensure the following files are present in the project directory:
- `model.h5` (Trained deep learning model)
- `label_encoder_gender.pkl` (Label encoder for gender)
- `One_hot_encoder_geography.pkl` (One-hot encoder for geography)
- `scaler.pkl` (Standard scaler for feature scaling)

Run the application:
```bash
streamlit run app.py
```

## 🗂️ Project Structure
- **`app.py`**: Main Streamlit app for churn prediction.
- **`model.h5`**: Pre-trained deep learning model.
- **`label_encoder_gender.pkl`**: Label encoding for gender.
- **`One_hot_encoder_geography.pkl`**: One-hot encoding for geography.
- **`scaler.pkl`**: StandardScaler for numerical features.
- **`requirements.txt`**: List of required dependencies.
- **`images/`**: Contains screenshots and banners.

## 🔒 Data Preprocessing
The application applies:
- **Label Encoding:** Converts categorical gender data into numerical format.
- **One-Hot Encoding:** Encodes geographical data into machine-readable form.
- **Feature Scaling:** Standardizes numerical data for better model performance.

## 📈 Model & Prediction
- Uses a **deep learning model** built with TensorFlow.
- Takes user inputs and transforms them using preprocessing techniques.
- Predicts **churn probability** and provides an actionable insight.

## 📊 AI-Driven Insights
The AI model analyzes key customer attributes like **credit score, age, account balance, and activity level** to determine the likelihood of churn. This helps businesses make data-driven decisions and improve customer retention strategies.

## 🌐 Live Demo & Website
Check out the live application at:
[Your Website Link](https://yourwebsite.com)

## 📜 License
This project is licensed under the **MIT License**.

## 👨‍💻 Author
[Your Name](https://github.com/your-username)

---
This project is an exciting demonstration of **AI in business intelligence**. Feel free to contribute or improve!

