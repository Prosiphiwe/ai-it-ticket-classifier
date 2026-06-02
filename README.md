# 🧠 AI IT Ticket Classifier

## 🚀 Overview
This project is an AI-powered IT support ticket classifier that predicts the category and priority of IT issues using Machine Learning and Natural Language Processing (NLP).

It also generates a ticket ID and saves tickets automatically like a real IT Service Management system.

---

## ⚙️ Features
- Predicts IT ticket category (Network, Hardware, Software, etc.)
- Predicts ticket priority (Low, Medium, High)
- Shows confidence scores
- Displays Top 3 predictions
- Auto-generates ticket ID (INC-001)
- Saves tickets to CSV database
- Streamlit web interface

---

## 🧠 Tech Stack
- Python
- Scikit-learn
- NLP (TF-IDF)
- Streamlit
- Pandas
- Joblib

---

## 📂 Project Structure
- `app.py` → Streamlit web app  
- `train.py` → Model training script  
- `data.csv` → Training dataset  
- `model_category.pkl` → Trained category model  
- `model_priority.pkl` → Trained priority model  
- `vectorizer.pkl` → NLP vectorizer  

---

## ▶️ How to Run

```bash
pip install -r requirements.txt
streamlit run app.py