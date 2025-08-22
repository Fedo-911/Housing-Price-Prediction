# 🏡 California Housing Price Prediction

This project predicts **median house prices** in California using the **California Housing dataset**.  
It uses **machine learning (Random Forest Regressor)** to model housing prices based on features like median income, average rooms, population, and location (latitude & longitude).

The project also includes a **Streamlit web app** for interactive predictions and a **Jupyter Notebook** for training and evaluation.



## 🌍 Live Demo

Try the app here:  
👉 [California Housing Price Prediction App](https://housing-price-prediction-fardeen.streamlit.app/)



## 📂 Project Structure
housing_price_prediction/

│── app.py # Streamlit app

│── requirements.txt # Dependencies

│── model_training.ipynb # Model training notebook

│── README.md # Project documentation




## 🚀 Features
- Train a **Random Forest Regressor** on the California Housing dataset.
- Predict house prices based on input features:
  - Median Income  
  - House Age  
  - Average Rooms  
  - Average Bedrooms  
  - Population  
  - Average Occupancy  
  - Latitude  
  - Longitude  
- Interactive **Streamlit UI**:
  - Enter custom housing data → get predicted price  
   



## 🛠️ Installation

Clone the repository:

git clone https://github.com/Fedo-911/Housing-Price-Prediction.git 

cd housing-price-prediction



## Install dependencies:
pip install -r requirements.txt



## ▶️ Running the Streamlit App
streamlit run app.py

The app will open in your browser at http://localhost:8501/



## 📊 Example Usage

## Input :
 
 **MedInc:**     8.3252
 
**HouseAge:**   41

**AveRooms:**   6.9841

**AveBedrms:**  1.0238

**Population:** 322

**AveOccup:**   2.5556

**Latitude:**   37.88

**Longitude:** -122.23

## Output:

**💰 Predicted Median House Value: $423,879.27**



## 🧠 Model Training

The full training process (data preparation, training, and evaluation) is available in the notebook:

**📓** [`Model Training Notebook`](Notebooks/Housing_Price_Prediction.ipynb) 



## 📈 Model Performance

• **Algorithm:** Random Forest Regressor


• **Dataset:** California Housing (1990 Census data)


• **R² Score ≈** 0.75 (good predictive power)


• **Evaluation Metrics:** R² Score, RMSE



## 📌 Requirements

• Python 3.8+

• Streamlit

• Scikit-learn

• Pandas

• Numpy

**Install with:**

pip install -r requirements.txt



## 🙌 Acknowledgements

• **Dataset:** [California Housing Dataset (Scikit-learn)](https://scikit-learn.org/stable/datasets/real_world.html#california-housing-dataset)

• **Framework:** [Streamlit](https://streamlit.io/)

