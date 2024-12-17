from flask import Flask, jsonify, request

app = Flask(__name__)
import pandas as pd
import numpy as np
import datetime as dt
import sklearn
import pickle
from collections import Counter
# Modeling
from sklearn.preprocessing import LabelEncoder
# !pip install xgboost
# import xgboost as xgb
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import requests
from datetime import datetime, timedelta
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
# from wordcloud import WordCloud
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, BaggingClassifier
import warnings

warnings.simplefilter('ignore')
sns.set_theme(style="dark")

from flask import Flask, request, jsonify
# import gradio as gr
# import gradio as gr
import joblib
import pandas as pd
import numpy as np
import requests
import tensorflow as tf
from firebase_admin import firestore

# app = Flask(_name_)

# Initialize Firebase
import firebase_admin
from firebase_admin import credentials, firestore

cred = credentials.Certificate("/home/urnisha/Downloads/beaches-b18be-firebase-adminsdk-ng9uo-e46a7a7be1.json")
firebase_admin.initialize_app(cred)
db = firestore.client()
# Load Models
scaler = joblib.load("/home/urnisha/Downloads/finalweather.pkl")  # Load scaler
model = tf.keras.models.load_model("/home/urnisha/Downloads/summarygenerate.h5")  # Load Keras model
# from sklearn.tree import DecisionTreeClassifier
# import joblib

# # Load the old model
# with open("/home/urnisha/Downloads/MAIN_decision_tree_model.pkl", "rb") as f:
#     decision_model = joblib.load(f)

# # Save it with the current scikit-learn version
# joblib.dump(decision_model, "/path/to/new_decision_tree_model.pkl")

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def fetch_weather_data(api_key, endpoint, lat, lng, days=10):
    headers = {
        "Authorization": api_key
    }
    end_time = datetime.utcnow()
    start_time = datetime.utcnow() - timedelta(days=days)

    params = {
        "lat": lat,
        "lng": lng,
        "params": "precipitation,airTemperature,humidity,windSpeed,windDirection,visibility,pressure,currentSpeed,seaLevel,swellHeight,swellPeriod,waveHeight,wavePeriod",
        "start": int(start_time.timestamp()),
        "end": int(end_time.timestamp())
    }

    response = requests.get(endpoint, headers=headers, params=params)
    if response.status_code == 200:
        data = response.json()
        weather_data = []
        for entry in data.get("hours", []):
            timestamp = entry.get("time")
            date = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M:%S')
            precipitation = entry.get("precipitation", {}).get("noaa", np.nan)
            temperature = entry.get("airTemperature", {}).get("noaa", np.nan)
            humidity = entry.get("humidity", {}).get("noaa", np.nan)
            wind_speed = entry.get("windSpeed", {}).get("noaa", np.nan)
            wind_bearing = entry.get("windDirection", {}).get("noaa", np.nan)
            visibility = entry.get("visibility", {}).get("noaa", np.nan)
            pressure = entry.get("pressure", {}).get("noaa", np.nan)
            weather_data.append(
                [date, precipitation, temperature, humidity, wind_speed, wind_bearing, visibility, pressure])

        weather_df = pd.DataFrame(weather_data,
                                  columns=["date", "precipitation", "temperature", "humidity", "wind_speed",
                                           "wind_bearing", "visibility", "pressure"])
        return weather_df
    else:
        print(f"Error: {response.status_code}, {response.text}")
        return None


# Fetch Latitude and Longitude from Firebase based on place name
def get_lat_lon_from_firebase(place_name):
    # Assuming 'place_name' matches the document ID (e.g., "Goa Beach")
    places_ref = db.collection('beaches')  # 'beaches' collection in Firestore
    place_doc = places_ref.document(place_name).get()  # Fetch document using place_name as doc ID

    if place_doc.exists:
        place_data = place_doc.to_dict()
        return place_data['latitude'], place_data['longitude']
    else:
        print(f"Place {place_name} not found in Firestore.")
        return None, None


def classify_temperature(temp):
    if temp > 30:
        return "Hot"
    elif temp >= 15:
        return "Mild"
    else:
        return "Cool"


def classify_humidity(humidity):
    return "High" if humidity >= 60 else "Normal"


def classify_wind_speed(wind_speed):
    return "Strong" if wind_speed >= 10 else "Weak"


def fetch_ocean_data(station, parameter, api_key):
    # Replace placeholders with actual values
    api_url = f"https://gemini.incois.gov.in/OceanDataAPI/api/wqns/{station}/{parameter}"

    headers = {
        'Authorization': api_key
    }

    try:
        # Send GET request to the API
        response = requests.get(api_url, headers=headers)

        # Check if the response is successful
        if response.status_code == 200:
            data = response.json()
            return data[parameter][0]
            # /print("Data fetched successfully!")
            # ph = data['ph'][0]
            # sl= data['salinity'][0]
        else:
            print(f"Error: Received status code {response.status_code}")
            return None

    except requests.RequestException as e:
        print("Fetch error:", e)
        return None


# Example usage
# Replace with your API key


# Prediction Function
def predict_risk(place_name):
    # Step 1: Get latitude and longitude from Firebase
    lat, lng = get_lat_lon_from_firebase(place_name)

    if lat is None or lng is None:
        return f"Place {place_name} not found in Firebase."

    station = "Kochi"  # Replace with actual station
    wave_api_key = "INCOIS_API_KEY"
    paramswv = ["ph", "salinity", "dissolvedoxygen", "dissolvedmethane"]
    # wv_data = []
    # for p in paramswv:
    #     wv_data.append(float(fetch_ocean_data(station, p, wave_api_key)))
    # rc = 0
    # if wv_data[0] < 6.5 or wv_data[0] > 8.5:
    #     rc = rc + 1
    # if wv_data[1] > 1000:
    #     rc = rc + 1
    # if wv_data[2] < 5:
    #     rc = rc + 1
    # if wv_data[3] > 10:
    #     rc = rc + 1

    # Step 2: Fetch weather data using the API
    api_key = "stormglass_api_key"
    endpoint = "https://api.stormglass.io/v2/weather/point"
    weather_df = fetch_weather_data(api_key, endpoint, lat, lng)

    if weather_df is None:
        return "Error fetching weather data."

    # Preprocessing (scaling, prediction, etc.)
    # weather_df = weather_df.drop(columns=["date", "precipitation"], axis=1)
    # weather_df = weather_df.round(2)
    # Formatting Date Column. This can be used to identify any seasonality and trends
    weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce')
    # Extracting the relevant components
    weather_df["Time"] = [d.time() for d in weather_df['date']]
    weather_df["Time"] = weather_df["Time"].astype(str)
    weather_df["Time"] = weather_df["Time"].str.split(':').str[0].astype(int)
    weather_df["Date"] = [d.date() for d in weather_df['date']]
    weather_df["Date"] = weather_df["Date"].astype(str)
    weather_df["Year"] = weather_df["Date"].str.split('-').str[0].astype(int)
    weather_df["Month"] = weather_df["Date"].str.split('-').str[1].astype(int)
    weather_df["Day"] = weather_df["Date"].str.split('-').str[2].astype(int)
    weather_df = weather_df.drop(columns=['Date'], axis=1)
    input_df = weather_df.drop(columns=["date", "precipitation"], axis=1)
    input_df = input_df.round(2)

    # Scaling the features
    weather_data_scaled = scaler.transform(input_df)

    # Predict the weather outlook
    predictions = model.predict(weather_data_scaled)
    predicted_class = np.argmax(predictions, axis=1)

    # Reverse encoding for the predicted class
    class_mapping_reverse = {0: 'Clear', 1: 'Overcast', 2: 'Rain'}
    predicted_label = class_mapping_reverse[predicted_class[0]]
    weather_df['Outlook'] = predicted_label

    weather_df = weather_df.drop(columns=['Time', 'Year', 'Month', 'Day'])
    weather_df["day"] = weather_df.date.dt.floor('D')
    daily_activity = weather_df.groupby(["day"]).agg(
        {
            "precipitation": ["mean"],
            "temperature": ["mean"],
            "humidity": ["mean"],
            "wind_speed": ["mean"],
            "wind_bearing": ["mean"],
            "visibility": ["mean"],
            "pressure": ["mean"],
            "Outlook": lambda x: x.mode().iloc[0]  # Taking the first mode for the 'summary' column
        }
    ).reset_index()
    daily_activity = daily_activity.droplevel(axis=1, level=1)
    daily_activity['Temperature'] = daily_activity['temperature'].apply(classify_temperature)
    daily_activity['Humidity'] = daily_activity['humidity'].apply(classify_humidity)
    daily_activity['Wind'] = daily_activity['wind_speed'].apply(classify_wind_speed)
    daily_activity = daily_activity.drop(
        columns=['precipitation', 'temperature', 'humidity', 'wind_speed', 'wind_bearing', 'visibility', 'pressure'])
    daily_activity['Outlook'] = daily_activity['Outlook'].replace('Clear', 'Sunny')
    input_df_2 = daily_activity.drop(columns=["day"], axis=1)

    # Classifying risk
    # daily_activity = weather_df.copy()
    outlook_mapping = {'Sunny': 0, 'Overcast': 1, 'Rain': 2}
    temp_mapping = {
        'Cool': 0,
        'Mild': 1,
        'Hot': 2
    }
    humid_mapping = {
        'High': 0,
        'Normal': 1
    }
    wind_mapping = {
        'Weak': 0,
        'Strong': 1,

    }
    safe_mapping = {
        'no risk': 0,
        'risk': 1,

    }
    # daily_activity['Outlook'] = daily_activity['Outlook'].map(outlook_mapping)
    # input_df_2['Outlook'] = input_df_2['Outlook'].map(outlook_mapping)
    # input_df_2['Temperature'] = input_df_2['Temperature'].map(temp_mapping)
    # input_df_2['Humidity'] = input_df_2['Humidity'].map(humid_mapping)
    # input_df_2['Wind'] = input_df_2['Wind'].map(wind_mapping)
    # risk_prediction = decision_model.predict(input_df_2)

    if (input_df_2['Outlook'] == 'Rain').any():
        risk_level = "Risk"
    elif (input_df_2['Outlook'] == 'Overcast').any() and (input_df_2['Humidity'] == 'High').any():
        risk_level = "Risk"
    else:
        risk_level = "No Risk"
    # risk_count=0
    # for it in risk_prediction:
    #     risk_count+= it

    # if risk_count == 0 and rc == 0:
    #     risk_level = "No Risk"
    # # elif risk_count == 1 :
    # #     risk_level = "Mild Risk"
    # else:
    #     risk_level = "Risk"

    return risk_level
# /print(predict_risk("Kochi Beach"))

@app.route('/api/safety', methods=['POST'])
def safety():
    # data = request.get_json()  # Get input data as JSON
    # features = data['places']  # Extract feature values
    prediction = predict_risk("Kochi Beach")  # Make prediction
    return jsonify({"prediction": prediction})


# Run the app
if __name__ == '_main_':
    app.run(debug=True)
# print(predict_risk('Kochi Beach'))

# # Classifying risk
# daily_activity = weather_df.copy()
# outlook_mapping = {'Sunny': 0, 'Overcast': 1, 'Rain': 2}
# daily_activity['Outlook'] = daily_activity['Outlook'].map(outlook_mapping)
# risk_prediction = decision_model.predict(daily_activity)

# risk_mapping = {0: 'no risk', 1: 'risk'}
# risk_status = risk_mapping[risk_prediction[0]]

# return f"Weather Outlook: {predicted_label}, Risk Status: {risk_status}"
# # Your Gradio function
# def greet(name):
#     return f"Hello, {name}!"

# # Gradio interface
# iface = gr.Interface(fn=greet, inputs="text", outputs="text")
# iface.launch(share=False, prevent_thread_lock=True)

# @app.route("/greet", methods=["POST"])
# def greet_api():
#     data = request.get_json()
#     name = data.get("name")
#     result = greet(name)
#     return jsonify({"message": result})

# if _name_ == "_main_":
#     app.run()
