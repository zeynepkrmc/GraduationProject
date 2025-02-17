
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import joblib
from matplotlib import scale
import numpy as np

app = Flask(__name__)

# Model ve yardımcı dosyaları yükleme
model = joblib.load("best_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")  # Scaler burada tanımlandı

@app.route("/")
def index():
    return render_template("index.html")

@app.route('/predict', methods=["GET","POST"])
def predict():
    if request.method == "POST":
        try:
            data = request.json

            # Eksik veya yanlış veri kontrolü
            required_fields = ['disease','fever', 'cough', 'fatigue', 'breathing', 'age', 'gender', 'bloodPressure', 'cholesterol']
            for field in required_fields:
                if field not in data:
                    return jsonify({"error": f"Missing field: {field}"}), 400

            # Özellikleri alma ve encode etme
            features = [
                label_encoders['Disease'].transform([data['disease']])[0],
                label_encoders['Fever'].transform([data['fever']])[0],
                label_encoders['Cough'].transform([data['cough']])[0],
                label_encoders['Fatigue'].transform([data['fatigue']])[0],
                label_encoders['Difficulty Breathing'].transform([data['breathing']])[0],
                int(data['age']),
                label_encoders['Gender'].transform([data['gender']])[0],
                label_encoders['Blood Pressure'].transform([data['bloodPressure']])[0],
                label_encoders['Cholesterol Level'].transform([data['cholesterol']])[0]
            ]

            # Normalize etme
            input_features = scaler.transform([features]) #scaler

            # Scaler yerine doğrudan NumPy array'e dönüştürme eklendi
            # input_features = np.array([features])

            # Tahmin yapma
            prediction = model.predict(input_features)
            prediction_label = "Positive" if prediction[0] == 1 else "Negative"
            #session['prediction'] = prediction
            print(f"prediction: {prediction}")
            return jsonify({"Prediction": prediction_label})
        except KeyError as e:
            return jsonify({"error": f"Missing or invalid field: {str(e)}"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500
    return render_template("index.html")
if __name__ == '__main__':
    app.run(port=8080, debug=True)
