from flask import Flask, request, jsonify
from keras.models import load_model
import numpy as np

app = Flask(__name__)

# Carica il modello Keras
model = load_model('../classification_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    # Ottieni i dati in JSON
    data = request.get_json(force=True)

    # Prepara i dati per la previsione
    input_data = np.array(data['input']).reshape(1, -1)

    # Fai la previsione
    prediction = model.predict(input_data)

    # Restituisci il risultato
    return jsonify(prediction.tolist())


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)