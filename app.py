from flask import Flask, jsonify, request
from sklearn.linear_model import LinearRegression
import numpy as np

app = Flask(__name__)

# Sample model
model = LinearRegression()

# Train the model with some dummy data
X = np.array([[1], [2], [3], [4], [5]])
y = np.array([1, 2, 3, 4, 5])
model.fit(X, y)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    X_new = np.array(data['input']).reshape(-1, 1)
    prediction = model.predict(X_new)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
