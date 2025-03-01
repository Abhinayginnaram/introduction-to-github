import tensorflow as tf
import numpy as np
import json

# Load intents and responses from a JSON file
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Preprocess user input
def preprocess_input(text):
    # Tokenize, vectorize, etc. (Simplified here)
    return [1 if word in text.lower() else 0 for word in ["headache", "fever", "cough"]]

# Define a simple TensorFlow model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(len(intents['intents']), activation='softmax')
])

# Example prediction (replace with actual trained model)
def predict_intent(user_input):
    processed_input = preprocess_input(user_input)
    prediction = model.predict(np.array([processed_input]))
    predicted_index = np.argmax(prediction[0])
    return intents['intents'][predicted_index]['tag']

# Example response generation
def generate_response(tag):
    for intent in intents['intents']:
        if intent['tag'] == tag:
            return np.random.choice(intent['responses'])

# Example usage
user_input = "I have a headache and fever"
predicted_tag = predict_intent(user_input)
response = generate_response(predicted_tag)
print(response)
