import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "models", "student_performance_model.pkl")

model = joblib.load(MODEL_PATH)

print("Loaded predict.py from:", __file__)


def predict_student_performance(student_data):

    prediction = model.predict(student_data)

    raw_prediction = float(prediction[0])

    print("=" * 50)
    print("Raw Prediction :", raw_prediction)

    predicted_marks = max(0, min(100, raw_prediction))

    print("After Clamp :", predicted_marks)

    print("Returning :", predicted_marks)
    print("=" * 50)

    return predicted_marks