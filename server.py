from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load dataset & train model (you can move training to a separate script)
data = pd.read_csv("college_student_placement_dataset.csv")

# Preprocessing
label_enc = LabelEncoder()
data["Company"] = label_enc.fit_transform(data["Company"])

X = data.drop("Placed", axis=1)  # Features
y = data["Placed"]              # Target

model = RandomForestClassifier()
model.fit(X, y)

# Save model
joblib.dump(model, "placement_model.pkl")

# Load model
model = joblib.load("placement_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    content = request.json
    company = content["company"]
    skills = content["skills"]

    # Example features (you should match dataset columns properly)
    sample = pd.DataFrame([{
        "CGPA": content.get("cgpa", 7),
        "Skills": len(skills),  # Example: number of skills
        "Company": label_enc.transform([company])[0] if company in label_enc.classes_ else 0
    }])

    prediction = model.predict(sample)[0]

    if prediction == 1:
        return jsonify({"result": f"✅ Likely to get placed in {company}"})
    else:
        return jsonify({"result": f"❌ Improve your skills for {company}"})

if __name__ == "__main__":
    app.run(debug=True)
