from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
import joblib
import numpy as np

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Load model
model = joblib.load('iris_model.joblib')

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.post("/predict")
async def predict(
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)
    probabilities = model.predict_proba(features)[0]
    
    # Map prediction to class name
    class_names = ["Setosa", "Versicolor", "Virginica"]
    result = class_names[prediction[0]]
    
    # Convert probabilities to a dict for better frontend handling
    prob_dict = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}
    
    return JSONResponse(content={"prediction": result, "probabilities": prob_dict})
