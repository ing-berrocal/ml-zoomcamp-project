#Aplicacion con FastAPI para predecir expectativa de vida usando un modelo XGBoost entrenado.
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI(title="Life Expectancy Prediction API",
              description="API para predecir la expectativa de vida usando un modelo XGBoost  entrenado.",
                version="1.0.0")

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar el modelo y el DictVectorizer
MODEL_PATH = Path(__file__).parent / "model" / "model_life_expectancy_edaix.pkl"

try:
    print(MODEL_PATH)
    with open(MODEL_PATH, 'rb') as f:
        pipeline, model, columns, country, status = pickle.load(f)
        #model_data = pickle.load(f)
        #model = model_data['model']
        #dv = model_data['dv']
    print("✅ Modelo cargado exitosamente")
except Exception as e:
    print(f"❌ Error cargando modelo: {e}")
    model = None
    pipeline = None

class LifeExpectancyInput(BaseModel):
    """Modelo de entrada para la predicción de expectativa de vida"""
    adult_mortality: float = Field(..., example=263.0, description="Mortalidad adulta por 1000 habitantes")
    infant_deaths: Optional[float] = Field(62.0, example=62.0, description="Muertes infantiles por 1000 nacimientos")
    alcohol: Optional[float] = Field(0.01, example=0.01, description="Consumo de alcohol per cápita (litros)")
    percentage_expenditure: Optional[float] = Field(71.28, example=71.28, description="Gasto en salud como % del PIB")
    hepatitis_b: Optional[float] = Field(65.0, example=65.0, description="Cobertura de vacunación Hepatitis B (%)")
    bmi: Optional[float] = Field(19.1, example=19.1, description="Índice de masa corporal promedio")
    under_five_deaths: Optional[float] = Field(83.0, example=83.0, description="Muertes menores de 5 años por 1000 nacimientos")
    polio: Optional[float] = Field(6.0, example=6.0, description="Cobertura de vacunación Polio (%)")
    total_expenditure: Optional[float] = Field(8.16, example=8.16, description="Gasto total en salud (% del PIB)")
    diphtheria: Optional[float] = Field(65.0, example=65.0, description="Cobertura de vacunación Difteria (%)")
    hiv_aids: Optional[float] = Field(0.1, example=0.1, alias="hiv/aids", description="Muertes por VIH/SIDA por 1000 nacimientos")
    thinness_1_19_years: Optional[float] = Field(17.2, example=17.2, alias="thinness__1-19_years", description="Prevalencia de delgadez 10-19 años (%)")
    thinness_5_9_years: Optional[float] = Field(17.3, example=17.3, alias="thinness_5-9_years", description="Prevalencia de delgadez 5-9 años (%)")
    income_composition_of_resources: Optional[float] = Field(0.479, example=0.479, description="Índice de desarrollo de recursos humanos")
    schooling: Optional[float] = Field(10.1, example=10.1, description="Años promedio de escolaridad")
    
    class Config:
        populate_by_name = True

class LifeExpectancyOutput(BaseModel):
    """Modelo de salida para la predicción de expectativa de vida"""
    predicted_life_expectancy: float = Field(..., description="Expectativa de vida predicha en años")
    input_data: dict = Field(..., description="Datos de entrada procesados")
    model_version: str = Field("1.0.0", description="Versión del modelo")


@app.get("/")
async def root():
    """Endpoint raíz con información de la API"""
    return {
        "message": "Life Expectancy Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Realizar predicción de expectativa de vida",
            "POST /predict_batch": "Realizar predicciones en lote",
            "GET /health": "Verificar estado de la API",
            "GET /docs": "Documentación interactiva"
        }
    }


@app.post("/predict", response_model=LifeExpectancyOutput)
async def predict_life_expectancy(data: LifeExpectancyInput):
    """
    Predecir la expectativa de vida basada en los datos de entrada
    
    Recibe datos demográficos y de salud en formato JSON y devuelve
    la predicción de expectativa de vida en años.
    """
    if model is None or pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Por favor, verifica que el archivo del modelo existe."
        )
    
    try:
        # Convertir el input a diccionario
        input_dict = data.model_dump(by_alias=True)
        
        # Crear DataFrame para mantener consistencia con el entrenamiento
        df_input = pd.DataFrame([input_dict])
        
        # Transformar usando el DictVectorizer
        #X_input = dv.transform(df_input.to_dict(orient='records'))
        X_input = pd.DataFrame(pipeline.fit_transform(df_input), columns=columns)
        
        # Realizar la predicción
        prediction = model.predict(X_input)[0]
        
        # Asegurar que la predicción sea un valor válido
        prediction = float(np.clip(prediction, 0, 120))  # Limitar entre 0 y 120 años
        
        return LifeExpectancyOutput(
            predicted_life_expectancy=round(prediction, 2),
            input_data=input_dict,
            model_version="1.0.0"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción: {str(e)}"
        ) 

@app.get("/health")
async def health_check():
    """Verificar el estado de salud de la API"""
    if model is None or pipeline is None:
        raise HTTPException(status_code=503, detail="Modelo no disponible")
    
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "vectorizer_loaded": pipeline is not None
    }


@app.post("/predict_batch")
async def predict_batch(data_list: list[LifeExpectancyInput]):
    """
    Realizar predicciones en lote
    
    Recibe una lista de datos y devuelve predicciones para cada uno.
    """
    if model is None or pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible"
        )
    
    try:
        predictions = []
        
        for data in data_list:
            input_dict = data.model_dump(by_alias=True)
            df_input = pd.DataFrame([input_dict])
            X_input = pd.DataFrame(pipeline.fit_transform(df_input), columns=columns)
            prediction = model.predict(X_input)[0]
            prediction = float(np.clip(prediction, 0, 120))
            
            predictions.append({
                "predicted_life_expectancy": round(prediction, 2),
                "input_data": input_dict
            })
        
        return {
            "predictions": predictions,
            "count": len(predictions),
            "model_version": "1.0.0"
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error en la predicción batch: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
