FROM python:3.11-slim

WORKDIR /app

# Copiar archivos de dependencias
COPY requirements.txt .

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar código de la aplicación
COPY model/* /app/model/

COPY ml-app.py .


# Copiar el modelo (ajusta la ruta según tu estructura)
#COPY ../model/model_life_expectancy_edaviii.pkl /app/model/

# Exponer el puerto
EXPOSE 8000

# Comando para ejecutar la aplicación
CMD ["uvicorn", "ml-app:app", "--host", "0.0.0.0", "--port", "8000"]
