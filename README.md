# FastAPI Vision Preprocessor

API de procesamiento de imágenes para corrección de inclinación (deskew) y perspectiva usando FastAPI y OpenCV.

## 📋 Descripción

Este proyecto proporciona una API REST que permite procesar imágenes para:
- **Corregir la inclinación** (deskew) de documentos o tablas usando la transformada de Hough
- **Corregir la perspectiva** (warp) detectando cuadriláteros en la imagen
- Procesar imágenes desde URL o archivos subidos
- Devolver resultados en formato JPEG o JSON con base64

## 🔧 Requisitos Previos

- Python 3.8 o superior
- pip (gestor de paquetes de Python)

## 📦 Instalación

### 1. Crear y activar el entorno virtual

```bash
# Crear entorno virtual
python3 -m venv env

# Activar entorno virtual
# En macOS/Linux:
source env/bin/activate

# En Windows:
env\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Las dependencias incluyen:
- `fastapi==0.115.0` - Framework web
- `uvicorn==0.30.6` - Servidor ASGI
- `opencv-python==4.12.0.88` - Procesamiento de imágenes
- `numpy==2.1.2` - Operaciones numéricas
- `httpx==0.27.2` - Cliente HTTP asíncrono
- `python-multipart==0.0.9` - Manejo de formularios multipart

## 🚀 Levantar el Proyecto

### Modo desarrollo

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Modo producción

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

La API estará disponible en:
- **URL local**: http://localhost:8000
- **Documentación interactiva**: http://localhost:8000/docs
- **Documentación alternativa**: http://localhost:8000/redoc


## 🔍 Cómo Funciona

### Algoritmo de Deskew

1. **Preprocesamiento**: Convierte la imagen a escala de grises y aplica blur gaussiano
2. **Binarización**: Usa umbralización adaptativa para detectar bordes
3. **Detección de líneas**: Enfatiza líneas horizontales con operaciones morfológicas
4. **Transformada de Hough**: Detecta líneas y calcula sus ángulos
5. **Cálculo del ángulo**: Usa la mediana de los ángulos de líneas casi horizontales
6. **Rotación**: Aplica la corrección de rotación a la imagen original

### Corrección de Perspectiva (Warp)

1. **Detección de contornos**: Encuentra el contorno más grande en la imagen
2. **Aproximación poligonal**: Busca un cuadrilátero que represente la tabla o documento
3. **Transformación perspectiva**: Corrige la perspectiva para obtener una vista frontal

