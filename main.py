# app/main.py
import io
from typing import Optional, Tuple, List

import cv2 as cv
import numpy as np
import httpx
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="FastAPI Vision Preprocessor", version="1.0.0")

# CORS (ajustá origins según tu entorno)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------
# Utils
# --------------------------
async def fetch_image_bytes(url: str) -> bytes:
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.get(url)
            r.raise_for_status()
            return r.content
    except httpx.HTTPError as e:
        raise HTTPException(status_code=400, detail=f"No pude descargar la imagen: {e}")

def read_bgr(img_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv.imdecode(nparr, cv.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Imagen inválida o no soportada.")
    return img

def encode_jpeg(img_bgr: np.ndarray, quality: int = 92) -> bytes:
    ok, buf = cv.imencode(".jpg", img_bgr, [cv.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise HTTPException(status_code=500, detail="Error codificando JPEG.")
    return buf.tobytes()


# --------------------------
# Core CV: Deskew por Hough
# --------------------------
def deskew_by_hough(img_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (3, 3), 0)

    # Binarización robusta (inversa: líneas = blanco)
    bw = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 31, 15
    )

    # Enfatizar líneas horizontales
    h_kernel = cv.getStructuringElement(cv.MORPH_RECT, (40, 1))
    horiz = cv.morphologyEx(bw, cv.MORPH_OPEN, h_kernel, iterations=1)

    # Bordes + Hough
    edges = cv.Canny(horiz, 50, 150, apertureSize=3)
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=int(img_bgr.shape[1] * 0.4),
        maxLineGap=10,
    )

    angles: List[float] = []
    if lines is not None:
        for (x1, y1, x2, y2) in lines[:, 0, :]:
            ang = np.degrees(np.arctan2((y2 - y1), (x2 - x1)))
            if -20 <= ang <= 20:  # casi horizontales
                angles.append(ang)

    angle = float(np.median(angles)) if len(angles) > 0 else 0.0

    (h, w) = img_bgr.shape[:2]
    M = cv.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    rotated = cv.warpAffine(
        img_bgr,
        M,
        (w, h),
        flags=cv.INTER_LINEAR,
        borderMode=cv.BORDER_REPLICATE,
    )
    return rotated, angle


# --------------------------
# Opcional: warp de perspectiva
# --------------------------
def try_warp_table(img_bgr: np.ndarray) -> np.ndarray:
    """Detecta un cuadrilátero grande (probable tabla) y hace warpPerspective.
       Si no encuentra, devuelve la imagen original.
    """
    gray = cv.cvtColor(img_bgr, cv.COLOR_BGR2GRAY)
    bw = cv.adaptiveThreshold(
        gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 31, 10
    )
    bw = cv.bitwise_not(bw)

    contours, _ = cv.findContours(bw, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img_bgr

    c = max(contours, key=cv.contourArea)
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) != 4:
        return img_bgr

    pts = approx.reshape(4, 2).astype(np.float32)  
    # Ordena puntos en [tl, tr, br, bl]
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).reshape(-1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    ordered = np.array([tl, tr, br, bl], dtype=np.float32)

    # Dimensiones destino
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))
    if maxW <= 0 or maxH <= 0:
        return img_bgr

    dst = np.array(
        [[0, 0], [maxW - 1, 0], [maxW - 1, maxH - 1], [0, maxH - 1]],
        dtype=np.float32,
    )
    M = cv.getPerspectiveTransform(ordered, dst)
    warped = cv.warpPerspective(img_bgr, M, (maxW, maxH))
    return warped


# --------------------------
# Endpoints
# --------------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/deskew")
async def deskew_endpoint(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    warp: bool = Form(False),  # si True: deskew + warpPerspective
):
    """
    Devuelve la imagen procesada (JPEG) como bytes.
    - Enviar 'url' (string) o 'file' (multipart).
    - Si 'warp' es True, intenta corregir perspectiva luego del deskew.
    - Ángulo de deskew en header 'X-Deskew-Angle'.
    """
    print(f"🔍 [DEBUG] Recibido - url: {url}, file: {file}, warp: {warp}")
    
    if not url and not file:
        print("❌ [ERROR] No se recibió ni url ni file")
        raise HTTPException(status_code=400, detail="Envia 'url' o 'file'.")

    print(f"✅ [INFO] Procesando imagen desde {'URL' if url else 'archivo'}")
    
    try:
        img_bytes = await fetch_image_bytes(url) if url else await file.read()
        print(f"✅ [INFO] Imagen descargada/leída: {len(img_bytes)} bytes")
    except Exception as e:
        print(f"❌ [ERROR] Error al obtener bytes de imagen: {e}")
        raise

    try:
        img = read_bgr(img_bytes)
        print(f"✅ [INFO] Imagen decodificada: {img.shape}")
    except Exception as e:
        print(f"❌ [ERROR] Error al decodificar imagen: {e}")
        raise

    # Deskew
    try:
        rotated, angle = deskew_by_hough(img)
        print(f"✅ [INFO] Deskew aplicado: ángulo={angle:.2f}°")
    except Exception as e:
        print(f"❌ [ERROR] Error en deskew: {e}")
        raise

    # Warp opcional
    try:
        output = try_warp_table(rotated) if warp else rotated
        if warp:
            print(f"✅ [INFO] Warp de perspectiva aplicado")
    except Exception as e:
        print(f"⚠️ [WARNING] Error en warp (continuando): {e}")
        output = rotated

    try:
        jpeg = encode_jpeg(output, quality=92)
        print(f"✅ [INFO] JPEG generado: {len(jpeg)} bytes")
    except Exception as e:
        print(f"❌ [ERROR] Error al codificar JPEG: {e}")
        raise

    headers = {"X-Deskew-Angle": str(angle)}
    return Response(content=jpeg, media_type="image/jpeg", headers=headers)


@app.post("/deskew_json")
async def deskew_json_endpoint(
    url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
    warp: bool = Form(False),
):
    """
    Devuelve JSON con:
    - angle (float)
    - image_base64 (JPEG)
    Útil si preferís manejarlo todo en JSON desde Node.
    """
    import base64

    if not url and not file:
        raise HTTPException(status_code=400, detail="Envia 'url' o 'file'.")

    img_bytes = await fetch_image_bytes(url) if url else await file.read()
    img = read_bgr(img_bytes)
    rotated, angle = deskew_by_hough(img)
    output = try_warp_table(rotated) if warp else rotated

    jpeg = encode_jpeg(output, quality=92)
    b64 = base64.b64encode(jpeg).decode("utf-8")
    return {"angle": angle, "image_base64": b64}
