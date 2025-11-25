from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
import torch
from PIL import Image
import numpy as np
import boto3
import os
from dotenv import load_dotenv
from uuid import uuid4
import io
import requests
import torchvision.models as models
import httpx
from jose import jwt

from fastapi.security import HTTPBearer
security = HTTPBearer()

from yolo_detector import UIDetector
from algorithms import run_algorithms, generate_message

# ========================================
#  환경 변수
# ========================================
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH)

app = FastAPI()
print("=== FastAPI Backend Started ===")
print("GOOGLE_REDIRECT_URI =", os.getenv("GOOGLE_REDIRECT_URI"))

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

SECRET_KEY = "MY_SECRET_JWT_KEY"
ALGORITHM = "HS256"

# ========================================
#  CORS 설정 (정식 / 유일한 설정)
# ========================================
origins = [
    "https://mysketchcheck.netlify.app",
    "https://sketchcheck.shop",
    "https://www.sketchcheck.shop",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================================
#  AWS S3
# ========================================
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
BUCKET = os.getenv("S3_BUCKET_NAME")

# ========================================
#  MySQL
# ========================================
import mysql.connector

def get_db():
    print("[DB] Opening connection…")
    return mysql.connector.connect(
        host="localhost",
        user="team6",
        password="0000",
        database="sketchcheck"
    )

# ========================================
#  JWT 해독
# ========================================
def decode_jwt(token: str):
    print("[decode_jwt] token =", token)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("[decode_jwt] payload =", payload)
        return payload
    except Exception as e:
        print("[decode_jwt] ERROR =", e)
        return None

# ========================================
#  기본 API
# ========================================
@app.get("/")
def read_root():
    print("GET /")
    return {"message": "DGU OpenSW Team6 v2 Backend Running"}

@app.get("/returnScore")
def return_score():
    print("GET /returnScore")
    return {"점수": [1,2,3,4], "평가": ["a","b","c","d"]}

# ========================================
#  모델 로드
# ========================================
print("[MODEL] Loading model...")
MODEL_PATH = "ui_classifier.pt"
model = models.resnet34(num_classes=21)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()
print("[MODEL] Loaded successfully")

detector = UIDetector()

# ========================================
#  AI 분석 함수
# ========================================
def run_full_ai_pipeline(img_bytes):
    print("[AI] Running YOLO detection...")
    detections = detector.run(img_bytes)
    print("[AI] Detection done")

    print("[AI] Running rule-based algorithms...")
    analysis = run_algorithms(detections)
    print("[AI] Algorithm analysis done")

    message = generate_message(analysis)
    print("[AI] Message generated")

    return {
        "detections": detections,
        "analysis": analysis,
        "message": message
    }

# ========================================
#  업로드 + AI 통합 API (/upload)
# ========================================
@app.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    authorization: str = Header(None, alias="Authorization"),
):
    print("\n========== [POST /upload] ==========")
    print("[DEBUG] Authorization Header =", authorization)
    print("[DEBUG] File received filename =", file.filename)

    if authorization is None or not authorization.startswith("Bearer "):
        print("[/upload] error missing token")
        raise HTTPException(status_code=401, detail="Missing token")

    token = authorization.split(" ")[1]
    print("[DEBUG] Extracted token:", token)

    user = decode_jwt(token)
    if user is None:
        print("[/upload] decode failed")
        raise HTTPException(status_code=401, detail="Invalid token")

    # === 수정된 부분 (INT → 문자열) ===
    user_id = str(user["sub"])
    print("[/upload] user_id =", user_id)

    ext = file.filename.split(".")[-1].lower()
    print("[DEBUG] File extension =", ext)

    if ext not in ["jpg", "jpeg", "png"]:
        print("[ERROR] Unsupported file type")
        return JSONResponse({"error": "지원되지 않는 파일 형식입니다."}, status_code=415)

    s3_key = f"uploads/{uuid4()}.{ext}"
    print("[DEBUG] S3 upload path =", s3_key)

    # 1. S3 업로드
    try:
        s3.upload_fileobj(
            file.file, BUCKET, s3_key,
            ExtraArgs={"ContentType": file.content_type}
        )
        print("[DEBUG] S3 upload success")
    except Exception as e:
        print("[ERROR] S3 upload failed:", e)
        raise HTTPException(status_code=500, detail="S3 upload failed")

    s3_url = f"https://{BUCKET}.s3.ap-northeast-2.amazonaws.com/{s3_key}"
    print("[DEBUG] S3 URL:", s3_url)

    # 2. S3에서 다시 다운로드해서 AI 분석
    file.file.seek(0)
    img_bytes = file.file.read()
    print("[DEBUG] Running AI pipeline…")

    ai_result = run_full_ai_pipeline(img_bytes)
    print("[DEBUG] AI pipeline complete")

    # 3. DB 저장
    print("[DEBUG] Writing DB record...")
    conn = get_db()
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO uploads (user_id, s3_key, s3_url)
        VALUES (%s, %s, %s)
        """,
        (user_id, s3_key, s3_url)
    )

    conn.commit()
    cursor.close()
    conn.close()

    return {
        "file_name": file.filename,
        "s3_url": s3_url,
        "ai_result": ai_result
    }


# ========================================
#  JWT 검증 후 업로드 기록 조회 (/myuploads)
# ========================================
@app.get("/myuploads")
def get_my_uploads(authorization: str = Header(None)):
    print("[GET /myuploads]", authorization)

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = authorization.split(" ")[1]
    user = decode_jwt(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    # === 수정된 부분 ===
    user_id = str(user["sub"])

    conn = get_db()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT * FROM uploads WHERE user_id = %s ORDER BY created_at DESC",
        (user_id,)
    )
    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    return rows
