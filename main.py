from fastapi import FastAPI, File, UploadFile, HTTPException, Header
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

# ========================================
#  환경 변수 로드
# ========================================
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH)
app = FastAPI()

# ========================================
#  Google OAuth 관련 환경변수
# ========================================
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv("GOOGLE_REDIRECT_URI")

SECRET_KEY = "MY_SECRET_JWT_KEY"
ALGORITHM = "HS256"

# ========================================
#  JWT 해독 함수
# ========================================
def decode_jwt(token: str):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception:
        return None

# ========================================
#  CORS 설정
# ========================================
origins = [
    "https://mysketchcheck.netlify.app",
    "http://localhost:5173",
    "https://sketchcheck.shop",
    "https://www.sketchcheck.shop",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*", "Authorization"],
)


# ========================================
#  AWS S3 설정
# ========================================
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
)
BUCKET = os.getenv("S3_BUCKET_NAME")

# ========================================
#  MySQL 연결
# ========================================
import mysql.connector

def get_db():
    return mysql.connector.connect(
        host="localhost",
        user="team6",
        password="0000",
        database="sketchcheck"
    )

# ========================================
#  기본 테스트 엔드포인트
# ========================================
@app.get("/")
def read_root():
    return {"message": "DGU OpenSW Team6 v2 Backend Running"}

# ========================================
#  테스트용 점수 반환
# ========================================
@app.get("/returnScore")
def return_score():
    return {"점수": [1, 2, 3, 4], "평가": ['a', 'b', 'c', 'd']}

# ========================================
#  모델 로드
# ========================================
MODEL_PATH = "ui_classifier.pt"
model = models.resnet34(num_classes=21)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()

# ========================================
#  내부 함수: 이미지 평가
# ========================================
def evaluate_image(image_url: str):
    response = requests.get(image_url)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")

    img_size = 224
    IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    image = image.resize((img_size, img_size))
    arr = np.array(image).astype('float32') / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.tensor(arr, dtype=torch.float32)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = x.unsqueeze(0)

    with torch.no_grad():
        preds = model(x)
        prob = torch.softmax(preds, dim=1)
        pred_label = torch.argmax(prob, dim=1).item()
        confidence = torch.max(prob).item()

    return {
        "predicted_label": int(pred_label),
        "confidence": round(confidence * 100, 2),
        "score1": 0.0,
        "score2": 0.0,
        "score3": 0.0,
        "score4": 0.0,
    }

# ========================================
#  DB: user 저장
# ========================================
def save_user_to_db(userinfo):
    db = get_db()
    cursor = db.cursor()

    google_id = userinfo["id"]
    email = userinfo.get("email")
    name = userinfo.get("name")
    picture = userinfo.get("picture")

    cursor.execute("SELECT id FROM users WHERE google_id=%s", (google_id,))
    result = cursor.fetchone()

    if result:
        user_id = result[0]
    else:
        cursor.execute(
            "INSERT INTO users (google_id, email, name, profile_url) VALUES (%s, %s, %s, %s)",
            (google_id, email, name, picture)
        )
        db.commit()
        user_id = cursor.lastrowid

    cursor.close()
    db.close()
    return user_id

# ========================================
#  DB: 업로드 기록 저장 (★ score1~4 포함)
# ========================================
def save_upload_to_db(user_id, s3_key, file_url, result):
    db = get_db()
    cursor = db.cursor()

    cursor.execute("""
        INSERT INTO uploads
        (user_id, s3_key, s3_url,
         predicted_label, confidence,
         score1, score2, score3, score4)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        user_id, s3_key, file_url,
        result["predicted_label"], result["confidence"],
        result["score1"], result["score2"], result["score3"], result["score4"]
    ))

    db.commit()
    cursor.close()
    db.close()

# ========================================
#  Google OAuth Callback
# ========================================
@app.get("/auth/callback")
async def auth_callback(code: str):
    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code"
    }

    async with httpx.AsyncClient() as client:
        token_res = await client.post(token_url, data=data)

    if token_res.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch Google token")

    tokens = token_res.json()
    access_token = tokens["access_token"]

    async with httpx.AsyncClient() as client:
        userinfo_res = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )

    userinfo = userinfo_res.json()

    user_id = save_user_to_db(userinfo)

    payload = {
        "sub": user_id,
        "email": userinfo["email"],
        "name": userinfo.get("name"),
        "picture": userinfo.get("picture"),
    }

    jwt_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "google_user": userinfo,
        "jwt_token": jwt_token
    }

# ========================================
#  업로드 + 평가 + 저장
# ========================================
@app.post("/upload")
async def upload_and_evaluate(
    file: UploadFile = File(...),
    Authorization: str = Header(None)
):
    if Authorization is None or not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = Authorization.split(" ")[1]
    user = decode_jwt(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = user["sub"]

    try:
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in ["jpg", "jpeg", "png"]:
            return JSONResponse({"error": "지원되지 않는 파일 형식입니다."}, status_code=415)

        s3_key = f"uploads/{uuid4()}.{file_ext}"
        s3.upload_fileobj(file.file, BUCKET, s3_key, ExtraArgs={"ContentType": file.content_type})
        file_url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"

        result = evaluate_image(file_url)

        with open("uploaded_history.txt", "a", encoding="utf8") as f:
            f.write(f"{user_id},{file_url}\n")

        save_upload_to_db(user_id, s3_key, file_url, result)

        return {
            "user_id": user_id,
            "image_url": file_url,
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "score1": result["score1"],
            "score2": result["score2"],
            "score3": result["score3"],
            "score4": result["score4"],
            "message": "Upload + AI 접근성 평가 완료"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ========================================
#  마이페이지
# ========================================
@app.get("/mypage")
async def mypage(Authorization: str = Header(None)):
    if Authorization is None or not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    user = decode_jwt(Authorization.split(" ")[1])
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {
        "email": user["email"],
        "name": user.get("name"),
        "profile_image": user.get("picture")
    }

# ========================================
#  DB 기반 업로드 목록 조회
# ========================================
def get_uploads_from_db(user_id):
    db = get_db()
    cursor = db.cursor(dictionary=True)

    cursor.execute("""
        SELECT
            s3_url,
            predicted_label,
            confidence,
            score1, score2, score3, score4,
            created_at
        FROM uploads
        WHERE user_id=%s
        ORDER BY created_at DESC
    """, (user_id,))

    result = cursor.fetchall()

    cursor.close()
    db.close()
    return result

# ========================================
#  내가 올린 업로드 조회
# ========================================
@app.get("/myuploads")
async def my_uploads(Authorization: str = Header(None)):
    if Authorization is None or not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = Authorization.split(" ")[1]
    user = decode_jwt(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = user["sub"]

    uploads_file = []
    if os.path.exists("uploaded_history.txt"):
        with open("uploaded_history.txt", "r", encoding="utf8") as f:
            for line in f:
                uid, url = line.strip().split(",")
                if str(uid) == str(user_id):
                    uploads_file.append(url)

    uploads_db = get_uploads_from_db(user_id)

    return {
        "uploads_textfile": uploads_file,
        "uploads_db": uploads_db
    }
