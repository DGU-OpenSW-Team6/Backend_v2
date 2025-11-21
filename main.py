from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Depends
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
#  Security
# ========================================
from fastapi.security import HTTPBearer
security = HTTPBearer()

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
#  CORS
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

# ========================================
#  이미지 평가 함수
# ========================================
def evaluate_image(image_url: str):
    print("[evaluate_image] url =", image_url)

    response = requests.get(image_url)
    img = Image.open(io.BytesIO(response.content)).convert("RGB")

    img = img.resize((224, 224))
    arr = np.array(img).astype("float32") / 255.0
    arr = np.transpose(arr, (2, 0, 1))

    x = torch.tensor(arr, dtype=torch.float32)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
    x = (x - mean) / std
    x = x.unsqueeze(0)

    with torch.no_grad():
        preds = model(x)
        prob = torch.softmax(preds, dim=1)
        label = torch.argmax(prob, dim=1).item()
        conf = torch.max(prob).item()

    print("[evaluate_image] label =", label, "conf =", conf)

    return {
        "predicted_label": label,
        "confidence": round(conf * 100, 2),
        "score1": 0.0, "score2": 0.0, "score3": 0.0, "score4": 0.0
    }

# ========================================
#  DB: 유저 저장
# ========================================
def save_user_to_db(userinfo):
    print("[DB] save_user_to_db:", userinfo)

    db = get_db()
    cursor = db.cursor()

    google_id = userinfo["id"]
    email = userinfo.get("email")
    name = userinfo.get("name")
    pic = userinfo.get("picture")

    cursor.execute("SELECT id FROM users WHERE google_id=%s", (google_id,))
    exist = cursor.fetchone()

    if exist:
        print("[DB] User exists:", exist[0])
        user_id = exist[0]
    else:
        cursor.execute(
            "INSERT INTO users (google_id,email,name,profile_url) VALUES (%s,%s,%s,%s)",
            (google_id, email, name, pic)
        )
        db.commit()
        user_id = cursor.lastrowid
        print("[DB] New user created:", user_id)

    cursor.close()
    db.close()
    return user_id

# ========================================
#  Google OAuth Callback (🔥 최종 수정본)
# ========================================
@app.get("/auth/callback")
async def auth_callback(code: str):
    print("[auth_callback] code =", code)

    token_url = "https://oauth2.googleapis.com/token"
    data = {
        "code": code,
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "redirect_uri": GOOGLE_REDIRECT_URI,
        "grant_type": "authorization_code",
    }

    # 구글 액세스 토큰 요청
    async with httpx.AsyncClient() as client:
        token_res = await client.post(token_url, data=data)

    if token_res.status_code != 200:
        print("[auth_callback] ERROR fetching token")
        raise HTTPException(status_code=400, detail="Failed to fetch Google token")

    tokens = token_res.json()
    access_token = tokens["access_token"]
    print("[auth_callback] access_token =", access_token)

    # 사용자 정보 요청
    async with httpx.AsyncClient() as client:
        userinfo_res = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    userinfo = userinfo_res.json()
    print("[auth_callback] userinfo =", userinfo)

    user_id = save_user_to_db(userinfo)

    # 🔥 JWT 생성 (sub 반드시 문자열!!)
    payload = {
        "sub": str(user_id),           # FIXED: must be string!!!
        "email": userinfo["email"],
        "name": userinfo.get("name"),
        "picture": userinfo.get("picture"),
    }

    jwt_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    print("[auth_callback] jwt_token =", jwt_token)

    # React 앱으로 리다이렉트
    redirect_to = f"http://localhost:5173/callback?token={jwt_token}"
    print("[auth_callback] redirect ->", redirect_to)

    return RedirectResponse(url=redirect_to)

# ========================================
#  업로드 API
# ========================================
@app.post("/upload", dependencies=[Depends(security)])
async def upload_and_evaluate(
    file: UploadFile = File(...),
    authorization: str = Header(None, alias="Authorization"),
):
    print("[POST /upload] called")
    print("Authorization =", authorization)

    if authorization is None or not authorization.startswith("Bearer "):
        print("[/upload] error missing token")
        raise HTTPException(status_code=401, detail="Missing token")

    token = authorization.split(" ")[1]
    user = decode_jwt(token)
    if user is None:
        print("[/upload] decode failed")
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = user["sub"]
    print("[/upload] user_id =", user_id)

    # 파일 검사
    ext = file.filename.split(".")[-1].lower()
    if ext not in ["jpg", "jpeg", "png"]:
        return JSONResponse({"error": "지원되지 않는 파일 형식입니다."}, status_code=415)

    s3_key = f"uploads/{uuid4()}.{ext}"
    print("[/upload] Upload →", s3_key)

    s3.upload_fileobj(file.file, BUCKET, s3_key, ExtraArgs={"ContentType": file.content_type})

    url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
    print("[/upload] S3 URL:", url)

    result = evaluate_image(url)

    # DB 저장
    db = get_db()
    cursor = db.cursor()
    cursor.execute("""
        INSERT INTO uploads (user_id, s3_key, s3_url,
            predicted_label, confidence, score1, score2, score3, score4)
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        user_id, s3_key, url,
        result["predicted_label"], result["confidence"],
        result["score1"], result["score2"], result["score3"], result["score4"],
    ))
    db.commit()
    cursor.close()
    db.close()

    return {
        "user_id": user_id,
        "image_url": url,
        "predicted_label": result["predicted_label"],
        "confidence": result["confidence"],
        "score1": result["score1"],
        "score2": result["score2"],
        "score3": result["score3"],
        "score4": result["score4"],
        "message": "Upload + AI 평가 완료",
    }

# ========================================
#  마이페이지
# ========================================
@app.get("/mypage", dependencies=[Depends(security)])
async def mypage(authorization: str = Header(None, alias="Authorization")):
    print("GET /mypage")

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    user = decode_jwt(authorization.split(" ")[1])
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    return {
        "email": user["email"],
        "name": user["name"],
        "profile_image": user["picture"],
    }

# ========================================
#  업로드 목록 조회
# ========================================
@app.get("/myuploads", dependencies=[Depends(security)])
async def my_uploads(authorization: str = Header(None, alias="Authorization")):
    print("GET /myuploads")

    if authorization is None or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    user = decode_jwt(authorization.split(" ")[1])
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = user["sub"]

    db = get_db()
    cursor = db.cursor(dictionary=True)
    cursor.execute("""
        SELECT s3_url, predicted_label, confidence,
               score1, score2, score3, score4, created_at
        FROM uploads
        WHERE user_id=%s ORDER BY created_at DESC
    """, (user_id,))
    rows = cursor.fetchall()
    cursor.close()
    db.close()

    return {"uploads": rows}
