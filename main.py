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

# ========================================
#  Security: Swagger + FastAPI 연동
# ========================================
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
security = HTTPBearer()

# ========================================
#  환경 변수 로드
# ========================================
ENV_PATH = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path=ENV_PATH)
app = FastAPI()

print("=== 서버 시작됨 ===")
print("GOOGLE_REDIRECT_URI =", os.getenv("GOOGLE_REDIRECT_URI"))

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
    print("[decode_jwt] token =", token)
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        print("[decode_jwt] payload =", payload)
        return payload
    except Exception as e:
        print("[decode_jwt] ERROR =", e)
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
    print("[MySQL] Connecting...")
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
    print("GET /")
    return {"message": "DGU OpenSW Team6 v2 Backend Running"}

# ========================================
#  테스트용 점수 반환
# ========================================
@app.get("/returnScore")
def return_score():
    print("GET /returnScore")
    return {"점수": [1, 2, 3, 4], "평가": ['a', 'b', 'c', 'd']}

# ========================================
#  모델 로드
# ========================================
MODEL_PATH = "ui_classifier.pt"
print("[MODEL] Loading model...")
model = models.resnet34(num_classes=21)
state_dict = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state_dict, strict=False)
model.eval()
print("[MODEL] Loaded successfully")

# ========================================
#  내부 함수: 이미지 평가
# ========================================
def evaluate_image(image_url: str):
    print("[evaluate_image] image_url =", image_url)
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

    print("[evaluate_image] pred =", pred_label, "conf =", confidence)

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
    print("[DB] save_user_to_db:", userinfo)

    db = get_db()
    cursor = db.cursor()

    google_id = userinfo["id"]
    email = userinfo.get("email")
    name = userinfo.get("name")
    picture = userinfo.get("picture")

    cursor.execute("SELECT id FROM users WHERE google_id=%s", (google_id,))
    result = cursor.fetchone()

    if result:
        print("[DB] Existing user:", result[0])
        user_id = result[0]
    else:
        cursor.execute(
            "INSERT INTO users (google_id, email, name, profile_url) VALUES (%s, %s, %s, %s)",
            (google_id, email, name, picture)
        )
        db.commit()
        user_id = cursor.lastrowid
        print("[DB] New user created:", user_id)

    cursor.close()
    db.close()
    return user_id

# ========================================
#  Google OAuth Callback (수정됨)
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
        "grant_type": "authorization_code"
    }

    print("[auth_callback] Fetching Google tokens...")

    async with httpx.AsyncClient() as client:
        token_res = await client.post(token_url, data=data)

    if token_res.status_code != 200:
        print("[auth_callback] ERROR: Token fetch failed")
        raise HTTPException(status_code=400, detail="Failed to fetch Google token")

    tokens = token_res.json()
    access_token = tokens["access_token"]
    print("[auth_callback] AccessToken =", access_token)

    async with httpx.AsyncClient() as client:
        userinfo_res = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"}
        )

    userinfo = userinfo_res.json()
    print("[auth_callback] userinfo =", userinfo)

    user_id = save_user_to_db(userinfo)

    payload = {
        "sub": user_id,
        "email": userinfo["email"],
        "name": userinfo.get("name"),
        "picture": userinfo.get("picture"),
    }

    jwt_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    print("[auth_callback] jwt_token =", jwt_token)

    # 🔥 React로 리다이렉트
    redirect_url = f"http://localhost:5173/callback?token={jwt_token}"
    print("[auth_callback] redirect to =", redirect_url)

    return RedirectResponse(url=redirect_url)

# ========================================
#  업로드 + 평가 + 저장
# ========================================
@app.post("/upload", dependencies=[Depends(security)])
async def upload_and_evaluate(
    file: UploadFile = File(...),
    authorization: str = Header(None, alias="Authorization")
):
    print("[POST /upload] called")
    print("[POST /upload] Authorization =", authorization)

    if authorization is None or not authorization.startswith("Bearer "):
        print("[POST /upload] ERROR: Missing Bearer")
        raise HTTPException(status_code=401, detail="Missing token")

    token = authorization.split(" ")[1]
    user = decode_jwt(token)
    if user is None:
        print("[POST /upload] ERROR: Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = user["sub"]
    print("[POST /upload] user_id =", user_id)

    try:
        file_ext = file.filename.split(".")[-1].lower()
        print("[POST /upload] Uploaded file_ext =", file_ext)

        if file_ext not in ["jpg", "jpeg", "png"]:
            print("[POST /upload] ERROR: Unsupported format")
            return JSONResponse({"error": "지원되지 않는 파일 형식입니다."}, status_code=415)

        s3_key = f"uploads/{uuid4()}.{file_ext}"
        print("[POST /upload] s3_key =", s3_key)

        s3.upload_fileobj(file.file, BUCKET, s3_key, ExtraArgs={"ContentType": file.content_type})
        file_url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"

        print("[POST /upload] file_url =", file_url)

        result = evaluate_image(file_url)

        with open("uploaded_history.txt", "a", encoding="utf8") as f:
            f.write(f"{user_id},{file_url}\n")

        save_upload_to_db(user_id, s3_key, file_url, result)

        print("[POST /upload] SUCCESS")
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
        print("[POST /upload] ERROR:", e)
        return JSONResponse({"error": str(e)}, status_code=500)

# ========================================
#  마이페이지
# ========================================
@app.get("/mypage", dependencies=[Depends(security)])
async def mypage(authorization: str = Header(None, alias="Authorization")):
    print("GET /mypage")

    if authorization is None or not authorization.startswith("Bearer "):
        print("[/mypage] ERROR: Missing Bearer")
        raise HTTPException(status_code=401, detail="Missing token")

    user = decode_jwt(authorization.split(" ")[1])
    if user is None:
        print("[/mypage] ERROR: Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")

    print("[/mypage] user =", user)

    return {
        "email": user["email"],
        "name": user.get("name"),
        "profile_image": user.get("picture")
    }

# ========================================
#  내가 올린 업로드 조회
# ========================================
@app.get("/myuploads", dependencies=[Depends(security)])
async def my_uploads(authorization: str = Header(None, alias="Authorization")):
    print("GET /myuploads")

    if authorization is None or not authorization.startswith("Bearer "):
        print("[/myuploads] ERROR: Missing Bearer")
        raise HTTPException(status_code=401, detail="Missing token")

    token = authorization.split(" ")[1]
    user = decode_jwt(token)
    if user is None:
        print("[/myuploads] ERROR: Invalid token")
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = user["sub"]
    print("[/myuploads] user_id =", user_id)

    uploads_file = []
    if os.path.exists("uploaded_history.txt"):
        with open("uploaded_history.txt", "r", encoding="utf8") as f:
            for line in f:
                uid, url = line.strip().split(",")
                if str(uid) == str(user_id):
                    uploads_file.append(url)

    uploads_db = get_uploads_from_db(user_id)

    print("[/myuploads] Count =", len(uploads_db))

    return {
        "uploads_textfile": uploads_file,
        "uploads_db": uploads_db
    }
