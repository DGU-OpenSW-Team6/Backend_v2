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
#  CORS 설정 (Netlify + 로컬)
# ========================================
origins = [
    "https://mysketchcheck.netlify.app",
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
#  기본 테스트 엔드포인트
# ========================================
@app.get("/")
def read_root():
    print("DEBUG GOOGLE_CLIENT_ID:", GOOGLE_CLIENT_ID)
    print("DEBUG GOOGLE_REDIRECT_URI:", GOOGLE_REDIRECT_URI)
    print("DEBUG GOOGLE_CLIENT_SECRET:", GOOGLE_CLIENT_SECRET)
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
        "confidence": round(confidence * 100, 2)
    }


# ========================================
#  업로드 + 내부 AI 평가 (로그인 유저 연동)
# ========================================
@app.post("/upload")
async def upload_and_evaluate(
    file: UploadFile = File(...),
    Authorization: str = Header(None)
):
    # 0) 로그인 확인
    if Authorization is None or not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = Authorization.split(" ")[1]
    user = decode_jwt(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = user["sub"]

    try:
        # ① 파일 확장자 검사
        file_ext = file.filename.split(".")[-1].lower()
        if file_ext not in ["jpg", "jpeg", "png"]:
            return JSONResponse({"error": "지원되지 않는 파일 형식입니다."}, status_code=415)

        # ② S3 업로드
        s3_key = f"uploads/{uuid4()}.{file_ext}"
        s3.upload_fileobj(file.file, BUCKET, s3_key, ExtraArgs={"ContentType": file.content_type})
        file_url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"

        # ③ AI 평가
        result = evaluate_image(file_url)

        # ④ 업로드 기록 저장 (DB 대신 파일)
        with open("uploaded_history.txt", "a", encoding="utf8") as f:
            f.write(f"{user_id},{file_url}\n")

        return {
            "user_id": user_id,
            "image_url": file_url,
            "predicted_label": result["predicted_label"],
            "confidence": result["confidence"],
            "message": "Upload + AI 접근성 평가 완료"
        }

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ========================================
#  Google 로그인
# ========================================
@app.get("/login")
def login():
    google_auth_url = (
        "https://accounts.google.com/o/oauth2/v2/auth"
        "?response_type=code"
        f"&client_id={GOOGLE_CLIENT_ID}"
        f"&redirect_uri={GOOGLE_REDIRECT_URI}"
        "&scope=openid%20email%20profile"
        "&access_type=offline"
        "&prompt=consent"
    )
    return RedirectResponse(google_auth_url)


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

    userinfo_url = "https://www.googleapis.com/oauth2/v2/userinfo"
    headers = {"Authorization": f"Bearer {access_token}"}

    async with httpx.AsyncClient() as client:
        userinfo_res = await client.get(userinfo_url, headers=headers)

    if userinfo_res.status_code != 200:
        raise HTTPException(status_code=400, detail="Failed to fetch user info")

    userinfo = userinfo_res.json()

    # 프로필 이미지도 저장
    payload = {
        "sub": userinfo["id"],
        "email": userinfo["email"],
        "name": userinfo.get("name"),
        "picture": userinfo.get("picture")
    }

    jwt_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    return {
        "google_user": userinfo,
        "jwt_token": jwt_token
    }


# ========================================
#  마이페이지 (JWT 기반)
# ========================================
@app.get("/mypage")
async def mypage(Authorization: str = Header(None)):
    if Authorization is None or not Authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing token")

    token = Authorization.split(" ")[1]
    user = decode_jwt(token)

    if user is None:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return {
        "email": user["email"],
        "name": user.get("name"),
        "profile_image": user.get("picture")
    }


# ========================================
#  내가 업로드한 이미지 목록 조회
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

    uploads = []
    if os.path.exists("uploaded_history.txt"):
        with open("uploaded_history.txt", "r", encoding="utf8") as f:
            for line in f:
                uid, url = line.strip().split(",")
                if uid == user_id:
                    uploads.append(url)

    return {"uploads": uploads}
