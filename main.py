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
# 완전 관대한 CORS 옵션
# ========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],        # 모든 HTTP 메서드 허용
    allow_headers=["*"],        # 모든 헤더 허용
)

# ========================================
#  모든 응답에 CORS 헤더 강제 추가
# ========================================
@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse(
            status_code=200,
            content={"message": "OK (CORS preflight allowed)"},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "*",
                "Access-Control-Allow-Methods": "*",
                "Access-Control-Allow-Credentials": "true",
            }
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Credentials"] = "true"
    return response

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

@app.post("/upload")
async def upload_and_analyze(
    file: UploadFile = File(...),
    authorization: str = Header(None, alias="Authorization"),
):
    print("\n====================== [POST /upload 시작] ======================")

    # -------------------------------
    # 0) JWT 인증
    # -------------------------------
    print("\n[1단계] JWT 인증 처리 시작")
    print("[DEBUG] Authorization 헤더 =", authorization)

    if authorization is None or not authorization.startswith("Bearer "):
        print("[ERROR] JWT 토큰 없음 → 401")
        raise HTTPException(status_code=401, detail="Missing token")

    try:
        token = authorization.split(" ")[1]
        user = decode_jwt(token)
    except Exception as e:
        print("[ERROR] JWT 파싱 실패:", e)
        raise HTTPException(status_code=401, detail="Invalid token")

    if user is None:
        print("[ERROR] JWT 해독 실패 → 401")
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = user["sub"]
    print("[SUCCESS] JWT 인증 완료 → user_id =", user_id)

    # -------------------------------
    # 1) 원본 이미지 S3 업로드
    # -------------------------------
    print("\n[2단계] 원본 이미지 S3 업로드 시작")
    print("[DEBUG] 파일명 =", file.filename)

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["jpg", "jpeg", "png"]:
        print("[ERROR] 이미지 확장자 불가:", ext)
        return JSONResponse({"error": "지원되지 않는 파일 형식입니다."}, status_code=415)

    s3_key = f"uploads/{uuid4()}.{ext}"
    print("[DEBUG] 업로드 경로 =", s3_key)

    try:
        s3.upload_fileobj(
            file.file,
            BUCKET,
            s3_key,
            ExtraArgs={"ContentType": file.content_type}
        )
        print("[SUCCESS] 원본 이미지 업로드 완료")
    except Exception as e:
        print("[ERROR] 원본 이미지 업로드 실패:", e)
        raise HTTPException(status_code=500, detail="S3 upload error")

    original_url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
    print("[DEBUG] 원본 이미지 URL =", original_url)

    # -------------------------------
    # 2) AI 분석용 이미지 다운로드
    # -------------------------------
    print("\n[3단계] 원본 이미지 다운로드 → AI 분석 준비")

    try:
        img_res = requests.get(original_url)
        img_bytes = img_res.content
        print("[SUCCESS] 다운로드 완료 (크기:", len(img_bytes), "bytes )")
    except Exception as e:
        print("[ERROR] 다운로드 실패:", e)
        raise HTTPException(status_code=500, detail="Image download error")

    # -------------------------------
    # 3) AI 분석
    # -------------------------------
    print("\n[4단계] AI 전체 파이프라인 실행 시작")

    try:
        ai_result = run_full_ai_pipeline(img_bytes)
        print("[SUCCESS] AI 분석 완료")
    except Exception as e:
        print("[ERROR] AI 분석 실패:", e)
        raise HTTPException(status_code=500, detail="AI processing failed")

    # -------------------------------
    # 3-1) 점수 계산
    # -------------------------------
    print("\n[4-1단계] 점수 계산 시작")

    try:
        from algorithms import compute_score

        analysis = ai_result["analysis"]
        detections = ai_result["detections"]

        score = compute_score(analysis, detections)
        ai_result["analysis"]["summary"]["score"] = score

        print("[SUCCESS] 점수 계산 완료 → score =", score)
    except Exception as e:
        print("[ERROR] 점수 계산 실패:", e)
        raise HTTPException(status_code=500, detail="Score calculation failed")

    # -------------------------------
    # 4) 디버그 이미지 생성
    # -------------------------------
    print("\n[5단계] 디버그 이미지 생성 시작")

    try:
        from debug_visualizer import draw_debug_image
        from io import BytesIO

        debug_buffer = BytesIO()
        detections = ai_result["detections"]
        violations = ai_result["analysis"]["violations"]

        print("[DEBUG] detections 개수 =", len(detections))
        print("[DEBUG] violations 개수 =", len(violations))

        draw_debug_image(
            img_bytes,
            detections,
            violations,
            debug_buffer
        )
        debug_buffer.seek(0)

        print("[SUCCESS] 디버그 이미지 생성 완료")
    except Exception as e:
        print("[ERROR] 디버그 이미지 생성 실패:", e)
        raise HTTPException(status_code=500, detail="Debug image generation failed")

    # -------------------------------
    # 5) 디버그 이미지 S3 업로드
    # -------------------------------
    print("\n[6단계] 디버그 이미지 S3 업로드 시작")

    debug_key = f"debug/{uuid4()}.png"

    try:
        s3.upload_fileobj(
            debug_buffer,
            BUCKET,
            debug_key,
            ExtraArgs={"ContentType": "image/png"}
        )
        print("[SUCCESS] 디버그 이미지 업로드 완료")
    except Exception as e:
        print("[ERROR] 디버그 이미지 업로드 실패:", e)
        raise HTTPException(status_code=500, detail="Debug S3 upload error")

    debug_url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{debug_key}"
    print("[DEBUG] 디버그 이미지 URL =", debug_url)

    # -------------------------------
    # 6) DB 저장
    # -------------------------------
    print("\n[7단계] DB 저장 시작")

    try:
        db = get_db()
        cursor = db.cursor()

        cursor.execute("""
            INSERT INTO uploads (
                user_id, s3_key, s3_url, filename,
                predicted_label, confidence,
                score1, score2, score3, score4,
                debug_image_url
            )
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
        """, (
            user_id,
            s3_key,
            original_url,
            file.filename,
            None, None,
            score, 0.0, 0.0, 0.0,
            debug_url
        ))

        db.commit()
        cursor.close()
        db.close()

        print("[SUCCESS] DB 저장 완료")
    except Exception as e:
        print("[ERROR] DB 저장 실패:", e)
        raise HTTPException(status_code=500, detail="Database error")

    # -------------------------------
    # 7) 프론트 응답
    # -------------------------------
    response_json = {
        "user_id": user_id,
        "image_url": original_url,
        "debug_image_url": debug_url,
        "filename": file.filename,
        "score": score,
        "ai_result": ai_result,
        "message": "Upload + AI 평가 + 디버그 이미지 생성 + DB 저장 완료"
    }

    print("\n========== [프론트로 보낼 최종 JSON] ==========")
    print(response_json)
    print("\n====================== [/upload 완료] ======================\n")

    return [response_json]


# ========================================
#  로컬 테스트 업로드 (JWT 없음)
# ========================================
@app.post("/uploadonlyfortest")
async def upload_only_for_test(file: UploadFile = File(...)):
    print("[POST /uploadonlyfortest] called")

    ext = file.filename.split(".")[-1].lower()
    if ext not in ["jpg", "jpeg", "png"]:
        return JSONResponse({"error": "지원되지 않는 파일 형식입니다."}, status_code=415)

    s3_key = f"uploads/{uuid4()}.{ext}"
    print("[/uploadonlyfortest] Upload →", s3_key)

    s3.upload_fileobj(
        file.file,
        BUCKET,
        s3_key,
        ExtraArgs={"ContentType": file.content_type},
    )

    url = f"https://{BUCKET}.s3.{os.getenv('AWS_REGION')}.amazonaws.com/{s3_key}"
    print("[/uploadonlyfortest] S3 URL:", url)

    img_res = requests.get(url)
    img_bytes = img_res.content

    ai_result = run_full_ai_pipeline(img_bytes)

    return {
        "file_name": file.filename,
        "s3_url": url,
        "ai_result": ai_result,
        "message": "Upload + S3 저장 + AI 분석 완료 (TEST MODE: no DB, no JWT)"
    }

# ========================================
#  마이페이지
# ========================================
@app.get("/mypage")
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
from datetime import datetime

from datetime import datetime
from fastapi import Header, HTTPException

@app.get("/myuploads")
def get_my_uploads(authorization: str = Header(None)):

    # -------------------------------
    # 1) JWT 인증
    # -------------------------------
    if authorization is None or not authorization.startswith("Bearer "):

        raise HTTPException(status_code=401, detail="Missing token")

    token = authorization.split(" ")[1]
    user = decode_jwt(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")

    user_id = str(user["sub"])

    # -------------------------------
    # 2) DB 조회
    # -------------------------------

    try:
        conn = get_db()
        cursor = conn.cursor(dictionary=True)

        query = """
            SELECT 
                id,
                user_id,
                s3_key,
                s3_url,
                filename,
                score1,
                score2,
                score3,
                score4,
                debug_image_url,
                created_at
            FROM uploads
            WHERE user_id = %s
            ORDER BY created_at DESC
        """

        cursor.execute(query, (user_id,))
        rows = cursor.fetchall()

        cursor.close()
        conn.close()

    except Exception as e:
        raise HTTPException(status_code=500, detail="Database error")

    # -------------------------------
    # 3) created_at → ISO 문자열 변환
    # -------------------------------

    for row in rows:
        if isinstance(row.get("created_at"), datetime):
            row["created_at"] = row["created_at"].isoformat()

    return rows



# ========================================
#  Google OAuth Callback
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

    async with httpx.AsyncClient() as client:
        token_res = await client.post(token_url, data=data)

    if token_res.status_code != 200:
        print("[auth_callback] ERROR fetching token")
        raise HTTPException(status_code=400, detail="Failed to fetch Google token")

    tokens = token_res.json()
    access_token = tokens["access_token"]

    async with httpx.AsyncClient() as client:
        userinfo_res = await client.get(
            "https://www.googleapis.com/oauth2/v2/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
        )
    userinfo = userinfo_res.json()

    payload = {
        "sub": str(userinfo["id"]),
        "email": userinfo["email"],
        "name": userinfo.get("name"),
        "picture": userinfo.get("picture"),
    }

    jwt_token = jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

    redirect_url = f"https://mysketchcheck.netlify.app/callback?token={jwt_token}"

    return RedirectResponse(url=redirect_url)
