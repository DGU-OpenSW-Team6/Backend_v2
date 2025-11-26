import requests
from yolo_detector import UIDetector
from algorithms import (
    run_algorithms,
    generate_message,
    compute_score
)
from debug_visualizer import draw_debug_image

# 팀원 백엔드 주소 (나중에 실제 주소로 변경)
BACKEND_URL = "http://YOUR_BACKEND_URL/api/upload"


def analyze_and_send(image_path):
    print("[INFO] Reading image...")
    with open(image_path, "rb") as f:
        img_bytes = f.read()

    # YOLO 실행
    print("[INFO] Running YOLO inference...")
    detector = UIDetector()
    detections = detector.run(img_bytes)

    # YOLO 결과 출력
    print("\n=== YOLO Detections ===")
    for d in detections:
        print(f"- {d}")

    # 접근성 알고리즘 실행
    print("\n[INFO] Running accessibility analysis...")
    analysis = run_algorithms(detections)

    # 점수 계산
    score = compute_score(analysis, detections)
    analysis["summary"]["score"] = score  # summary 안에도 같이 넣어 줌

    print("\n=== Accessibility Score ===")
    print(f"Score: {score}")

    # 디버그 이미지 생성
    violations = analysis["violations"]
    draw_debug_image(
        img_bytes,
        detections,
        violations,
        "src/inference/debug_output.png"
    )

    # 규칙 위반 상세 출력
    print("\n=== Violations Detail ===")
    if analysis["summary"]["passed"]:
        print("문제 없음.")
    else:
        for v in analysis["violations"]:
            print(f"[{v['index']}] {v['rule']}: {v['detail']}")

    # 메시지 생성
    final_message = generate_message(analysis)
    print("\n=== Final Message ===")
    print(final_message)

    # 백엔드 전송 Payload
    payload = {
        "detections": detections,
        "analysis": analysis,
        "score": score,           # 🔥 백엔드/프론트 용 점수
        "message": final_message  # 사람이 읽을 수 있는 메시지
    }

    print("\n[INFO] Sending result to backend...")
    try:
        response = requests.post(BACKEND_URL, json=payload)
        print("[INFO] Backend response:", response.text)
    except Exception as e:
        print("[ERROR] Failed to send to backend:", e)

    return final_message, analysis, score


if __name__ == "__main__":
    # 테스트용 로컬 이미지 경로
    analyze_and_send("src/inference/testimg/122.jpg")