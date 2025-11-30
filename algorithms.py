import math

# UI 요소 중 터치 크기 검사를 해야 하는 대상
TOUCH_CLASSES = ["Button", "Switch", "Slider", "TextBox", "CheckBox", "ListPicker"]


# -------------------------------
# 기본 유틸
# -------------------------------
def compute_center(bbox):
    x, y, w, h = bbox
    return (x + w / 2, y + h / 2)


def calc_min_distance(boxA, boxB):
    """
    두 bbox 사이의 최소 거리 계산
    box: [x, y, w, h]  (top-left + width/height)
    """
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB

    # 두 사각형이 떨어져 있을 때의 x, y 방향 최소 거리
    horiz = max(0, max(ax - (bx + bw), bx - (ax + aw)))
    vert = max(0, max(ay - (by + bh), by - (ay + ah)))

    return math.sqrt(horiz ** 2 + vert ** 2)


# -------------------------------
# 1) 요소 간 간격 검사
# -------------------------------
def spacing_test(detections, min_spacing=8):
    violations = []

    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            A = detections[i]
            B = detections[j]

            # Screen은 간격 검사 대상에서 제외 (원하면 제거 가능)
            if A["class"] == "Screen" or B["class"] == "Screen":
                continue

            d = calc_min_distance(A["bbox"], B["bbox"])
            if d < min_spacing:
                violations.append({
                    "id1": A["id"],                 # 🔥 두 요소의 ID 저장
                    "id2": B["id"],
                    "classes": [A["class"], B["class"]],
                    "distance": round(d, 2),
                })

    return {
        "passed": len(violations) == 0,
        "violations": violations
    }


# -------------------------------
# 2) 터치 타깃 크기 검사 (너무 작음 / 너무 큼)
# -------------------------------
def target_size_test(detections, min_size=44, max_size=200):
    violations = []

    for det in detections:
        cls = det["class"]

        # TextBox는 크기 제한에서 제외
        if cls == "TextBox":
            continue

        if cls not in TOUCH_CLASSES:
            continue

        _, _, w, h = det["bbox"]

        if w < min_size or h < min_size:
            violations.append({
                "id": det["id"],
                "element": cls,
                "bbox": det["bbox"],
                "reason": "too_small",
                "detail": f"{cls} too small ({round(w, 1)}x{round(h, 1)})"
            })

    return {
        "passed": len(violations) == 0,
        "violations": violations
    }


# -------------------------------
# 3) 라벨 - 입력창(TextBox) 연관성 검사
# -------------------------------
def label_pairing_test(detections):
    results = []

    textboxes = [d for d in detections if d["class"] == "TextBox"]
    labels = [d for d in detections if d["class"] == "Label"]

    for tb in textboxes:
        x, y, w, h = tb["bbox"]
        found_label = False

        for lb in labels:
            lx, ly, lw, lh = lb["bbox"]

            # 좌우/상하 방향 근접성 + 정렬 체크
            horiz_close = abs((lx + lw) - x) < 40
            vert_align = abs((ly + lh / 2) - (y + h / 2)) < 20

            vertical_close = abs((ly + lh) - y) < 40
            horiz_align = abs((lx + lw / 2) - (x + w / 2)) < 30

            if (horiz_close and vert_align) or (vertical_close and horiz_align):
                found_label = True
                break

        results.append({
            "id": tb["id"],            # 🔥 TextBox ID 저장
            "textbox": tb["bbox"],
            "label_found": found_label
        })

    # label_missing 만 추려냄
    missing = [r for r in results if not r["label_found"]]

    return {
        "passed": len(missing) == 0,
        "details": results,
        "violations": missing
    }


# -------------------------------
# 4) 위반사항 통합 (index + rule + detail + ids)
# -------------------------------
def merge_violations(spacing, target_size, label_pair):
    unified = []

    # spacing 위반
    for v in spacing["violations"]:
        clsA, clsB = v["classes"]
        unified.append({
            "rule": "spacing",
            "ids": [v["id1"], v["id2"]],          # 🔥 두 요소 ID
            "classes": [clsA, clsB],
            "detail": f"{clsA} - {clsB}"
        })

    # target size 위반
    for v in target_size["violations"]:
        cls = v["element"]
        unified.append({
            "rule": "target_size",
            "ids": [v["id"]],                     # 🔥 한 요소 ID
            "classes": [cls],
            "detail": v["detail"]
        })

    # label pairing 위반
    for v in label_pair["violations"]:
        unified.append({
            "rule": "label_pair",
            "ids": [v["id"]],                     # 🔥 TextBox ID
            "classes": ["TextBox"],
            "detail": "TextBox missing label"
        })

    # 인덱스 부여
    for i, item in enumerate(unified):
        item["index"] = i

    return unified


# -------------------------------
# 5) 최종 통합 결과 생성
# -------------------------------
def run_algorithms(detections):
    spacing = spacing_test(detections)
    target_size = target_size_test(detections)
    label_pair = label_pairing_test(detections)

    # 통합된 위반 리스트 생성
    violations = merge_violations(spacing, target_size, label_pair)

    summary = {
        "passed": len(violations) == 0,
        "total_violations": len(violations)
    }

    return {
        "summary": summary,
        "violations": violations,
        "spacing_result": spacing,
        "target_size_result": target_size,
        "label_pairing_result": label_pair
    }


# -------------------------------
# 6) 점수 계산  (100 * (1 - 위반 개수 / 총 비교 개수))
# -------------------------------
def compute_score(analysis, detections):
    """
    analysis: run_algorithms() 결과
    detections: YOLO 결과
    """

    # 1) spacing 비교 횟수 (Screen 제외한 쌍)
    spacing_pairs = 0
    for i in range(len(detections)):
        for j in range(i + 1, len(detections)):
            A = detections[i]
            B = detections[j]
            if A["class"] == "Screen" or B["class"] == "Screen":
                continue
            spacing_pairs += 1

    # 2) target_size 비교 횟수 (TextBox 제외, 터치 대상만)
    target_size_comparisons = len([
        d for d in detections
        if d["class"] in TOUCH_CLASSES and d["class"] != "TextBox"
    ])

    # 3) label pairing 비교 횟수 (TextBox 개수)
    label_pairing_comparisons = len([
        d for d in detections
        if d["class"] == "TextBox"
    ])

    total_comparisons = (
        spacing_pairs
        + target_size_comparisons
        + label_pairing_comparisons
    )

    total_violations = analysis["summary"]["total_violations"]

    if total_comparisons == 0:
        return 100.0  # 비교 대상이 없으면 100점

    score = 100 * (1 - (total_violations / total_comparisons))

    # 점수는 0~100 범위로 clip + 소수 둘째자리까지
    score = max(0.0, min(100.0, round(score, 2)))

    return score


# -------------------------------
# 최종 메시지 생성
# -------------------------------
def generate_message(analysis_result):
    summary = analysis_result["summary"]
    if summary["passed"]:
        return "모든 접근성 기준을 만족합니다."

    return f"총 {summary['total_violations']}개의 접근성 문제가 발견되었습니다. 수정이 필요합니다."