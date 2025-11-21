import math

# UI 요소 중 터치 크기 검사를 해야 하는 대상
TOUCH_CLASSES = ["Button", "Switch", "Slider", "TextBox", "CheckBox", "ListPicker"]


# -------------------------------
# 기본 유틸
# -------------------------------
def compute_center(bbox):
    x, y, w, h = bbox
    return (x + w/2, y + h/2)


def calc_min_distance(boxA, boxB):
    """
    두 bbox 사이의 최소 거리 계산
    """
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB

    horiz = max(0, max(ax - (bx + bw), bx - (ax + aw)))
    vert = max(0, max(ay - (by + bh), by - (ay + ah)))

    return math.sqrt(horiz**2 + vert**2)


# -------------------------------
# 1) 요소 간 간격 검사
# -------------------------------
def spacing_test(detections, min_spacing=8):
    violations = []

    for i in range(len(detections)):
        for j in range(i+1, len(detections)):
            A = detections[i]
            B = detections[j]

            # 🔥 Screen은 간격 검사 대상에서 제외
            if A["class"] == "Screen" or B["class"] == "Screen":
                continue

            d = calc_min_distance(A["bbox"], B["bbox"])
            if d < min_spacing:
                violations.append({
                    "elements": [A["class"], B["class"]],
                    "distance": round(d, 2)
                })

    return {
        "passed": len(violations) == 0,
        "violations": violations
    }


# -------------------------------
# 2) 터치 타깃 크기 검사
# -------------------------------
def target_size_test(detections, min_size=44):
    violations = []

    for det in detections:
        if det["class"] not in TOUCH_CLASSES:
            continue

        _, _, w, h = det["bbox"]

        if w < min_size or h < min_size:
            violations.append({
                "element": det["class"],
                "bbox": det["bbox"],
                "reason": f"size too small ({round(w,1)}x{round(h,1)})"
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

            horiz_close = abs((lx + lw) - x) < 40
            vert_align = abs((ly + lh/2) - (y + h/2)) < 20

            vertical_close = abs((ly + lh) - y) < 40
            horiz_align = abs((lx + lw/2) - (x + w/2)) < 30

            if (horiz_close and vert_align) or (vertical_close and horiz_align):
                found_label = True
                break

        results.append({
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
# 4) 위반사항 통합 (index + rule + detail)
# -------------------------------
def merge_violations(spacing, target_size, label_pair):
    unified = []

    # spacing 위반
    for v in spacing["violations"]:
        unified.append({
            "rule": "spacing",
            "detail": f"{v['elements'][0]} - {v['elements'][1]} (distance: {v['distance']}px)"
        })

    # target size 위반
    for v in target_size["violations"]:
        w = round(v["bbox"][2], 1)
        h = round(v["bbox"][3], 1)
        unified.append({
            "rule": "target_size",
            "detail": f"{v['element']} too small ({w}x{h})"
        })

    # label pairing 위반
    for v in label_pair["violations"]:
        unified.append({
            "rule": "label_pair",
            "detail": f"TextBox missing label (bbox={v['textbox']})"
        })

    # 인덱스 부여
    for i, item in enumerate(unified):
        item["index"] = i

    return unified


# -------------------------------
# 최종 통합 결과 생성
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
# 최종 메시지 생성
# -------------------------------
def generate_message(analysis_result):
    summary = analysis_result["summary"]
    if summary["passed"]:
        return "모든 접근성 기준을 만족합니다. 문제 없음."

    return f"총 {summary['total_violations']}개의 접근성 문제가 발견되었습니다. 수정이 필요합니다."