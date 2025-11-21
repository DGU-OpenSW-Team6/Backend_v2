from PIL import Image, ImageDraw
import io

def draw_debug_image(img_bytes, detections, spacing_violations, output_path):
    # 이미지 불러오기
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    # 색상
    base_color = (180, 180, 255)      # 연한 파란색 (기본 bbox)
    violation_color = (255, 60, 60)   # 빨간색 (위반 bbox)
    line_color = (255, 0, 0)          # 위반 연결선

    # --------------------------
    # 1) spacing 위반 대상 bbox 찾기
    # --------------------------
    violation_pairs = []
    for v in spacing_violations:
        class1, class2 = v["elements"]
        box1 = next(d for d in detections if d["class"] == class1)
        box2 = next(d for d in detections if d["class"] == class2)
        violation_pairs.append((box1, box2))

    # --------------------------
    # 2) 먼저 모든 bbox 연하게 그리기
    # --------------------------
    for det in detections:
        x, y, w, h = det["bbox"]
        draw.rectangle([x, y, x+w, y+h], outline=base_color, width=2)

    # --------------------------
    # 3) spacing 위반 난 bbox만 강조해서 빨간색으로 그리기
    # --------------------------
    for box1, box2 in violation_pairs:
        for box in [box1, box2]:
            x, y, w, h = box["bbox"]
            draw.rectangle([x, y, x+w, y+h], outline=violation_color, width=4)
            draw.text((x, y - 14), box["class"], fill=violation_color)

        # 두 박스 중심점
        c1 = (box1["bbox"][0] + box1["bbox"][2]/2,
              box1["bbox"][1] + box1["bbox"][3]/2)
        c2 = (box2["bbox"][0] + box2["bbox"][2]/2,
              box2["bbox"][1] + box2["bbox"][3]/2)

        draw.line([c1, c2], fill=line_color, width=3)

    # --------------------------
    # 저장
    # --------------------------
    img.save(output_path)
    print(f"[DEBUG] Saved annotated image → {output_path}")