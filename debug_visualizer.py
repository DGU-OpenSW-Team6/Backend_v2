from PIL import Image, ImageDraw, ImageFont
import io


def draw_debug_image(img_bytes, detections, violations, output_path):
    """
    detections: [{id, class, bbox}, ...]
    violations: run_algorithms()['violations']  (ids, rule, index, classes, detail 포함)
    """
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    draw = ImageDraw.Draw(img)

    base_color = (180, 180, 255)      # 기본 bbox
    violation_color = (255, 0, 0)     # bbox 선 색
    index_color = (0, 0, 0)           # 인덱스 글씨 색(검정)

    # 폰트 로드
    try:
        # Linux (AWS Ubuntu) 기본 폰트
        font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font_large = ImageFont.truetype(font_path, 50)
        font_small = ImageFont.truetype(font_path, 20)
    except Exception:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()


    # 1) 모든 bbox 기본(연한) 표시
    for det in detections:
        x, y, w, h = det["bbox"]
        draw.rectangle([x, y, x + w, y + h], outline=base_color, width=2)

    # 2) 위반 bbox + 인덱스/클래스 표시 (ID 기반)
    for v in violations:
        index = v["index"]
        ids = v["ids"]            # 이 violation에 해당하는 detection id 리스트

        for det_id in ids:
            box = next((d for d in detections if d["id"] == det_id), None)
            if box is None:
                continue

            x, y, w, h = box["bbox"]

            # 위반 bbox 강조
            draw.rectangle([x, y, x + w, y + h], outline=violation_color, width=4)

            # -------------------------
            # 인덱스 텍스트 위치/크기 계산
            # -------------------------
            index_text = f"[{index}]"

            # Pillow 10 이후: textbbox로 크기 계산
            idx_bbox = draw.textbbox((0, 0), index_text, font=font_large)
            idx_w = idx_bbox[2] - idx_bbox[0]
            idx_h = idx_bbox[3] - idx_bbox[1]

            # 기본 위치: bbox 위 가운데
            text_x = x + w / 2 - idx_w / 2
            text_y = y - idx_h - 6

            # 위로 튀어나가면 아래로 이동
            if text_y < 0:
                text_y = y + h + 4

            # 흰 배경 박스 (겹쳐도 잘 보이게)
            draw.rectangle(
                [text_x - 3, text_y - 3, text_x + idx_w + 3, text_y + idx_h + 3],
                fill=(255, 255, 255)
            )

            # 검정색 인덱스 그리기
            draw.text((text_x, text_y), index_text, fill=index_color, font=font_large)

            # -------------------------
            # 클래스명 텍스트 (인덱스 아래쪽, 빨간색)
            # -------------------------
            cls_text = box["class"]
            cls_bbox = draw.textbbox((0, 0), cls_text, font=font_small)
            cls_w = cls_bbox[2] - cls_bbox[0]
            cls_h = cls_bbox[3] - cls_bbox[1]

            cls_x = x + w / 2 - cls_w / 2
            cls_y = text_y + idx_h + 4  # 인덱스 바로 아래

            # 흰 배경 박스
            draw.rectangle(
                [cls_x - 2, cls_y - 2, cls_x + cls_w + 2, cls_y + cls_h + 2],
                fill=(255, 255, 255)
            )

            draw.text((cls_x, cls_y), cls_text, fill=violation_color, font=font_small)

        # PNG 형식으로 저장 (BytesIO일 때는 format 필수)
    img.save(output_path, format="PNG")
    print(f"[DEBUG] Saved annotated image → {output_path}")
