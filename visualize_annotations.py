import csv
import re
from PIL import Image, ImageDraw, ImageFont
import os
import textwrap

# CSV 파일 읽기 함수
def read_csv_file(filename):
    """CSV 파일을 읽어서 딕셔너리 리스트로 반환"""
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        return list(reader)

# 텍스트 정규화 함수
def normalize_text(text):
    """텍스트 정규화 (대소문자, 공백 등)"""
    if not text or text == '':
        return ''
    return re.sub(r'\s+', ' ', str(text).strip().lower())

# 텍스트를 여러 줄로 나누기
def wrap_text(text, font, max_width):
    """텍스트를 최대 너비에 맞게 여러 줄로 나누기"""
    words = text.split(' ')
    lines = []
    current_line = []
    current_width = 0
    
    for word in words:
        # 단어의 너비 측정
        try:
            word_bbox = font.getbbox(word)
            word_width = word_bbox[2] - word_bbox[0]
        except:
            word_width = len(word) * 10
        
        # 공백 추가한 너비
        if current_line:
            space_width = font.getbbox(' ')[2] - font.getbbox(' ')[0] if hasattr(font, 'getbbox') else 5
            test_width = current_width + space_width + word_width
        else:
            test_width = word_width
        
        if test_width <= max_width:
            current_line.append(word)
            current_width = test_width
        else:
            if current_line:
                lines.append(' '.join(current_line))
                current_line = [word]
                current_width = word_width
            else:
                # 단어 자체가 너무 길면 그냥 추가
                lines.append(word)
                current_line = []
                current_width = 0
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return lines if lines else [text]

# 구간 찾기 함수 (원문 기준)
def find_segments(original_text, annotation_texts):
    """원문과 annotation 텍스트들을 비교하여 하이라이트할 구간 찾기"""
    char_counts = {}
    for i in range(len(original_text)):
        char_counts[i] = 0
    
    # 각 annotation 텍스트가 원문의 어느 부분과 매칭되는지 찾기
    for ann_text in annotation_texts:
        if not ann_text:
            continue
        
        ann_lower = ann_text.lower()
        original_lower = original_text.lower()
        
        start_idx = 0
        while True:
            idx = original_lower.find(ann_lower, start_idx)
            if idx == -1:
                break
            for i in range(idx, idx + len(ann_text)):
                if i < len(original_text):
                    char_counts[i] += 1
            start_idx = idx + 1
    
    # 연속된 같은 count를 가진 구간 찾기
    segments = []
    if len(original_text) > 0:
        current_start = 0
        current_count = char_counts[0]
        
        for i in range(1, len(original_text)):
            if char_counts[i] != current_count:
                if current_count > 0:
                    segments.append((current_start, i-1, current_count))
                current_start = i
                current_count = char_counts[i]
        
        if current_count > 0:
            segments.append((current_start, len(original_text)-1, current_count))
    
    return segments

# 여러 줄 텍스트에 하이라이트 그리기
def draw_multiline_highlights(img, draw, original_text, wrapped_lines, segments, total_annotations, color, padding, font, line_height, start_y):
    """여러 줄 텍스트에 하이라이트 그리기"""
    current_pos = 0
    y = start_y
    
    for line in wrapped_lines:
        line_start_pos = current_pos
        line_end_pos = current_pos + len(line)
        
        # 이 줄에 해당하는 segments 찾기
        for seg_start, seg_end, count in segments:
            if count > 0:
                # 구간이 이 줄과 겹치는지 확인
                if seg_start < line_end_pos and seg_end >= line_start_pos:
                    # 구간의 이 줄에서의 실제 위치 계산
                    seg_in_line_start = max(0, seg_start - line_start_pos)
                    seg_in_line_end = min(len(line), seg_end - line_start_pos + 1)
                    
                    if seg_in_line_start < seg_in_line_end:
                        segment_text = line[seg_in_line_start:seg_in_line_end]
                        # 구두점이나 공백만 있는 구간은 제외
                        if re.search(r'[a-zA-Z0-9]', segment_text):
                            opacity = int((count / total_annotations) * 255) if total_annotations > 0 else 0
                            
                            # 이전 텍스트의 너비 계산
                            prefix_text = line[:seg_in_line_start]
                            try:
                                prefix_bbox = draw.textbbox((0, 0), prefix_text, font=font)
                                rect_x = padding + (prefix_bbox[2] - prefix_bbox[0])
                            except:
                                try:
                                    prefix_bbox = font.getbbox(prefix_text)
                                    rect_x = padding + (prefix_bbox[2] - prefix_bbox[0])
                                except:
                                    rect_x = padding + len(prefix_text) * 10
                            
                            # 세그먼트 너비 계산
                            try:
                                segment_bbox = draw.textbbox((0, 0), segment_text, font=font)
                                rect_width = segment_bbox[2] - segment_bbox[0]
                            except:
                                try:
                                    segment_bbox = font.getbbox(segment_text)
                                    rect_width = segment_bbox[2] - segment_bbox[0]
                                except:
                                    rect_width = len(segment_text) * 10
                            
                            # 반투명 레이어 생성
                            overlay = Image.new('RGBA', (img.width, img.height), (0, 0, 0, 0))
                            overlay_draw = ImageDraw.Draw(overlay)
                            overlay_draw.rectangle(
                                [rect_x, y, rect_x + rect_width, y + line_height],
                                fill=(*color, opacity)
                            )
                            img.paste(Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB'), (0, 0))
        
        current_pos = line_end_pos
        # 줄바꿈 문자 처리
        if current_pos < len(original_text) and original_text[current_pos] == '\n':
            current_pos += 1
        y += line_height

# 하나의 sample_key와 method에 대한 이미지 생성
def create_image(sample_key, method, original_text, filtered_annotations, output_dir):
    """하나의 sample_key와 method에 대해 good과 bad를 함께 표시한 이미지 생성"""
    
    # 해당하는 annotation 데이터 필터링
    filtered_rows = []
    for row in filtered_annotations:
        if str(row.get('sample_key', '')) == str(sample_key) and row.get('method', '').lower() == method.lower():
            filtered_rows.append(row)
    
    # Good과 Bad 데이터 분리
    good_rows = []
    bad_rows = []
    for row in filtered_rows:
        good = row.get('Good', '').strip() if row.get('Good') else ''
        bad = row.get('Bad', '').strip() if row.get('Bad') else ''
        if good:
            good_rows.append(row)
        if bad:
            bad_rows.append(row)
    
    # 폰트 설정
    font_size = 28
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except:
        try:
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    # 최대 이미지 너비 설정 (줄바꿈 기준)
    max_img_width = 1200
    padding = 40
    line_spacing = 10
    
    # 텍스트를 여러 줄로 나누기
    try:
        # 대략적인 문자 너비 계산
        test_char_width = font.getbbox('M')[2] - font.getbbox('M')[0] if hasattr(font, 'getbbox') else 15
        chars_per_line = int((max_img_width - padding * 2) / test_char_width)
        wrapped_lines = textwrap.wrap(original_text, width=chars_per_line)
    except:
        wrapped_lines = wrap_text(original_text, font, max_img_width - padding * 2)
    
    # 각 줄의 높이 계산
    try:
        bbox = font.getbbox('Ag')
        line_height = bbox[3] - bbox[1] + line_spacing
    except:
        try:
            bbox = font.getbbox('Ag')
            line_height = bbox[3] - bbox[1] + line_spacing
        except:
            line_height = 35
    
    # 이미지 크기 계산
    num_lines = len(wrapped_lines)
    section_height = (line_height * num_lines) + padding * 2
    label_height = 30
    
    # Good과 Bad 섹션 높이 계산
    total_height = label_height * 2  # 라벨 공간
    if good_rows:
        total_height += section_height + line_spacing
    if bad_rows:
        total_height += section_height + line_spacing
    
    img_width = max_img_width
    img_height = total_height
    
    # 이미지 생성
    img = Image.new('RGB', (img_width, img_height), color='white')
    draw = ImageDraw.Draw(img)
    
    current_y = 0
    
    # Good 하이라이트
    if good_rows:
        good_texts = [normalize_text(row.get('Good', '')) for row in good_rows]
        good_segments = find_segments(original_text, good_texts)
        
        # 라벨
        label_font_size = 18
        try:
            label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", label_font_size)
        except:
            try:
                label_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", label_font_size)
            except:
                label_font = ImageFont.load_default()
        
        draw.text((padding, current_y), f"Good ({len(good_rows)} annotations)", fill='green', font=label_font)
        current_y += label_height
        
        # 하이라이트 그리기
        draw_multiline_highlights(img, draw, original_text, wrapped_lines, good_segments, len(good_rows), (0, 255, 0), padding, font, line_height, current_y)
        
        # 텍스트 그리기
        y = current_y
        for line in wrapped_lines:
            draw.text((padding, y), line, fill='black', font=font)
            y += line_height
        
        current_y += section_height + line_spacing
    
    # Bad 하이라이트
    if bad_rows:
        bad_texts = [normalize_text(row.get('Bad', '')) for row in bad_rows]
        bad_segments = find_segments(original_text, bad_texts)
        
        # 라벨
        label_font_size = 18
        try:
            label_font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", label_font_size)
        except:
            try:
                label_font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", label_font_size)
            except:
                label_font = ImageFont.load_default()
        
        draw.text((padding, current_y), f"Bad ({len(bad_rows)} annotations)", fill='red', font=label_font)
        current_y += label_height
        
        # 하이라이트 그리기
        draw_multiline_highlights(img, draw, original_text, wrapped_lines, bad_segments, len(bad_rows), (255, 0, 0), padding, font, line_height, current_y)
        
        # 텍스트 그리기
        y = current_y
        for line in wrapped_lines:
            draw.text((padding, y), line, fill='black', font=font)
            y += line_height
    
    # 이미지 저장
    os.makedirs(output_dir, exist_ok=True)
    output_filename = os.path.join(output_dir, f'output_{sample_key}_{method}.png')
    img.save(output_filename)
    print(f"Saved: {output_filename}")

# 메인 실행
def main():
    # CSV 파일 읽기
    annotation_data = read_csv_file('human-study-data/results/annotation_data.csv')
    model_explanations_data = read_csv_file('human-study-data/results/model_explanations.csv')
    
    # Good과 Bad가 모두 비어있는 행 제거
    filtered_annotations = []
    for row in annotation_data:
        good = row.get('Good', '').strip() if row.get('Good') else ''
        bad = row.get('Bad', '').strip() if row.get('Bad') else ''
        if good or bad:
            filtered_annotations.append(row)
    
    # 모든 sample_key와 method 조합 찾기
    combinations = []
    for row in model_explanations_data:
        sample_key = str(row.get('Sample key', ''))
        method = row.get('Method', '').strip()
        if sample_key and method:
            combinations.append((sample_key, method))
    
    # 중복 제거
    combinations = list(set(combinations))
    combinations.sort()
    
    print(f"Found {len(combinations)} combinations to process")
    
    # 각 조합에 대해 이미지 생성
    output_dir = 'human-study-data/results'
    for sample_key, method in combinations:
        # 원문 설명 가져오기
        original_text = None
        for row in model_explanations_data:
            if str(row.get('Sample key', '')) == sample_key and row.get('Method', '').strip().lower() == method.lower():
                original_text = row.get('Explanation', '').strip()
                break
        
        if not original_text:
            print(f"Warning: No explanation found for sample_key={sample_key}, method={method}")
            continue
        
        print(f"\nProcessing: sample_key={sample_key}, method={method}")
        create_image(sample_key, method, original_text, filtered_annotations, output_dir)
    
    print(f"\nAll images generated in {output_dir}")

if __name__ == '__main__':
    main()
