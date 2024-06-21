from ultralytics import YOLO
import torch
import torchvision.transforms as transforms
from PIL import Image
import shutil
import os
import subprocess

# 모델 로드
model = YOLO('best(epoch100).pt')

# 이미지 경로 리스트
image_paths = [f"{i}.jpg" for i in range(1, 101)]

# 'stair'가 검출된 이미지 경로를 저장할 리스트
stair_detection_images = []

# 아무것도 검출되지 않은 이미지 경로를 저장할 리스트
no_detection_images = []

# 각 이미지를 처리
for image_path in image_paths:
    results = model.predict(image_path)
    # results가 리스트인 경우 첫 번째 요소를 사용
    result = results[0] if isinstance(results, list) else results
    
    # 탐지된 객체 정보 확인
    detected_stair = False
    if result.boxes is not None:
        for box in result.boxes:
            cls_id = int(box.cls)
            cls_name = result.names.get(cls_id, "Unknown")
            if cls_name == 'stair':
                detected_stair = True
                break
    
    # 'stair'가 검출된 이미지인 경우
    if detected_stair:
        stair_detection_images.append(image_path)
    else:
        no_detection_images.append(image_path)

textfilenames = []


for filename in no_detection_images:
    textfilename = os.path.splitext(filename)[0] + '.txt'
    if os.path.exists(textfilename):
        textfilenames.append(textfilename)

# locations.txt 파일에 내용 합치기
with open('locations.txt', 'w') as outfile:
    for textfilename in textfilenames:
        with open(textfilename, 'r') as infile:
            contents = infile.read()
            outfile.write(contents)
            outfile.write('\n') 

print("Contents of text files have been written to 'locations.txt'")

try:
    subprocess.run(['python', 'generate_map.py'], check=True)
    print("generate_map_with_template.py has been executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while executing generate_map.py: {e}")