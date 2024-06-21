import json

# 텍스트 파일에서 위도, 경도 및 음식점 이름을 읽고 HTML 템플릿 파일에 데이터를 삽입
locations_file = 'locations.txt'
template_html = 'map_template.html'
output_html = 'map.html'

# locations.txt 파일 읽기
locations = []
with open(locations_file, 'r') as file:
    for line in file:
        lat, lon, name, link = line.strip().split(' ')
        locations.append({"lat": float(lat), "lng": float(lon), "name": name, "link": link})

# 템플릿 HTML 파일 읽기
with open(template_html, 'r') as file:
    html_content = file.read()

# HTML 템플릿에 데이터 삽입
locations_data = json.dumps(locations)
html_content = html_content.replace('LOCATIONS_DATA', locations_data)

# 최종 HTML 파일 생성
with open(output_html, 'w') as file:
    file.write(html_content)

print(f"HTML file '{output_html}' has been created.")