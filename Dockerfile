# Dockerfile : 서버 컴퓨터를 자동으로 조립하고 설정하는 조립설계도
# -> 인증 후 gcloud builds submit --tag asia-northeast3-docker.pkg.dev/cogent-bolt-293201/my-vision-app/app:latest명령어를 통해 Dockerfile을 실행함

# 1. 베이스 이미지 선택 (Python 3.10 슬림 버전)
FROM python:3.10

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 시스템 라이브러리 설치 (OpenCV 구동에 필요)
# "본격적인 요리 전에, OpenCV라는 재료를 다루기 위해 **'libgl1-mesa-glx'**라는 특수 칼이 필요하니, 서랍에서 찾아서 설치해놔."
# RUN apt-get update && apt-get install -y \
#     build-essential \
#     libgl1-mesa-glx \ 
# 	libxml2-dev \
#     libxslt-dev

RUN apt-get update && apt-get install -y libgl1-mesa-glx

# 4. requirements.txt 파일을 컨테이너 안으로 복사 (/app에 저장)
COPY requirements.txt ./requirements.txt

# 5. pip을 업그레이드하고 라이브러리 설치
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 6. 나머지 앱 코드 전체를 컨테이너 안으로 복사 : 현재 폴더(.)의 모든 파일과 폴더를 /app 도마 위로 복사합니다.
COPY . .

# 7. 외부에서 접속할 포트 지정
EXPOSE 8080

# 8. 앱 실행 명령어
# 이제 모든 준비가 끝났으니, streamlit run main_app.py 명령어를 실행해서 레스토랑 영업을 시작해! 손님은 8080번 창구로 안내해
CMD ["streamlit", "run", "stocklens.py", "--server.port", "8080", "--server.enableCORS", "false"]

