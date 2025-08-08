# StockLens: AI 기업/제품 분석기

이미지 한 장으로 제품 정보, 제조사 프로필, 그리고 관련 주식 정보까지 한 번에 분석해주는 AI 기반 웹 애플리케이션입니다.

## 🌟 주요 기능

- **다중 객체 탐지**: 이미지 안에 여러 제품이 있어도 각각을 인식하고 개별적으로 분석합니다.
- **AI 기업 프로파일링**: Google Vision API와 Gemini 1.5 Pro를 결합한 2단계 분석을 통해, 이미지로부터 제조사, 제품명, 기업 소개, 주요 생산품, 국가, 주식 종목 코드(Ticker)까지 추론합니다.
- **주가 시각화**: 분석된 기업이 상장사일 경우, `FinanceDataReader`와 `Plotly`를 이용해 지난 1년간의 주가 추세를 봉 차트로 시각화합니다.
- **인터랙티브 웹 UI**: `Streamlit`을 사용하여 사용자가 쉽게 이미지를 업로드하거나 카메라로 촬영할 수 있는 반응형 웹 인터페이스를 제공합니다.

## 🛠️ 기술 스택

- **Frontend**: Streamlit
- **AI / Machine Learning**:
    - **객체 탐지**: YOLOv8 (`ultralytics`)
    - **1차 정보 추출**: Google Cloud Vision API
    - **2차 종합 추론**: Google Gemini 1.5 Pro
- **Data & Visualization**: Pandas, FinanceDataReader, Plotly
- **Development Environment**: Conda, Python 3.10

## ⚙️ 로컬 환경 설정 및 실행 방법

#### 1. Conda 가상환경 생성 및 활성화
```bash
conda create -n stocklens-env python=3.10
conda activate stocklens-env
```

#### 2. 필수 라이브러리 설치
이 프로젝트는 복잡한 C++/GPU 의존성을 가진 라이브러리를 포함하므로, 아래 순서대로 설치하는 것을 강력하게 권장합니다.

a. **Conda로 핵심 AI 프레임워크 설치:**
```bash
conda install pytorch torchvision opencv -c pytorch -c conda-forge
```

b. **Pip으로 나머지 라이브러리 설치:**
```bash
pip install streamlit pandas google-cloud-vision google-generativeai python-dotenv Pillow ultralytics FinanceDataReader plotly
```
*(전체 목록은 `requirements.txt` 파일 참조)*

#### 3. `.env` 파일 설정
프로젝트 루트 폴더에 `.env` 파일을 만들고, Google AI Studio에서 발급받은 API 키를 입력합니다.
```
GEMINI_API_KEY="AIzaSy...와 같은 실제 Gemini API 키"
```

#### 4. Google Cloud 인증
로컬 터미널에서 Google Cloud에 로그인하여 ADC(Application Default Credentials)를 설정합니다.
```bash
gcloud auth application-default login
```

#### 5. 애플리케이션 실행
모든 준비가 완료되면, 아래 명령어로 Streamlit 앱을 실행합니다.
```bash
streamlit run stocklens.py
```
잠시 후 웹 브라우저에 앱이 자동으로 열립니다.