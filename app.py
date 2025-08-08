# --- 1. 필요한 라이브러리 임포트 ---
import streamlit as st
import os
import io
from dotenv import load_dotenv
from PIL import Image
from typing import Dict, List, Tuple, Optional

# Google Cloud 관련
from google.cloud import vision
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import google.auth
from google.auth.exceptions import DefaultCredentialsError

# --- 2. 환경 변수 및 API 클라이언트 초기화 ---
# Custom Search API 키는 .env에서 계속 사용합니다.
load_dotenv()
CUSTOM_SEARCH_API_KEY = os.getenv("CUSTOM_SEARCH_API_KEY")
CUSTOM_SEARCH_ENGINE_ID = os.getenv("CUSTOM_SEARCH_ENGINE_ID")

# Streamlit의 캐시 기능을 사용하여 API 클라이언트를 한 번만 초기화합니다.
@st.cache_resource
def initialize_vision_client():
    """Vision API 클라이언트를 초기화하고 캐시에 저장합니다."""
    try:
        credentials, project_id = google.auth.default(
            scopes=['https://www.googleapis.com/auth/cloud-vision']
        )
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        return vision_client
    except (DefaultCredentialsError, google.auth.exceptions.RefreshError) as e:
        st.error("Google Cloud 인증에 실패했습니다. ADC 설정을 확인하거나 다시 로그인해주세요.")
        st.error(f"오류 상세: {e}")
        return None
    except Exception as e:
        st.error(f"클라이언트 초기화 중 알 수 없는 오류 발생: {e}")
        return None

vision_client = initialize_vision_client()

# --- 3. 핵심 기능 함수 ---
@st.cache_data
def analyze_image_like_lens(_vision_client, image_bytes: bytes) -> Dict[str, List[str]]:
    """Vision API의 여러 기능을 호출하여 이미지 정보를 종합적으로 분석합니다."""
    if not _vision_client:
        return {}
    
    image = vision.Image(content=image_bytes)
    results = {}
    
    try:
        # 로고, 웹, 라벨 탐지를 한 번에 요청 (API 호출 최적화)
        features = [
            vision.Feature(type_=vision.Feature.Type.LOGO_DETECTION),
            vision.Feature(type_=vision.Feature.Type.WEB_DETECTION),
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
        ]
        request = vision.AnnotateImageRequest(image=image, features=features)
        response = _vision_client.annotate_image(request=request)

        # 결과 파싱
        if response.logo_annotations:
            results['logos'] = [logo.description for logo in response.logo_annotations]
        if response.web_detection and response.web_detection.web_entities:
            results['web_entities'] = [entity.description for entity in response.web_detection.web_entities]
        if response.web_detection and response.web_detection.best_guess_labels:
            results['best_guess'] = [label.label for label in response.web_detection.best_guess_labels]
        if response.label_annotations:
            results['labels'] = [label.description for label in response.label_annotations]
        
        return results
        
    except Exception as e:
        st.error(f"❌ 이미지 분석 중 오류 발생: {e}")
        return results

def search_manufacturer(query: str) -> Optional[List[Tuple[str, str, str]]]:
    """Google Custom Search API를 호출하여 검색 결과를 반환합니다."""
    try:
        service = build("customsearch", "v1", developerKey=CUSTOM_SEARCH_API_KEY)
        res = service.cse().list(q=f"{query} 제조사 또는 브랜드", cx=CUSTOM_SEARCH_ENGINE_ID, num=3).execute()
        if "items" not in res: return None
        search_results = []
        for item in res.get("items", []):
            search_results.append((
                item.get("title", "제목 없음"),
                item.get("snippet", "설명 없음"),
                item.get("link", "#")
            ))
        return search_results
    except Exception as e:
        st.error(f"❌ 검색 중 오류 발생: {e}")
        return None

# --- 4. Streamlit 웹 애플리케이션 UI 구성 ---
st.set_page_config(page_title="AI 제품 분석기", layout="wide")
st.title("📸 AI 제품 분석 및 주가 정보 조회")

# 이미지 입력 방식 선택 (파일 업로드 또는 카메라)
st.markdown("---")
input_method_tab, result_tab = st.tabs(["🖼️ 이미지 입력", "📈 분석 결과"])
image_source = None

with input_method_tab:
    st.subheader("사진을 올리거나 직접 찍어보세요")
    
    uploaded_file = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png"])
    camera_photo = st.camera_input("카메라로 직접 찍기")

    if uploaded_file:
        image_source = uploaded_file.getvalue()
    elif camera_photo:
        image_source = camera_photo.getvalue()
        
    if image_source:
        st.image(image_source, caption="분석할 이미지", width=300)
        # 사용자에게 초록색 성공 메시지 상자를 보여주며, 이제 다음 단계인 '분석 결과' 탭을 확인하라고 안내
        st.success("이미지가 준비되었습니다. '분석 결과' 탭을 확인하세요!")

# 이미지 분석 및 결과 표시 로직
with result_tab:
    if not vision_client:
        st.warning("Vision API 클라이언트가 초기화되지 않았습니다. 인증 상태를 확인해주세요.")
    elif not image_source:
        st.info("먼저 '이미지 입력' 탭에서 이미지를 준비해주세요.")
    else:
        # API가 응답하는 동안 사용자가 지루하지 않도록 '분석 중...'이라는 로딩 메시지와 빙글빙글 돌아가는 아이콘을 보여줌
        with st.spinner('이미지를 종합적으로 분석 중입니다 (로고, 웹, 라벨)...'):
            analysis_results = analyze_image_like_lens(vision_client, image_source)
        
        st.subheader("📊 이미지 분석 결과")

        if analysis_results:
            if analysis_results.get('logos'):
                st.success(f"**인식된 로고:** `{', '.join(analysis_results['logos'])}`")
            if analysis_results.get('best_guess'):
                st.info(f"**AI 최고의 추측:** `{', '.join(analysis_results['best_guess'])}`")
            if analysis_results.get('web_entities'):
                st.markdown(f"**관련 웹 키워드:** `{', '.join(analysis_results.get('web_entities', [])[:5])}` ...")
        
            st.markdown("---")
            
            # 지능적인 검색어 선택 로직
            search_term = ""
            if analysis_results.get('logos'):
                search_term = analysis_results['logos'][0]
                st.info(f"👉 1순위: 인식된 로고 **'{search_term}'**(으)로 제조사를 검색합니다.")
            elif analysis_results.get('best_guess'):
                search_term = analysis_results['best_guess'][0]
                st.info(f"👉 2순위: AI 최고의 추측 **'{search_term}'**(으)로 제조사를 검색합니다.")
            elif analysis_results.get('web_entities'):
                search_term = analysis_results['web_entities'][0]
                st.info(f"👉 3순위: 관련 웹 키워드 **'{search_term}'**(으)로 제조사를 검색합니다.")
            elif analysis_results.get('labels'):
                search_term = analysis_results['labels'][0]
                st.info(f"👉 4순위: 일반 라벨 **'{search_term}'**(으)로 제조사를 검색합니다.")
            
            # 제조사 검색 및 결과 표시
            if search_term:
                st.subheader(f"🌐 '{search_term}' 관련 웹 검색 결과")
                with st.spinner(f"'{search_term}' 관련 정보를 검색 중입니다..."):
                    search_results = search_manufacturer(search_term)
                
                if search_results:
                    for title, snippet, link in search_results:
                        st.markdown(f"**[{title}]({link})**")
                        st.caption(snippet)
                else:
                    st.warning("관련 제조사 정보를 찾는 데 실패했습니다.")
            
            # (향후 추가될 부분) 금융 데이터 라이브러리 연동
            st.markdown("---")
            st.subheader("💹 주가 정보 (구현 예정)")
            if search_term:
                st.info(f"향후 '{search_term}'의 주식 종목 코드를 찾아 시세와 차트를 여기에 표시할 예정입니다.")

        else:
            st.error("이미지에서 어떤 정보도 분석하지 못했습니다.")