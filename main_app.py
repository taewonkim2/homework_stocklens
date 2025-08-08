# --- 1. 필요한 라이브러리 임포트 ---
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- 1, 2번 (라이브러리 임포트, 클라이언트 초기화)는 기존과 동일 ---
import streamlit as st
import os
import io
import json
import re
import pandas as pd
from dotenv import load_dotenv
from PIL import Image
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from google.cloud import vision
import google.auth
import google.generativeai as genai

import FinanceDataReader as fdr
import plotly.express as px

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@st.cache_resource
def initialize_vision_client():
    try:
        credentials, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-vision'])
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        return vision_client
    except Exception as e:
        st.error(f"Vision 클라이언트 초기화 오류: {e}")
        return None

@st.cache_resource
def initialize_gemini_model():
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        return model
    except Exception as e:
        st.error(f"Gemini 모델 초기화 오류: {e}")
        return None

vision_client = initialize_vision_client()
gemini_model = initialize_gemini_model()


# --- 3. 핵심 기능 함수 (프롬프트 최종 강화) ---
@st.cache_data
def analyze_image_with_vision_api(_vision_client, image_bytes: bytes) -> Dict[str, List[str]]:
    # ... (기존과 동일) ...
    if not _vision_client: return {}
    image = vision.Image(content=image_bytes)
    features = [
        vision.Feature(type_=vision.Feature.Type.LOGO_DETECTION),
        vision.Feature(type_=vision.Feature.Type.WEB_DETECTION),
        vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
        vision.Feature(type_=vision.Feature.Type.TEXT_DETECTION),
    ]
    request = vision.AnnotateImageRequest(image=image, features=features)
    response = _vision_client.annotate_image(request=request)
    results = {}
    if response.logo_annotations: results['logos'] = [logo.description for logo in response.logo_annotations]
    if response.web_detection and response.web_detection.web_entities: results['web_entities'] = [entity.description for entity in response.web_detection.web_entities]
    if response.label_annotations: results['labels'] = [label.description for label in response.label_annotations]
    if response.text_annotations: results['ocr_text'] = [text.description for text in response.text_annotations]
    return results

@st.cache_data
def get_company_profile_with_gemini(_gemini_model, image_bytes: bytes, vision_results: Dict) -> Optional[Dict]:
    """Gemini를 호출하여 제품 및 제조사 프로필, 그리고 '글로벌 모회사'의 종목 코드까지 분석하여 JSON으로 반환합니다."""
    if not _gemini_model: return None
    
    # --- [수정] 프롬프트에 '모회사'를 찾으라는 지시 추가 ---
    prompt = f"""
    당신은 세계 최고의 IT 기업 및 글로벌 소비재 기업 분석 전문가입니다. 당신의 임무는 주어진 이미지와 텍스트 힌트를 종합하여, 해당 제품과 제조사에 대한 상세한 프로필을 작성하는 것입니다.

    [분석 대상 정보]
    - 이미지: (첨부됨)
    - 텍스트 힌트: {json.dumps(vision_results, ensure_ascii=False)}

    [매우 중요한 지시사항]
    - 만약 해당 브랜드가 특정 국가의 자회사(예: 한국 P&G, CJ LION)를 통해 유통되더라도, 주식 시장에 상장된 '글로벌 모회사(Parent Company)'를 기준으로 '제조사'와 '종목코드'를 찾아야 합니다.
    - '종목코드'는 FinanceDataReader 라이브러리가 지원하는 시장(KRX, NASDAQ, NYSE, AMEX, TSE, HKEX, SSE, SZSE) 중에서 찾아야 합니다.
    - 비상장 기업이거나 지원하지 않는 시장에 상장된 경우, '종목코드' 값은 "정보 없음"으로 설정해주세요.

    [최종 요청사항]
    위 모든 정보를 바탕으로, 아래 JSON 형식에 맞춰 답변을 생성해주세요. 추가적인 설명 없이 JSON 객체만 반환해야 합니다.
    ```json
    {{
      "제조사": "Lion Corporation",
      "제품명": "아이깨끗해 (Kirei Kirei) Hand Soap",
      "제조사_국가": "일본 (JPN)",
      "종목코드": "4912",
      "company_description": "라이언 주식회사(Lion Corporation)는 일본의 대표적인 생활용품 및 화학제품 제조 기업입니다. 세제, 치약, 비누 등 다양한 위생용품을 생산하며, 도쿄증권거래소에 '4912'라는 티커로 상장되어 있습니다.",
      "main_products": [
        {{"category": "생활용품", "description": "'아이깨끗해(キレイキレイ)' 손 세정제, 'CHARMY Magica' 주방 세제 등이 유명합니다."}},
        {{"category": "구강용품", "description": "'시스테마(Systema)', '크리니카(Clinica)' 등 다양한 치약 및 칫솔 브랜드를 보유하고 있습니다."}}
      ]
    }}
    ```
    """
    try:
        image_part = Image.open(io.BytesIO(image_bytes))
        response = _gemini_model.generate_content([prompt, image_part], request_options={"timeout": 120})
        match = re.search(r"```json\s*(\{.*?\})\s*```", response.text, re.DOTALL)
        if match: return json.loads(match.group(1))
        else: return json.loads(response.text)
    except Exception as e:
        st.error(f"🔮 Gemini 프로필 분석 중 오류: {e}")
        return None

@st.cache_data
def plot_stock_chart(ticker: str) -> Optional[object]:
    # ... (기존과 동일) ...
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        df_stock = fdr.DataReader(ticker, start_date, end_date)
        if df_stock.empty: return None
        fig = px.line(df_stock, y="Close", title=f"{ticker} 지난 1년간 종가(Close) 추세")
        fig.update_layout(xaxis_title="날짜", yaxis_title="가격")
        return fig
    except Exception as e:
        st.warning(f"📈 주가 정보를 불러오는 중 오류 발생: {e}")
        return None

# --- 4. Streamlit 웹 애플리케이션 UI 구성 (UI 수정) ---
st.set_page_config(page_title="AI 기업/제품 분석기", layout="centered")

if 'page' not in st.session_state: st.session_state.page = 'upload'
if 'image_source' not in st.session_state: st.session_state.image_source = None
if 'profile_info' not in st.session_state: st.session_state.profile_info = None

# --- 업로드 페이지 (기존과 동일) ---
if st.session_state.page == 'upload':
    st.title("📸 AI 기업/제품 분석기")
    st.markdown("---")
    # ... (내용 동일)
    st.subheader("알고 싶은 제품의 사진을 올려보세요")
    uploaded_file = st.file_uploader("이미지 파일 업로드", type=["jpg", "jpeg", "png"])
    camera_photo = st.camera_input("카메라로 직접 찍기")
    if uploaded_file:
        st.session_state.image_source = uploaded_file.getvalue()
        st.session_state.page = 'results'
        st.rerun()
    elif camera_photo:
        st.session_state.image_source = camera_photo.getvalue()
        st.session_state.page = 'results'
        st.rerun()

# --- 결과 페이지 (UI 수정) ---
elif st.session_state.page == 'results':
    if st.button("⬅️ 다른 이미지 분석하기"):
        st.session_state.page = 'upload'
        st.session_state.image_source = None
        st.session_state.profile_info = None
        st.rerun()

    st.title("🧠 AI 종합 분석 결과")
    st.markdown("---")
    st.image(st.session_state.image_source, use_container_width=True)
    st.markdown("---")

    if st.session_state.profile_info is None:
        with st.spinner('AI가 기업 프로필을 분석 중입니다...'):
            vision_results = analyze_image_with_vision_api(vision_client, st.session_state.image_source)
            st.session_state.profile_info = get_company_profile_with_gemini(gemini_model, st.session_state.image_source, vision_results)

    profile_info = st.session_state.profile_info
    if profile_info:
        manufacturer = profile_info.get("제조사", "정보 없음")
        product_name = profile_info.get("제품명", "정보 없음")
        country = profile_info.get("제조사_국가", "정보 없음") # <-- [수정] 국가 변수 다시 추출
        ticker = profile_info.get("종목코드", "정보 없음")
        description = profile_info.get("company_description", "")
        main_products = profile_info.get("main_products", [])
        
        # --- [수정] UI에 국가 정보 표시 복원 ---
        st.subheader("📋 기업 및 제품 정보")
        info_df = pd.DataFrame({
            "항목": ["제조사 / 브랜드", "제품명", "제조사 국가", "종목코드 (Ticker)"],
            "내용": [manufacturer, product_name, country, ticker]
        })
        st.table(info_df.set_index("항목"))

        st.markdown("---")
        st.subheader("📝 기업 소개")
        st.markdown(description)
        
        st.markdown("---")
        st.subheader("주요 생산 제품:")
        for product in main_products:
            st.markdown(f"**{product.get('category')}:** {product.get('description')}")
        
        st.markdown("---")
        st.subheader("💹 관련 주식 정보")
        if ticker and ticker != "정보 없음":
            with st.spinner(f"'{ticker}'의 주가 데이터를 불러오는 중..."):
                stock_chart_fig = plot_stock_chart(ticker)
            
            if stock_chart_fig:
                st.plotly_chart(stock_chart_fig, use_container_width=True)
            else:
                st.warning(f"'{ticker}'의 주가 차트를 불러오는 데 실패했습니다.")
        else:
            st.info("분석된 기업의 상장 정보를 찾을 수 없거나, 지원하지 않는 시장의 종목입니다.")
    else:
        st.error("이미지에서 기업 및 제품 프로필을 생성하는 데 실패했습니다.")