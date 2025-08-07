# --- 1. 필요한 라이브러리 임포트 ---
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

# --- Google Cloud 및 Generative AI 관련 ---
from google.cloud import vision
import google.auth
import google.generativeai as genai

# --- 금융 데이터 및 시각화 관련 ---
import FinanceDataReader as fdr
import plotly.graph_objects as go # <-- [수정] 더 상세한 차트를 위해 graph_objects 임포트

# --- 객체 탐지(YOLO) 관련 ---
from ultralytics import YOLO
import cv2
import numpy as np

# --- 2. 환경 변수 및 API/모델 클라이언트 초기화 (기존과 동일) ---
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@st.cache_resource
def initialize_vision_client():
    # ... (기존과 동일)
    try:
        credentials, project_id = google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-vision'])
        vision_client = vision.ImageAnnotatorClient(credentials=credentials)
        return vision_client
    except Exception as e:
        st.error(f"Vision 클라이언트 초기화 오류: {e}")
        return None

@st.cache_resource
def initialize_gemini_model():
    # ... (기존과 동일)
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-1.5-pro-latest')
        return model
    except Exception as e:
        st.error(f"Gemini 모델 초기화 오류: {e}")
        return None

@st.cache_resource
def load_yolo_model():
    # ... (기존과 동일)
    model = YOLO("yolov8n.pt")
    return model

vision_client = initialize_vision_client()
gemini_model = initialize_gemini_model()
yolo_model = load_yolo_model()

# --- 3. 핵심 기능 함수 (차트 함수 수정) ---
def detect_objects_with_yolo(image_bytes: bytes) -> List[bytes]:
    # ... (기존과 동일)
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = yolo_model(img_rgb)
        cropped_images = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_img = img_rgb[y1:y2, x1:x2]
                pil_img = Image.fromarray(cropped_img)
                buf = io.BytesIO()
                pil_img.save(buf, format='PNG')
                cropped_images.append(buf.getvalue())
        if not cropped_images: return [image_bytes]
        return cropped_images
    except Exception as e:
        st.error(f"📦 YOLO 객체 탐지 중 오류: {e}")
        return [image_bytes]

@st.cache_data
def analyze_image_with_vision_api(_vision_client, image_bytes: bytes) -> Dict[str, List[str]]:
    # ... (기존과 동일)
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
    # ... (기존과 동일)
    if not _gemini_model: return None
    prompt = f"""
    당신은 세계 최고의 IT 기업 및 글로벌 소비재 기업 분석 전문가입니다. 당신의 임무는 주어진 이미지와 텍스트 힌트를 종합하여, 해당 제품과 제조사에 대한 상세한 프로필을 작성하는 것입니다.
    [분석 대상 정보] - 이미지: (첨부됨), - 텍스트 힌트: {json.dumps(vision_results, ensure_ascii=False)}
    [매우 중요한 지시사항] - 만약 해당 브랜드가 특정 국가의 자회사(예: 한국 P&G, CJ LION)를 통해 유통되더라도, 주식 시장에 상장된 '글로벌 모회사(Parent Company)'를 기준으로 '제조사'와 '종목코드'를 찾아야 합니다. - '종목코드'는 FinanceDataReader 라이브러리가 지원하는 시장(KRX, NASDAQ, NYSE, AMEX, TSE, HKEX, SSE, SZSE) 중에서 찾아야 합니다. - 비상장 기업이거나 지원하지 않는 시장에 상장된 경우, '종목코드' 값은 "정보 없음"으로 설정해주세요.
    [최종 요청사항] 위 모든 정보를 바탕으로, 아래 JSON 형식에 맞춰 답변을 생성해주세요. 추가적인 설명 없이 JSON 객체만 반환해야 합니다.
    ```json
    {{
      "제조사": "Lion Corporation", "제품명": "아이깨끗해 (Kirei Kirei) Hand Soap", "제조사_국가": "일본 (JPN)", "종목코드": "4912",
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

# --- [수정된 함수] 주가 데이터 조회 및 '봉 차트' 생성 ---
@st.cache_data
def plot_stock_chart(ticker: str) -> Optional[object]:
    """주어진 Ticker로 1년간의 주가 데이터를 조회하고 Plotly 봉 차트(Candlestick)를 생성합니다."""
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # FinanceDataReader를 통해 시가, 고가, 저가, 종가(OHLC) 데이터 가져오기
        df_stock = fdr.DataReader(ticker, start_date, end_date)
        
        if df_stock.empty:
            return None
            
        # Plotly Graph Objects를 사용하여 봉 차트 생성
        fig = go.Figure(data=[go.Candlestick(x=df_stock.index,
                                             open=df_stock['Open'],
                                             high=df_stock['High'],
                                             low=df_stock['Low'],
                                             close=df_stock['Close'])])
        
        # 차트 레이아웃 업데이트
        fig.update_layout(
            title=f"{ticker} 지난 1년간 일봉 차트",
            xaxis_title="날짜",
            yaxis_title="가격",
            xaxis_rangeslider_visible=False # 차트 아래의 미니 차트(rangeslider) 숨기기
        )
        return fig
    except Exception as e:
        st.warning(f"📈 주가 정보를 불러오는 중 오류 발생: {e}")
        return None

# --- 4. Streamlit 웹 애플리케이션 UI 구성 (기존과 동일) ---
st.set_page_config(page_title="StockLens", layout="wide")

if 'page' not in st.session_state: st.session_state.page = 'upload'
if 'image_source' not in st.session_state: st.session_state.image_source = None
if 'final_results' not in st.session_state: st.session_state.final_results = []

# --- 업로드 페이지 ---
if st.session_state.page == 'upload':
    st.title("📸 StockLens: AI 기업/제품 분석기")
    st.markdown("---")
    st.subheader("분석하고 싶은 제품들이 포함된 사진을 올려보세요")
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

# --- 결과 페이지 ---
elif st.session_state.page == 'results':
    st.title("🧠 AI 종합 분석 결과")
    st.markdown("---")
    st.image(st.session_state.image_source, caption="원본 이미지", width=400)
    st.markdown("---")

    if not st.session_state.final_results:
        with st.spinner('1단계: 이미지에서 분석할 객체들을 찾고 있습니다...'):
            cropped_objects = detect_objects_with_yolo(st.session_state.image_source)
        
        st.info(f"{len(cropped_objects)}개의 분석 가능한 객체를 찾았습니다. 각 객체에 대한 심층 분석을 시작합니다.")

        all_results = []
        progress_bar = st.progress(0, text="개별 객체 분석 진행 중...")
        for i, obj_bytes in enumerate(cropped_objects):
            with st.spinner(f"{i+1}/{len(cropped_objects)}번째 객체 분석 중..."):
                vision_results = analyze_image_with_vision_api(vision_client, obj_bytes)
                profile_info = get_company_profile_with_gemini(gemini_model, obj_bytes, vision_results)
            
            if profile_info and profile_info.get("제조사", "정보 없음") != "정보 없음":
                all_results.append({"image": obj_bytes, "profile": profile_info})
            
            progress_bar.progress((i + 1) / len(cropped_objects), text=f"개별 객체 분석 진행 중... ({i+1}/{len(cropped_objects)})")

        st.session_state.final_results = all_results
        progress_bar.empty()

    final_results = st.session_state.final_results
    if final_results:
        st.success(f"총 {len(final_results)}개의 유의미한 제품 정보를 찾았습니다.")
        for i, result in enumerate(final_results):
            profile = result["profile"]
            with st.expander(f"**결과 {i+1}: {profile.get('제품명', '이름 없는 제품')} ({profile.get('제조사', '제조사 없음')})**", expanded=i==0):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(result["image"], use_container_width=True)
                with col2:
                    manufacturer = profile.get("제조사", "정보 없음")
                    product_name = profile.get("제품명", "정보 없음")
                    country = profile.get("제조사_국가", "정보 없음")
                    ticker = profile.get("종목코드", "정보 없음")
                    st.markdown(f"**제조사:** {manufacturer} ({country})")
                    st.markdown(f"**제품명:** {product_name}")
                    st.markdown(f"**종목코드:** {ticker}")
                    
                    if ticker and ticker != "정보 없음":
                        chart_fig = plot_stock_chart(ticker)
                        if chart_fig:
                            st.plotly_chart(chart_fig, use_container_width=True)
                        else:
                            st.warning("주가 차트를 불러오는 데 실패했습니다.")
    else:
        st.error("이미지에서 제조사 정보를 가진 유의미한 제품을 찾지 못했습니다.")

    st.markdown("---")
    if st.button("⬅️ 다른 이미지 분석하기"):
        st.session_state.page = 'upload'
        st.session_state.image_source = None
        st.session_state.final_results = []
        st.rerun()