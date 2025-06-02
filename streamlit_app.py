import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from wordcloud import WordCloud
import io
import base64
from PIL import Image
import numpy as np
import os
import re
import requests
import json
import networkx as nx
from collections import Counter
import itertools

# 한글 폰트 설정
# 폰트 파일 경로를 상수로 정의하여 재사용성을 높입니다.
FONT_PATH = "./fonts/Pretendard-Bold.ttf"

def setup_korean_font():
    """한글 폰트를 matplotlib에 설정"""
    if os.path.exists(FONT_PATH):
        # 폰트 등록
        font_prop = fm.FontProperties(fname=FONT_PATH)
        plt.rcParams['font.family'] = font_prop.get_name()
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지
        return True
    else:
        st.warning(f"한글 폰트 파일을 찾을 수 없어 기본 폰트를 사용합니다: {FONT_PATH}")
        st.info("`/fonts/Pretendard-Bold.ttf` 경로에 폰트 파일이 있는지 확인해주세요.")
        return False

# 앱 시작 시 폰트 설정
setup_korean_font()

# 페이지 설정
st.set_page_config(
    page_title="GPT API 키워드 추출 & 워드클라우드 생성기",
    page_icon="🤖",
    layout="wide"
)

def call_openai_api(text, api_key, model="gpt-4o-mini"): # 모델 기본값을 유효한 것으로 변경
    """OpenAI API를 호출하여 키워드 추출"""
    
    prompt = f"""
다음 텍스트에서 가장 중요한 키워드들을 추출하고 중요도에 따라 1~10 사이의 가중치를 부여해주세요.

텍스트: "{text}"

요구사항:
1. 10~20개의 핵심 키워드만 추출
2. 각 키워드에 1~10 사이의 가중치 부여 (10이 가장 중요)
3. 불용어(조사, 어미, 일반적인 단어) 제외
4. 복합어나 중요한 구문도 포함 가능
5. 아래 형식으로만 답변해주세요:

키워드A 5, 키워드B 4, 키워드C 3

위 형식 외의 다른 설명은 하지 마세요.
"""

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": model,
        "messages": [
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "max_tokens": 500,
        "temperature": 0.3
    }
    
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            return f"API 오류: {response.status_code} - {response.text}"
            
    except requests.exceptions.RequestException as e:
        return f"네트워크 오류: {str(e)}"
    except Exception as e:
        return f"오류 발생: {str(e)}"

def parse_gpt_response(response_text):
    """GPT 응답을 파싱하여 키워드 딕셔너리로 변환"""
    keywords_dict = {}
    
    try:
        # 쉼표로 분리
        items = response_text.split(',')
        
        for item in items:
            item = item.strip()
            if not item: # 빈 문자열 처리
                continue
            
            # 마지막 공백으로 분리된 숫자를 가중치로 처리
            parts = item.rsplit(' ', 1)
            
            if len(parts) == 2:
                keyword = parts[0].strip()
                try:
                    weight = int(parts[1].strip())
                    if 1 <= weight <= 10:  # 가중치 범위 검증
                        keywords_dict[keyword] = weight
                    else: # 가중치 범위 벗어남
                        keywords_dict[keyword] = 5 
                except ValueError: # 숫자가 아닌 경우
                    keywords_dict[item] = 5
            else: # 가중치가 없는 경우
                keywords_dict[item] = 5
                
        return keywords_dict
        
    except Exception as e:
        st.error(f"응답 파싱 오류: {str(e)}. 응답: '{response_text}'")
        return {}

def create_wordcloud_from_keywords(keywords_dict, width=800, height=600, bg_color='white'):
    """키워드 딕셔너리로부터 워드클라우드 생성"""
    if not keywords_dict:
        return None
    
    # 폰트 경로 확인
    if not os.path.exists(FONT_PATH):
        st.error(f"폰트 파일을 찾을 수 없습니다: {FONT_PATH}")
        st.info("폰트 파일이 `/fonts/Pretendard-Bold.ttf` 경로에 있는지 확인해주세요.")
        return None
    
    try:
        # 워드클라우드 생성
        wc = WordCloud(
            font_path=FONT_PATH, # 상수 사용
            width=width,
            height=height,
            background_color=bg_color,
            max_words=100,
            relative_scaling=0.5,
            min_font_size=10,
            colormap='viridis',
            prefer_horizontal=0.7
        ).generate_from_frequencies(keywords_dict)
        
        return wc
    except Exception as e:
        st.error(f"워드클라우드 생성 중 오류가 발생했습니다: {str(e)}")
        return None

def wordcloud_to_image(wc, width=800, height=600):
    """워드클라우드를 PIL Image로 변환"""
    fig, ax = plt.subplots(figsize=(width/100, height/100)) # dpi를 100으로 가정하여 figsize 계산
    ax.imshow(wc, interpolation='bilinear')
    ax.axis('off')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=100)
    buf.seek(0)
    plt.close(fig) # 그래프 리소스 해제
    
    image = Image.open(buf)
    return image

# 이 함수는 analyze_network_metrics 함수 내부에 있어서는 안 됩니다.
def get_image_download_link(img, filename, link_text):
    """이미지 다운로드 링크 생성"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{img_str}" download="{filename}">{link_text}</a>'
    return href


def create_network_analysis(text, keywords_dict, min_cooccurrence=1):
    """텍스트에서 키워드 간 네트워크 분석"""
    if not keywords_dict or len(keywords_dict) < 2:
        return None, None
    
    # 텍스트를 문장으로 분리
    sentences = re.split(r'[.!?]\s*', text)
    
    # 키워드 동시출현 매트릭스 생성
    co_occurrence = Counter()
    keyword_list = list(keywords_dict.keys())
    
    for sentence in sentences:
        # 문장에 포함된 키워드들 찾기
        found_keywords = []
        for keyword in keyword_list:
            # 대소문자 구분 없이 찾기 위해 소문자로 변환하거나 정규식 사용
            # 여기서는 간단하게 키워드 자체가 문장에 있는지 확인
            if keyword in sentence: 
                found_keywords.append(keyword)
        
        # 같은 문장에 나타난 키워드들의 조합 생성
        for combo in itertools.combinations(found_keywords, 2):
            # 알파벳 순으로 정렬하여 (A,B)와 (B,A)를 같게 처리
            sorted_combo = tuple(sorted(combo))
            co_occurrence[sorted_combo] += 1
    
    # 최소 동시출현 횟수 필터링
    filtered_co_occurrence = {k: v for k, v in co_occurrence.items() if v >= min_cooccurrence}

    if not filtered_co_occurrence:
        return None, None
    
    # 네트워크 그래프 생성
    G = nx.Graph()
    
    # 키워드를 노드로 추가 (가중치를 노드 크기로 사용)
    # 동시 출현된 키워드만 노드로 추가 (고립 노드 방지)
    nodes_to_add = set()
    for (k1, k2) in filtered_co_occurrence.keys():
        nodes_to_add.add(k1)
        nodes_to_add.add(k2)

    for keyword in nodes_to_add:
        weight = keywords_dict.get(keyword, 1) # 추출된 키워드에 없는 경우 기본값 1
        G.add_node(keyword, weight=weight)
    
    # 동시출현을 엣지로 추가
    for (keyword1, keyword2), freq in filtered_co_occurrence.items():
        G.add_edge(keyword1, keyword2, weight=freq)
    
    if G.number_of_nodes() < 2: # 최소 2개 노드 이상이어야 의미 있는 그래프
        return None, None

    return G, filtered_co_occurrence # 필터링된 co_occurrence 반환

def draw_network_graph(G, keywords_dict):
    """네트워크 그래프 그리기"""
    if G is None or len(G.nodes()) < 2:
        return None
    
    # 한글 폰트 설정
    font_prop = None
    if os.path.exists(FONT_PATH): # FONT_PATH 상수를 사용합니다.
        font_prop = fm.FontProperties(fname=FONT_PATH)
    
    # 그래프 레이아웃 설정
    fig, ax = plt.subplots(figsize=(12, 8)) # Streamlit에서 fig를 관리해야 함
    
    # 스프링 레이아웃 사용
    # k 값 조정으로 노드 간 간격 조절
    pos = nx.spring_layout(G, k=0.8 / np.sqrt(G.number_of_nodes()), iterations=50) 
    
    # 노드 크기 설정 (키워드 가중치 기반)
    node_sizes = [keywords_dict.get(node, 1) * 300 for node in G.nodes()] # 추출된 키워드 가중치 사용
    
    # 엣지 두께 설정 (동시출현 빈도 기반)
    edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
    edge_widths = [w * 1.5 for w in edge_weights] # 엣지 두께 배율 조정
    
    # 노드 그리기
    nx.draw_networkx_nodes(G, pos, 
                          node_size=node_sizes, 
                          node_color='lightblue', 
                          alpha=0.7,
                          edgecolors='darkblue',
                          linewidths=1.5,
                          ax=ax) # ax 명시
    
    # 엣지 그리기
    nx.draw_networkx_edges(G, pos, 
                          width=edge_widths, 
                          alpha=0.6, 
                          edge_color='gray',
                          ax=ax) # ax 명시
    
    # 라벨 그리기
    labels = {node: node for node in G.nodes()} # 모든 노드에 라벨 적용
    
    # 폰트 속성 적용
    if font_prop:
        nx.draw_networkx_labels(G, pos, 
                                labels=labels,
                               font_size=9, 
                               font_color='black',
                               font_weight='bold',
                               font_family=font_prop.get_name(), # <-- 이 부분을 수정했습니다!
                               ax=ax) # ax 명시
    else:
        nx.draw_networkx_labels(G, pos, 
                                labels=labels,
                               font_size=9, 
                               font_color='black',
                               font_weight='bold',
                               ax=ax) # ax 명시
    
    ax.set_title('키워드 네트워크 분석', 
              fontproperties=font_prop if font_prop else None, 
              fontsize=16, 
              fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    
    # 이미지로 변환
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf)
    plt.close(fig) # 그래프 리소스 해제
    
    return image

def analyze_network_metrics(G, keywords_dict):
    """네트워크 지표 분석"""
    if G is None or len(G.nodes()) < 2:
        return {}
    
    metrics = {}
    
    # 기본 네트워크 정보
    metrics['nodes'] = G.number_of_nodes()
    metrics['edges'] = G.number_of_edges()
    
    # connected components는 그래프가 분리되어 있는지 확인
    if nx.is_connected(G):
        metrics['density'] = nx.density(G)
        metrics['components'] = 1
    else:
        metrics['density'] = "그래프가 연결되어 있지 않음 (밀도 계산 불가)"
        metrics['components'] = nx.number_connected_components(G)
    
    # 중심성 지표 계산
    try:
        # 가중치가 없는 그래프에 대한 중심성
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        metrics['most_connected'] = max(degree_centrality, key=degree_centrality.get) if degree_centrality else "N/A"
        metrics['most_between'] = max(betweenness_centrality, key=betweenness_centrality.get) if betweenness_centrality else "N/A"
        metrics['most_close'] = max(closeness_centrality, key=closeness_centrality.get) if closeness_centrality else "N/A"
        
        # 상위 5개 키워드의 중심성 점수
        metrics['degree_top5'] = dict(sorted(degree_centrality.items(), 
                                           key=lambda x: x[1], reverse=True)[:5])
        metrics['betweenness_top5'] = dict(sorted(betweenness_centrality.items(), 
                                                key=lambda x: x[1], reverse=True)[:5])
        metrics['closeness_top5'] = dict(sorted(closeness_centrality.items(), 
                                                key=lambda x: x[1], reverse=True)[:5])
        
    except Exception as e:
        st.warning(f"네트워크 중심성 지표 계산 중 오류 발생: {e}")
        metrics['most_connected'] = "계산 불가"
        metrics['most_between'] = "계산 불가"
        metrics['most_close'] = "계산 불가"
        metrics['degree_top5'] = {}
        metrics['betweenness_top5'] = {}
        metrics['closeness_top5'] = {}
    
    return metrics


# 앱 제목 및 설명
st.title("🤖 GPT API 키워드 추출 & 워드클라우드 생성기")
st.markdown("GPT API를 활용하여 텍스트에서 핵심 키워드를 추출하고 가중치와 함께 워드클라우드를 생성합니다!")

# 사이드바 설정
st.sidebar.header("🔑 API 설정")

# API 키 입력
api_key = st.sidebar.text_input(
    "OpenAI API Key", 
    type="password",
    help="OpenAI API 키를 입력하세요. (https://platform.openai.com/api-keys)"
)

# GPT 모델 선택 (유효한 모델 이름으로 변경)
model_choice = st.sidebar.selectbox(
    "GPT 모델",
    ["gpt-4o-mini", "gpt-3.5-turbo"], # 유효한 모델 이름으로 변경
    index=0,
    help="사용할 GPT 모델을 선택하세요. gpt-4o-mini가 가장 경제적이면서도 성능이 좋습니다."
)

# 워드클라우드 설정
st.sidebar.header("🎨 시각화 설정")
wc_width = st.sidebar.slider("워드클라우드 너비", 400, 1200, 800)
wc_height = st.sidebar.slider("워드클라우드 높이", 300, 800, 600)
bg_color = st.sidebar.selectbox("배경색", ["white", "black"], index=0)

# 네트워크 분석 설정
show_network = st.sidebar.checkbox("🌐 네트워크 분석 포함", value=True)
min_cooccurrence = st.sidebar.slider("최소 동시출현 횟수", 1, 5, 1, 
                                    help="이 값 이상으로 함께 나타나는 키워드들만 연결선으로 표시")

# API 키 확인
if not api_key:
    st.warning("⚠️ OpenAI API 키를 왼쪽 사이드바에 입력해주세요.")
    st.info("""
    **API 키 발급 방법:**
    1. https://platform.openai.com 접속
    2. 로그인 후 'API Keys' 메뉴로 이동
    3. 'Create new secret key' 클릭
    4. 생성된 키를 복사하여 사이드바에 입력
    
    **주의사항:**
    - API 사용료가 부과됩니다
    - 키는 안전하게 보관하세요
    
    **폰트 파일 설정:**
    - `fonts/Pretendard-Bold.ttf` 파일이 필요합니다. `/fonts` 디렉토리를 생성하고 그 안에 넣어주세요.
    """)

# 메인 인터페이스
st.header("📝 텍스트 입력")

# 사용법 안내
with st.expander("💡 사용법 및 예시"):
    st.markdown("""
    **사용법:**
    1. OpenAI API 키를 사이드바에 입력
    2. 분석할 텍스트를 입력
    3. 'GPT로 키워드 추출' 버튼 클릭
    4. 추출된 키워드, 워드클라우드 및 네트워크 분석 결과 확인
    
    **예시 텍스트:**
    ```
    인공지능과 머신러닝은 현대 기술의 핵심 분야입니다. 
    데이터 과학자들은 빅데이터를 분석하여 패턴을 찾고, 
    딥러닝 알고리즘을 통해 예측 모델을 구축합니다. 
    자연어 처리와 컴퓨터 비전 기술이 발전하면서 
    다양한 산업 분야에서 혁신이 일어나고 있습니다.
    ```
    
    **GPT의 장점:**
    - 문맥을 이해한 정확한 키워드 추출
    - 동의어/유의어 그룹핑
    - 중요도에 따른 정교한 가중치 부여
    - 복합어와 전문용어 인식
    """)

# 텍스트 입력 영역
text_input = st.text_area(
    "분석할 텍스트를 입력하세요",
    height=200,
    placeholder="예시: 인공지능과 머신러닝은 현대 기술의 핵심 분야입니다...",
    help="GPT가 이 텍스트를 분석하여 핵심 키워드를 추출합니다."
)

# 키워드 추출 버튼
if st.button("🤖 GPT로 키워드 추출", type="primary", disabled=not api_key):
    if not api_key:
        st.error("OpenAI API 키를 먼저 입력해주세요.")
    elif not text_input.strip():
        st.warning("분석할 텍스트를 입력해주세요.")
    else:
        with st.spinner(f"{model_choice} 모델로 키워드를 추출하는 중..."):
            # GPT API 호출
            gpt_response = call_openai_api(text_input, api_key, model_choice)
            
            if gpt_response.startswith("API 오류") or gpt_response.startswith("네트워크 오류") or gpt_response.startswith("오류 발생"):
                st.error(gpt_response)
            else:
                # GPT 응답 표시
                st.success("키워드 추출 완료!")
                
                st.header("🤖 GPT 응답")
                st.code(gpt_response, language=None)
                
                # 키워드 파싱
                keywords_dict = parse_gpt_response(gpt_response)
                
                if keywords_dict:
                    st.header("📊 추출된 키워드")
                    
                    # 키워드 정보 표시
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.subheader("🎯 키워드 목록")
                        # 가중치가 0인 키워드는 제외하고 표시 (parse_gpt_response에서 0이 될 일은 없지만, 안전하게)
                        display_keywords = {k: v for k, v in keywords_dict.items() if v > 0}
                        if display_keywords:
                            for keyword, weight in sorted(display_keywords.items(), key=lambda x: x[1], reverse=True):
                                st.write(f"**{keyword}**: {weight}")
                        else:
                            st.info("추출된 유효한 키워드가 없습니다.")

                    with col2:
                        st.subheader("📈 가중치 분포")
                        weights = list(keywords_dict.values())
                        
                        fig, ax = plt.subplots(figsize=(6, 4))
                        ax.hist(weights, bins=range(1, 12), alpha=0.7, color='skyblue', edgecolor='black')
                        
                        # 한글 폰트 적용 (setup_korean_font에서 설정했으므로 font_family는 전역으로 적용되지만, 명시적으로 fontproperties를 사용할 수 있음)
                        if os.path.exists(FONT_PATH):
                            font_prop = fm.FontProperties(fname=FONT_PATH)
                            ax.set_xlabel('가중치', fontproperties=font_prop)
                            ax.set_ylabel('키워드 수', fontproperties=font_prop)
                            ax.set_title('키워드 가중치 분포', fontproperties=font_prop)
                        else:
                            ax.set_xlabel('Weight')
                            ax.set_ylabel('Keywords Count')
                            ax.set_title('Keyword Weight Distribution')
                            
                        ax.set_xticks(range(1, 11))
                        ax.grid(True, alpha=0.3)
                        st.pyplot(fig)
                        plt.close(fig) # 그래프 리소스 해제
                    
                    # 복사 가능한 결과
                    st.subheader("📋 복사용 결과")
                    st.text_area(
                        "워드클라우드 생성기에 사용할 수 있는 형식", 
                        value=gpt_response, 
                        height=100,
                        help="이 텍스트를 복사해서 다른 워드클라우드 도구에 사용할 수 있습니다."
                    )
                    
                    # 워드클라우드 생성
                    st.header("☁️ 워드클라우드")
                    
                    with st.spinner("워드클라우드를 생성하는 중..."):
                        wordcloud = create_wordcloud_from_keywords(keywords_dict, wc_width, wc_height, bg_color)
                        
                        if wordcloud:
                            # 워드클라우드를 이미지로 변환
                            img = wordcloud_to_image(wordcloud, wc_width, wc_height)
                            
                            # 이미지 표시
                            st.image(img, caption="GPT로 생성된 워드클라우드", use_column_width=True)
                            
                            # 다운로드 링크 (수정된 함수 호출)
                            download_link = get_image_download_link(img, "gpt_wordcloud.png", "📥 워드클라우드 다운로드")
                            st.markdown(download_link, unsafe_allow_html=True)
                            
                            # 생성 정보
                            st.info(f"모델: {model_choice} | 크기: {wc_width}x{wc_height} | 키워드 수: {len(keywords_dict)}개")
                
                else:
                    st.error("GPT 응답을 파싱할 수 없습니다. 응답 형식을 확인해주세요.")

                # 🌐 네트워크 분석 섹션 (새로 추가되거나 수정된 부분)
                if show_network:
                    st.header("🌐 키워드 네트워크 분석")
                    if len(keywords_dict) < 2:
                        st.warning("네트워크 분석을 위해서는 최소 2개 이상의 키워드가 필요합니다.")
                    else:
                        with st.spinner("키워드 네트워크를 분석하고 그리는 중..."):
                            G, co_occurrence_data = create_network_analysis(text_input, keywords_dict, min_cooccurrence)
                            
                            if G and G.number_of_nodes() >= 2:
                                network_img = draw_network_graph(G, keywords_dict)
                                if network_img:
                                    st.image(network_img, caption="키워드 네트워크 그래프", use_column_width=True)
                                    network_download_link = get_image_download_link(network_img, "keyword_network.png", "📥 네트워크 그래프 다운로드")
                                    st.markdown(network_download_link, unsafe_allow_html=True)

                                    st.subheader("📊 네트워크 지표")
                                    metrics = analyze_network_metrics(G, keywords_dict)
                                    if metrics:
                                        st.write(f"**총 노드 수 (키워드):** {metrics.get('nodes', 'N/A')}개")
                                        st.write(f"**총 엣지 수 (연결):** {metrics.get('edges', 'N/A')}개")
                                        st.write(f"**그래프 밀도:** {metrics.get('density', 'N/A'):.4f}" if isinstance(metrics.get('density'), float) else f"**그래프 밀도:** {metrics.get('density', 'N/A')}")
                                        st.write(f"**연결된 컴포넌트 수:** {metrics.get('components', 'N/A')}개")
                                        
                                        st.markdown("---")
                                        st.subheader("⭐ 주요 키워드 (중심성 지표)")
                                        st.markdown("**연결 중심성 (Degree Centrality):** 가장 많은 키워드와 연결된 키워드 (영향력)")
                                        if metrics.get('degree_top5'):
                                            for k, v in metrics['degree_top5'].items():
                                                st.write(f"- **{k}**: {v:.4f}")
                                        else:
                                            st.write("정보 없음")

                                        st.markdown("**매개 중심성 (Betweenness Centrality):** 다른 키워드들 사이의 다리 역할을 하는 키워드 (중개자 역할)")
                                        if metrics.get('betweenness_top5'):
                                            for k, v in metrics['betweenness_top5'].items():
                                                st.write(f"- **{k}**: {v:.4f}")
                                        else:
                                            st.write("정보 없음")
                                            
                                        st.markdown("**근접 중심성 (Closeness Centrality):** 다른 키워드에 얼마나 가깝게 도달할 수 있는지 나타내는 키워드 (정보 확산 용이성)")
                                        if metrics.get('closeness_top5'):
                                            for k, v in metrics['closeness_top5'].items():
                                                st.write(f"- **{k}**: {v:.4f}")
                                        else:
                                            st.write("정보 없음")

                                    st.subheader("↔️ 키워드 동시출현 빈도 (상위 10개)")
                                    if co_occurrence_data:
                                        sorted_co_occurrence = sorted(co_occurrence_data.items(), key=lambda item: item[1], reverse=True)[:10]
                                        for (k1, k2), freq in sorted_co_occurrence:
                                            st.write(f"- **{k1}**와 **{k2}**: {freq}회")
                                    else:
                                        st.info("동시출현 키워드가 부족하거나 최소 동시출현 횟수 설정이 너무 높습니다.")
                                else:
                                    st.error("네트워크 그래프 생성에 실패했습니다.")
                            else:
                                st.info(f"네트워크를 그릴 수 있는 충분한 키워드 연결이 없습니다. (현재 노드 수: {G.number_of_nodes() if G else 0})")
                                if G and G.number_of_nodes() > 0:
                                    st.info(f"선택된 최소 동시출현 횟수 ({min_cooccurrence}회) 이상으로 함께 나타나는 키워드가 부족합니다. 설정을 낮춰보세요.")

# 비용 안내
if api_key:
    st.sidebar.markdown("---")
    st.sidebar.header("💰 비용 안내")
    st.sidebar.markdown("""
    **예상 비용 (1회 요청):**
    - `gpt-4o-mini`: ~$0.000004/token (입력), ~$0.000012/token (출력)
    - `gpt-3.5-turbo`: ~$0.0000005/token (입력), ~$0.0000015/token (출력)
    
    *실제 비용은 입력 및 출력 토큰 수에 따라 달라집니다.*
    *`gpt-4o-mini`가 `gpt-3.5-turbo`보다 비싸지만, 훨씬 더 높은 품질의 키워드 추출이 가능합니다.*
    """)


# 푸터
st.markdown("---")
st.markdown("💡 **팁**: GPT가 문맥을 이해하므로 더 정확하고 의미있는 키워드를 추출할 수 있습니다!")
st.markdown("🤖 GPT API 키워드 추출기 | Made with Streamlit")