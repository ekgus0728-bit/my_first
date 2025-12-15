import streamlit as st
import pandas as pd
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from datetime import datetime
from wordcloud import WordCloud
import re
import networkx as nx
from itertools import combinations
import matplotlib.pyplot as plt

#=====AI코드를 인용해서 한글깨짐 방지====
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False


# 페이지 설정
st.set_page_config(page_title="케이팝 데몬 헌터스 대시보드", layout="wide")

# 제목
st.title("케이팝 데몬 헌터스 온라인 여론 분석")
st.write("학번: C321047 | 이름: 이다현") 


# ====== AI 코드참조=====
# CSV 불러오기
@st.cache_data
def load_data():
    df = pd.read_csv("kpop_demon_hunters_news.csv")
    return df

df = load_data()

# 날짜 컬럼 datetime 변환
df["pubDate"] = pd.to_datetime(df["pubDate"])


# 사이드바
st.sidebar.header("데이터 옵션")
num_rows = st.sidebar.slider("표시할 기사 수", 5, 1000, 20)

# 위젯 날짜 범위 선택
min_date = df["pubDate"].min()
max_date = df["pubDate"].max()

date_range = st.sidebar.date_input(
    "분석 기간 선택",
    [min_date, max_date]
)

# 위젯 키워드 검색
keyword = st.sidebar.text_input("제목 키워드 필터")

# 위젯 상위 키워드 개수
top_n = st.sidebar.selectbox("상위 키워드 개수", [10, 20, 30])

filtered_df = df.copy()

# 날짜 필터
if len(date_range) == 2:
    filtered_df = filtered_df[
        (filtered_df["pubDate"] >= pd.to_datetime(date_range[0])) &
        (filtered_df["pubDate"] <= pd.to_datetime(date_range[1]))
    ]

# 데이터 확인
st.subheader("수집된 뉴스 데이터")
st.dataframe(df.head(num_rows))

st.markdown("""
### 데이터 설명
- 네이버 뉴스 검색 API를 통해 수집된 기사 데이터
- 제목(title), 요약(description), 게시일(pubDate) 포함
""")


# 키워드 필터
if keyword:
    filtered_df = filtered_df[
        filtered_df["title"].str.contains(keyword, case=False, na=False)
    ]


st.subheader("기사 게시 추이 (Altair)")

daily_count = (
    filtered_df
    .groupby(filtered_df["pubDate"].dt.date)
    .size()
    .reset_index(name="count")
)

# ====AI 코드 참조(altair 시계열 그래프 x,y설정에 있어서)====
alt_chart = alt.Chart(daily_count).mark_line(point=True).encode(
    x="pubDate:T",
    y="count:Q",
    tooltip=["pubDate", "count"]
).properties(height=300)

st.altair_chart(alt_chart, use_container_width=True)

st.markdown("""
**해석**  
기사 수 변화를 봤을때 12월 11일에 기사량이 눈에 띄게 증가한 구간이 확인됩니다. 
이것은 케이팝 데몬 헌터스관련 이슈가 발생한 시점이라고 생각할 수 있겠고, 이 관심이 바로 줄어드는것을 보아 관련이슈가 짧은 기간 안에 발생했다는 것을 볼 수 있습니다.
""")
st.subheader("기사 제목 길이 분포 (Seaborn)")

filtered_df["title_length"] = filtered_df["title"].str.len()

fig, ax = plt.subplots()
sns.histplot(filtered_df["title_length"], bins=20, ax=ax)
ax.set_xlabel("제목 길이")
ax.set_ylabel("기사 수")

st.pyplot(fig)

st.markdown("""
**해석**  
기사 제목 길이는 30~40이 가장 많다고 보여집니다. 너무 긴 제목보다는 핵심 키워드를 포함한 비교적으로 짧은 제목을 사용한것을 알 수 있었습니다. 이것은 온라인에서 관심을 높이기 위한 제목작성방식이라고 해석할 수 있을거같습니다.
""")

st.subheader("핵심 키워드 빈도 (Plotly)")

words = filtered_df["title"].str.split().explode()
word_freq = words.value_counts().head(top_n).reset_index()
word_freq.columns = ["word", "count"]

fig2 = px.bar(
    word_freq,
    x="word",
    y="count",
    title="상위 키워드 빈도"
)

st.plotly_chart(fig2, use_container_width=True)

st.markdown("""
**해석**  
상위 키워드 빈도 분석 결과, 농심,수상,대상,케데헌,디자인,패키지,매기,강 등의 단어가 반복적으로 등장하는 것을 볼 수 있었습니다. 이것은 케데헌 관련 키워드빈도가 자체 콘텐츠에만 국한되있는것이아니라 캐릭터가 가지고있는 요소,디자인 이라든지 수상처럼 케데헌의 성과가 드러난다던지, 아니면 감독의 이름도 많이 언급되는 듯 팬덤 형성은 성과,캐릭터몰입 등등의 요소가 결합될 때 더 효과적으로 일어난것으로 해석할 수 있을거같습니다.
""")

st.subheader("키워드 WordCloud")

# HTML 태그 제거 함수
def clean_text(text):
    text = re.sub(r"<.*?>", "", str(text))
    return text

# title + description 합치기
filtered_df["clean_text"] = (
    filtered_df["title"].apply(clean_text) + " " +
    filtered_df["description"].apply(clean_text)
)

# 전체 텍스트 하나로 결합
text_data = " ".join(filtered_df["clean_text"].tolist())

# ===== AI 코드 참조(한글폰트)========
wordcloud = WordCloud(
    font_path="fonts/malgun.ttf", 
    background_color="white",
    width=800,
    height=400
).generate(text_data)

fig_wc, ax = plt.subplots(figsize=(10, 5))
ax.imshow(wordcloud)
ax.axis("off")

st.pyplot(fig_wc)

st.markdown("""
**해석**  
WordCloud 결과, 케이팝,데몬,헌터스, 넷플릭스,애니메이션,영화, 캐릭터 등등의 키워드가 크게나타났습니다. 이것은 케이팝데몬헌터스가 단순한 음악 콘텐츠가 아니라 애니메이션,영화적 요소도 잘되어있는 콘텐츠로 인식됨을 보여준다고 생각합니다. 그리고 넷플릭스키워드가 크게 나타난것을 봤을때. 플랫폼 자체의 영향력도 팬덤형성에 중요한 역할을 했다고 생각합니다. 그리고 캐릭터,디자인,세계관 등등의 단어들이 나타는것을 볼수있는데 이것은 케데헌을 즐겨보 팬들이 이러한 캐릭터설정,세계관 설정에몰입을 보이고있음을 의미합니다.
""")


from collections import Counter
from itertools import combinations

st.subheader("네이버 뉴스 키워드 네트워크")

# 토큰화 
all_tokens = (
    filtered_df["clean_text"]
    .str.split()
    .apply(lambda x: [w for w in x if len(w) > 1])
    .tolist()
)

# edge 리스트 생성
edge_list = []


#=======AI 코드 참조(키워드 네트워크 엣지 생성방식)====
for tokens in all_tokens:
    if len(tokens) > 1:
        edge_list.extend(combinations(sorted(set(tokens)), 2))

# edge 빈도 계산
edge_counts = Counter(edge_list)


#===AI 코드참조(엣지 필터링)======
st.caption(f"전체 edge 수: {len(edge_counts)}")
min_count = 20  
filtered_edges = {
    edge: weight
    for edge, weight in edge_counts.items()
    if weight >= min_count
}

st.caption(f"필터링된 edge 수: {len(filtered_edges)}")

import networkx as nx

G = nx.Graph()

weighted_edges = [
    (node1, node2, weight)
    for (node1, node2), weight in filtered_edges.items()
]

G.add_weighted_edges_from(weighted_edges)


#====AI 코드 참조(파라미터)=====
# 레이아웃 생성
pos_spring = nx.spring_layout(
    G,
    k=0.3,
    iterations=50,
    seed=42
)

# 노드 크기: 차수 기반
node_sizes = [G.degree(node) * 100 for node in G.nodes()]

# 엣지 두께: 가중치 기반
edge_widths = [G[u][v]["weight"] * 0.05 for u, v in G.edges()]

fig, ax = plt.subplots(figsize=(15, 15))

nx.draw_networkx(
    G,
    pos_spring,
    with_labels=True,
    node_size=node_sizes,
    width=edge_widths,
    font_family=plt.rcParams['font.family'],
    font_size=12,
    node_color="skyblue",
    edge_color="gray",
    alpha=0.8,
    ax=ax
)

ax.set_title("네이버 뉴스 키워드 네트워크", fontsize=20)
ax.axis("off")

st.pyplot(fig)

st.markdown("""
**해석**  
키워드 네트워크 그래프를 보면, 케이팝,데몬,헌터스,넷플릭스,애니메이션와 같은 키워드가 그래프 중심부에서 높은 연결성을 보여주는 것을 확인할 수 있었습니다. 이는 케이팝 데몬 헌터스가 단일 키워드 중심이 아닌, 음악(그중에서도 K-pop),애니메이션, 영화,캐릭터,넷플릭스 등등 이러한요소들이 강하게 결합된 구조라고 확인할 수 있습니다. 그리고 OST,농심,새우깡,데디 등등 이런 키워들이 하위 네트워크를 형성하고있는데 이것은 케데헌을 좋아하는 팬들이 단순히 스토리를 좋아하는 것을 넘어 캐릭터와 설정을 중심으로 콘텐츠를 해석하고 재생산하고 있음을 의미합니다.
""")


