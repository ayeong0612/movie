
import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(
    page_title="🎬 영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")
st.write("영화의 상영 규모와 관객 특성을 바탕으로 비슷한 영화끼리 세 유형으로 묶습니다.")

FEATURE_LABELS = {
    "log_first_scrn": "스크린 수 (상용로그)",
    "log_total_audi": "누적 관객 (상용로그)",
    "days_in_top10": "10위권 일수",
    "longrun_index": "롱런 지수",
}
FEATURES = list(FEATURE_LABELS.keys())
SYMBOLS = ["㉮", "㉯", "㉰"]
COLORS = {
    "㉮": "#e76f51",
    "㉯": "#2a9d8f",
    "㉰": "#457b9d",
}


@st.cache_data
def load_movies():
    df = pd.read_csv(DATA_URL, encoding="utf-8")

    required = [
        "movieCd",
        "movieNm",
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError("필수 열이 없습니다: " + ", ".join(missing))

    numeric_columns = [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 롱런 지수 = 누적 관객 수 / 첫 주 관객 수
    df["longrun_index"] = float("nan")
    first_week_valid = df["first_week_audi"] > 0

    df.loc[first_week_valid, "longrun_index"] = (
        df.loc[first_week_valid, "total_audi"]
        / df.loc[first_week_valid, "first_week_audi"]
    )

    # 롱런 지수는 20을 넘지 않도록 제한
    df["longrun_index"] = df["longrun_index"].clip(upper=20)

    # 스크린 수와 누적 관객 수에 상용로그 적용
    df["log_first_scrn"] = float("nan")
    positive_screens = df["first_scrn"] > 0

    df.loc[positive_screens, "log_first_scrn"] = (
        df.loc[positive_screens, "first_scrn"].map(math.log10)
    )

    df["log_total_audi"] = float("nan")
    positive_audience = df["total_audi"] > 0

    df.loc[positive_audience, "log_total_audi"] = (
        df.loc[positive_audience, "total_audi"].map(math.log10)
    )

    return df


try:
    raw = load_movies()
except Exception as error:
    st.error(f"데이터를 불러오지 못했습니다: {error}")
    st.info("인터넷 연결과 CSV 주소를 확인해 주세요.")
    st.stop()


total_count = len(raw)

# 네 속성 모두 값이 있고 계산 가능한 영화만 분석
valid = raw[
    raw["first_week_audi"].notna()
    & (raw["first_week_audi"] > 0)
    & raw[FEATURES].notna().all(axis=1)
    & (raw["first_scrn"] > 0)
    & (raw["total_audi"] > 0)
].copy()

st.caption(
    f"전체 영화 {total_count:,}편"
    f"  ·  분석에 묶은 영화 {len(valid):,}편"
)

if len(valid) < 3:
    st.error("세 묶음을 만들려면 분석 가능한 영화가 최소 3편 필요합니다.")
    st.stop()


# --------------------------------------------------
# 1. 분석 속성 선택
# --------------------------------------------------

st.subheader("1. 묶는 데 사용할 속성 선택")

selected_features = st.multiselect(
    "분석에 사용할 속성을 두 개 이상 고르세요.",
    options=FEATURES,
    default=FEATURES,
    format_func=lambda feature: FEATURE_LABELS[feature],
)

if len(selected_features) < 2:
    st.warning("군집 분석을 위해 속성을 두 개 이상 선택해 주세요.")
    st.stop()


# --------------------------------------------------
# 2. K-평균 군집 분석
# --------------------------------------------------

scaler = StandardScaler()
scaled_features = scaler.fit_transform(
    valid[selected_features].astype(float)
)

model = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

valid["군집번호"] = model.fit_predict(scaled_features)

# 누적 관객 평균이 가장 높은 군집부터 ㉮, ㉯, ㉰ 배정
cluster_order = (
    valid.groupby("군집번호")["total_audi"]
    .mean()
    .sort_values(ascending=False)
    .index.tolist()
)

cluster_labels = {
    cluster_id: SYMBOLS[index]
    for index, cluster_id in enumerate(cluster_order)
}

valid["영화 유형"] = valid["군집번호"].map(cluster_labels)
valid["유형"] = valid["영화 유형"]

color_map = {
    symbol: COLORS[symbol]
    for symbol in SYMBOLS
}


# --------------------------------------------------
# 3. 2차원 산점도
# --------------------------------------------------

st.subheader("2. 2차원 산점도")

col_x, col_y = st.columns(2)

with col_x:
    x_feature = st.selectbox(
        "가로축 속성",
        options=selected_features,
        index=0,
        format_func=lambda feature: FEATURE_LABELS[feature],
        key="scatter_x",
    )

with col_y:
    y_feature = st.selectbox(
        "세로축 속성",
        options=selected_features,
        index=1,
        format_func=lambda feature: FEATURE_LABELS[feature],
        key="scatter_y",
    )

fig2 = px.scatter(
    valid,
    x=x_feature,
    y=y_feature,
    color="유형",
    color_discrete_map=color_map,
    category_orders={"유형": SYMBOLS},
    hover_name="movieNm",
    hover_data={
        "영화 유형": True,
        "first_scrn": ":,",
        "total_audi": ":,",
        "days_in_top10": True,
        "longrun_index": ":.2f",
    },
    labels={
        **FEATURE_LABELS,
        "유형": "영화 유형",
        "movieNm": "영화 제목",
    },
    title="영화 유형별 2차원 분포",
)

fig2.update_traces(
    marker={"size": 8, "opacity": 0.78}
)
fig2.update_layout(legend_title_text="영화 유형")

st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# 4. 3차원 산점도
# --------------------------------------------------

st.subheader("3. 3차원 산점도")

if len(selected_features) < 3:
    st.info(
        "3차원 산점도를 보려면 분석 속성을 "
        "3개 이상 선택해 주세요."
    )
else:
    col_x3, col_y3, col_z3 = st.columns(3)

    with col_x3:
        x3 = st.selectbox(
            "X축 속성",
            selected_features,
            index=0,
            format_func=lambda feature: FEATURE_LABELS[feature],
            key="x3",
        )

    with col_y3:
        y3 = st.selectbox(
            "Y축 속성",
            selected_features,
            index=1,
            format_func=lambda feature: FEATURE_LABELS[feature],
            key="y3",
        )

    with col_z3:
        z3 = st.selectbox(
            "Z축 속성",
            selected_features,
            index=2,
            format_func=lambda feature: FEATURE_LABELS[feature],
            key="z3",
        )

    fig3 = px.scatter_3d(
        valid,
        x=x3,
        y=y3,
        z=z3,
        color="유형",
        color_discrete_map=color_map,
        category_orders={"유형": SYMBOLS},
        hover_name="movieNm",
        hover_data={
            "영화 유형": True,
            "first_scrn": ":,",
            "total_audi": ":,",
            "days_in_top10": True,
            "longrun_index": ":.2f",
        },
        labels={
            **FEATURE_LABELS,
            "유형": "영화 유형",
            "movieNm": "영화 제목",
        },
        title="영화 유형별 3차원 분포",
    )

    fig3.update_traces(
        marker={"size": 3, "opacity": 0.8}
    )
    fig3.update_layout(legend_title_text="영화 유형")

    st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------
# 5. 유형별 편수와 속성 평균
# --------------------------------------------------

st.subheader("4. 유형별 편수와 속성 평균")

summary = valid.groupby("영화 유형").agg(
    편수=("movieNm", "size"),
    스크린수_평균=("first_scrn", "mean"),
    누적관객_평균=("total_audi", "mean"),
    십위권일수_평균=("days_in_top10", "mean"),
    롱런지수_평균=("longrun_index", "mean"),
)

summary = summary.reindex(SYMBOLS).reset_index()

summary = summary.rename(columns={
    "스크린수_평균": "스크린 수 평균",
    "누적관객_평균": "누적 관객 평균",
    "십위권일수_평균": "10위권 일수 평균",
    "롱런지수_평균": "롱런 지수 평균",
})

st.dataframe(
    summary.style.format({
        "편수": "{:,.0f}",
        "스크린 수 평균": "{:,.1f}",
        "누적 관객 평균": "{:,.0f}",
        "10위권 일수 평균": "{:.2f}",
        "롱런 지수 평균": "{:.2f}",
    }),
    use_container_width=True,
    hide_index=True,
)


# --------------------------------------------------
# 6. 유형별 누적 관객 상위 영화 5편
# --------------------------------------------------

st.subheader("5. 유형별 누적 관객 상위 영화 5편")

for symbol in SYMBOLS:
    group = (
        valid[valid["영화 유형"] == symbol]
        .sort_values("total_audi", ascending=False)
        .head(5)
    )

    with st.expander(
        f"{symbol} 유형 — 누적 관객 상위 {len(group)}편",
        expanded=True,
    ):
        if group.empty:
            st.write("이 유형에 해당하는 영화가 없습니다.")
        else:
            top = group[
                [
                    "movieNm",
                    "total_audi",
                    "first_week_audi",
                    "first_scrn",
                    "days_in_top10",
                    "longrun_index",
                ]
            ].copy()

            top = top.rename(columns={
                "movieNm": "영화 제목",
                "total_audi": "누적 관객",
                "first_week_audi": "첫 주 관객",
                "first_scrn": "스크린 수",
                "days_in_top10": "10위권 일수",
                "longrun_index": "롱런 지수",
            })

            st.dataframe(
                top.style.format({
                    "누적 관객": "{:,.0f}",
                    "첫 주 관객": "{:,.0f}",
                    "스크린 수": "{:,.0f}",
                    "10위권 일수": "{:,.0f}",
                    "롱런 지수": "{:.2f}",
                }),
                use_container_width=True,
                hide_index=True,
            )


st.caption(
    "선택한 속성을 StandardScaler로 표준화한 뒤 "
    "KMeans(n_clusters=3, random_state=42, n_init=10)로 군집화했습니다. "
    "스크린 수와 누적 관객은 상용로그(log10)를 사용하고, "
    "롱런 지수는 누적 관객 ÷ 첫 주 관객으로 계산해 "
    "최댓값을 20으로 제한했습니다."
)
