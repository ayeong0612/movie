
import math

import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

DATA_URL = "https://raw.githubusercontent.com/it-teacher-go/data-share-/main/kobis_movies.csv"

st.set_page_config(
    page_title="🎬 영화 유형 나누기",
    page_icon="🎬",
    layout="wide",
)

st.title("🎬 영화 유형 나누기")
st.write("영화의 상영 규모와 관객 특성을 바탕으로 비슷한 영화끼리 묶습니다.")

FEATURE_LABELS = {
    "log_first_scrn": "스크린 수 (상용로그)",
    "log_total_audi": "누적 관객 (상용로그)",
    "days_in_top10": "10위권 일수",
    "longrun_index": "롱런 지수",
}

FEATURES = list(FEATURE_LABELS.keys())
SYMBOLS = ["㉮", "㉯", "㉰", "㉱", "㉲", "㉳", "㉴"]
COLORS = [
    "#e76f51",
    "#2a9d8f",
    "#457b9d",
    "#e9c46a",
    "#9b5de5",
    "#f15bb5",
    "#6c757d",
]


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

    for column in [
        "first_scrn",
        "first_week_audi",
        "total_audi",
        "days_in_top10",
    ]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # 롱런 지수 = 누적 관객 수 / 첫 주 관객 수
    df["longrun_index"] = float("nan")
    first_week_valid = df["first_week_audi"] > 0

    df.loc[first_week_valid, "longrun_index"] = (
        df.loc[first_week_valid, "total_audi"]
        / df.loc[first_week_valid, "first_week_audi"]
    )

    # 롱런 지수가 20을 넘으면 20으로 제한
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

# 분석 속성에 결측값이 있거나 계산할 수 없는 영화는 제외
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

if len(valid) < 7:
    st.error("묶음 수 1~7개를 비교하려면 분석 가능한 영화가 최소 7편 필요합니다.")
    st.stop()


# --------------------------------------------------
# 1. 묶음 수와 분석 속성 선택
# --------------------------------------------------

st.subheader("1. 분석 조건 설정")

col_k, col_features = st.columns([1, 3])

with col_k:
    cluster_count = st.selectbox(
        "묶음 수",
        options=list(range(2, 8)),
        index=1,
    )

with col_features:
    selected_features = st.multiselect(
        "묶음 분석에 사용할 속성 (2개 이상)",
        options=FEATURES,
        default=FEATURES,
        format_func=lambda feature: FEATURE_LABELS[feature],
    )

if len(selected_features) < 2:
    st.warning("분석 속성을 두 개 이상 선택해 주세요.")
    st.stop()


# 선택한 속성만 표준화
scaler = StandardScaler()
X_scaled = scaler.fit_transform(
    valid[selected_features].astype(float)
)


# --------------------------------------------------
# 2. 묶음 수별 WCSS 계산
# --------------------------------------------------

st.subheader("2. 적절한 묶음 수 살펴보기")

st.write(
    "WCSS는 각 영화와 자신이 속한 묶음 중심 사이의 "
    "거리 제곱을 모두 더한 값입니다. "
    "아래 계산에는 현재 선택한 속성만 사용합니다."
)

wcss_rows = []
max_k = min(7, len(valid) - 1)

for k in range(1, max_k + 1):
    elbow_model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
    )

    elbow_model.fit(X_scaled)

    wcss_rows.append({
        "묶음 수": k,
        "WCSS": elbow_model.inertia_,
    })

wcss_df = pd.DataFrame(wcss_rows)

fig_elbow = px.line(
    wcss_df,
    x="묶음 수",
    y="WCSS",
    markers=True,
    title="묶음 수에 따른 WCSS 변화",
)

fig_elbow.update_traces(
    line={"width": 3},
    marker={"size": 9},
)

fig_elbow.update_layout(
    xaxis={
        "tickmode": "linear",
        "dtick": 1,
        "title": "묶음 수",
    },
    yaxis_title="묶음 내 제곱거리 합 (WCSS)",
    hovermode="x unified",
)

# 현재 선택한 묶음 수 위치에 세로선 표시
fig_elbow.add_vline(
    x=cluster_count,
    line_dash="dash",
    line_width=2,
    line_color="red",
    annotation_text=f"현재 선택: {cluster_count}개",
    annotation_position="top right",
)

st.plotly_chart(fig_elbow, use_container_width=True)

# 바로 앞 묶음 수의 WCSS에서 현재 WCSS를 뺀 값
wcss_df["직전 값 대비 감소량"] = (
    wcss_df["WCSS"].shift(1) - wcss_df["WCSS"]
)

wcss_display = wcss_df.copy()

wcss_display["WCSS"] = wcss_display["WCSS"].map(
    lambda value: f"{value:,.2f}"
)

wcss_display["직전 값 대비 감소량"] = (
    wcss_display["직전 값 대비 감소량"].map(
        lambda value: "" if pd.isna(value) else f"{value:,.2f}"
    )
)

st.dataframe(
    wcss_display,
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "WCSS는 묶음 수가 증가하면 대체로 감소합니다. "
    "그래프가 급격히 꺾인 뒤 완만해지는 지점을 "
    "묶음 수를 정할 때 참고할 수 있습니다."
)


# --------------------------------------------------
# 3. 선택한 묶음 수로 군집 분석
# --------------------------------------------------

model = KMeans(
    n_clusters=cluster_count,
    random_state=42,
    n_init=10,
)

valid["군집번호"] = model.fit_predict(X_scaled)

# 누적 관객 평균이 높은 군집부터 기호 배정
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

active_symbols = [
    cluster_labels[cluster_id]
    for cluster_id in cluster_order
]

color_map = {
    symbol: COLORS[index]
    for index, symbol in enumerate(active_symbols)
}


# --------------------------------------------------
# 4. 실루엣 점수
# --------------------------------------------------

st.subheader("3. 현재 묶음 수의 실루엣 점수")

try:
    score = silhouette_score(
        X_scaled,
        valid["군집번호"],
    )

    st.write(
        f"현재 묶음 수: **{cluster_count}개**"
        f"  ·  실루엣 점수: **{score:.3f}**"
    )

    st.caption(
        "실루엣 점수는 -1에서 1 사이입니다. "
        "1에 가까울수록 같은 묶음 안의 영화끼리는 비슷하고 "
        "다른 묶음과는 잘 구분되는 경향이 있습니다. "
        "0에 가까우면 묶음 경계가 겹칠 수 있고, "
        "음수이면 일부 영화가 다른 묶음에 더 가까울 수 있습니다."
    )

except Exception as error:
    st.warning(f"실루엣 점수를 계산하지 못했습니다: {error}")


# --------------------------------------------------
# 5. 2차원 산점도
# --------------------------------------------------

st.subheader("4. 2차원 산점도")

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
    category_orders={"유형": active_symbols},
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
    title=f"영화 유형별 2차원 분포 ({cluster_count}개 묶음)",
)

fig2.update_traces(
    marker={"size": 8, "opacity": 0.78}
)
fig2.update_layout(legend_title_text="영화 유형")

st.plotly_chart(fig2, use_container_width=True)


# --------------------------------------------------
# 6. 3차원 산점도
# --------------------------------------------------

st.subheader("5. 3차원 산점도")

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
        category_orders={"유형": active_symbols},
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
        title=f"영화 유형별 3차원 분포 ({cluster_count}개 묶음)",
    )

    fig3.update_traces(
        marker={"size": 3, "opacity": 0.8}
    )
    fig3.update_layout(legend_title_text="영화 유형")

    st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------
# 7. 유형별 편수와 속성 평균
# --------------------------------------------------

st.subheader("6. 유형별 편수와 속성 평균")

summary = valid.groupby("영화 유형").agg(
    편수=("movieNm", "size"),
    스크린수_평균=("first_scrn", "mean"),
    누적관객_평균=("total_audi", "mean"),
    십위권일수_평균=("days_in_top10", "mean"),
    롱런지수_평균=("longrun_index", "mean"),
)

summary = summary.reindex(active_symbols).reset_index()

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
# 8. 유형별 누적 관객 상위 영화 5편
# --------------------------------------------------

st.subheader("7. 유형별 누적 관객 상위 영화 5편")

for symbol in active_symbols:
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
    f"KMeans(n_clusters={cluster_count}, random_state=42, n_init=10)로 "
    "군집화했습니다. 스크린 수와 누적 관객은 상용로그(log10)를 사용하고, "
    "롱런 지수는 누적 관객 ÷ 첫 주 관객으로 계산해 20으로 제한했습니다."
)
