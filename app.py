import streamlit as st
import pandas as pd
import altair as alt

st.set_page_config(page_title="M7 ESG Strategy", layout="wide")

# ---------------- UI 스타일 (안전 버전) ----------------
st.markdown(
    """
<style>
.stApp {
    background-color: #0E370C;
    color: #FFFFFF;
}
h1 {
    font-size: 34px;
    color: #FFFFFF;
}
.card {
    background: #1B4A18;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #4E7A4B;
    color: #FFFFFF;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- 제목 ----------------
st.markdown(
    """
<div class="card">
<h1>ESG Dashboard</h1>
<p>AI 기반 ESG 전략 분석 플랫폼 (Demo Mode)</p>
</div>
""",
    unsafe_allow_html=True,
)

PILLAR_WEIGHTS = {"E": 0.40, "S": 0.30, "G": 0.30}

E_METRIC_WEIGHTS = {
    "탄소/배출 관리": 0.45,
    "에너지 전환": 0.35,
    "자원/폐기물 효율": 0.20,
}

S_METRIC_WEIGHTS = {
    "노동/인권": 0.40,
    "안전/보건": 0.35,
    "다양성/포용": 0.25,
}

G_METRIC_WEIGHTS = {
    "이사회 독립성": 0.40,
    "윤리/컴플라이언스": 0.35,
    "공시 투명성": 0.25,
}

# ---------------- 데모 데이터 ----------------
DEMO = {
    "NVIDIA": {
        "E": {"탄소/배출 관리": 78, "에너지 전환": 74, "자원/폐기물 효율": 72},
        "S": {"노동/인권": 74, "안전/보건": 70, "다양성/포용": 71},
        "G": {"이사회 독립성": 80, "윤리/컴플라이언스": 77, "공시 투명성": 76},
        "investment": 78,
        "fit": 72,
    },
    "Microsoft": {
        "E": {"탄소/배출 관리": 84, "에너지 전환": 81, "자원/폐기물 효율": 79},
        "S": {"노동/인권": 80, "안전/보건": 77, "다양성/포용": 76},
        "G": {"이사회 독립성": 86, "윤리/컴플라이언스": 83, "공시 투명성": 82},
        "investment": 82,
        "fit": 80,
    },
    "Apple": {
        "E": {"탄소/배출 관리": 81, "에너지 전환": 79, "자원/폐기물 효율": 76},
        "S": {"노동/인권": 78, "안전/보건": 74, "다양성/포용": 73},
        "G": {"이사회 독립성": 84, "윤리/컴플라이언스": 81, "공시 투명성": 80},
        "investment": 76,
        "fit": 78,
    },
    "Alphabet": {
        "E": {"탄소/배출 관리": 79, "에너지 전환": 76, "자원/폐기물 효율": 74},
        "S": {"노동/인권": 75, "안전/보건": 72, "다양성/포용": 71},
        "G": {"이사회 독립성": 81, "윤리/컴플라이언스": 78, "공시 투명성": 76},
        "investment": 74,
        "fit": 73,
    },
    "Amazon": {
        "E": {"탄소/배출 관리": 77, "에너지 전환": 74, "자원/폐기물 효율": 70},
        "S": {"노동/인권": 72, "안전/보건": 70, "다양성/포용": 69},
        "G": {"이사회 독립성": 76, "윤리/컴플라이언스": 74, "공시 투명성": 72},
        "investment": 79,
        "fit": 66,
    },
    "Meta": {
        "E": {"탄소/배출 관리": 72, "에너지 전환": 69, "자원/폐기물 효율": 66},
        "S": {"노동/인권": 70, "안전/보건": 67, "다양성/포용": 66},
        "G": {"이사회 독립성": 74, "윤리/컴플라이언스": 71, "공시 투명성": 70},
        "investment": 70,
        "fit": 60,
    },
    "Tesla": {
        "E": {"탄소/배출 관리": 86, "에너지 전환": 82, "자원/폐기물 효율": 78},
        "S": {"노동/인권": 68, "안전/보건": 65, "다양성/포용": 64},
        "G": {"이사회 독립성": 72, "윤리/컴플라이언스": 69, "공시 투명성": 68},
        "investment": 83,
        "fit": 62,
    },
}


def weighted_score(score_map: dict, weight_map: dict) -> int:
    total = 0.0
    for metric, weight in weight_map.items():
        total += score_map.get(metric, 0) * weight
    return round(total)


def to_relative_0_100(series: pd.Series) -> pd.Series:
    if len(series) <= 1:
        return pd.Series([50.0] * len(series), index=series.index)
    ranks = series.rank(method="min", ascending=True)
    return ((ranks - 1) / (len(series) - 1) * 100).round(1)


REQUIRED_COLUMNS = [
    "company",
    "e_carbon",
    "e_energy",
    "e_resource",
    "s_labor",
    "s_safety",
    "s_diversity",
    "g_board",
    "g_ethics",
    "g_disclosure",
    "investment",
    "fit",
]

BASIS_SUMMARY_DEFAULT = {
    "NVIDIA": "100% 전력 재생매칭, Scope2(market) 0 달성, 포장재 재활용 97%, 안전지표/윤리교육 공개",
    "Microsoft": "재생전력 확대와 순환성(서버 재사용/재활용) 성과, 공급망 코드 준수 체계 반영",
    "Apple": "시설 재생전력 100%, 2030 탄소중립 경로, 포장 플라스틱 제거 및 재활용 소재 확대",
    "Alphabet": "글로벌 100% 재생에너지 매칭 유지, 24/7 CFE 확대와 배출 감축 목표 병행",
    "Amazon": "전력 100% 매칭, 플라스틱 포장 감축/대체, 폐기물 매립회피율 및 안전지표 개선",
    "Meta": "전력 100% 재생매칭, Scope1·2 감축목표 및 Scope3 관리, 공급망 RBA 감사 체계",
    "Tesla": "폐기물 전환율, 청정에너지 조달, 에너지 전환 리더십을 반영",
}


@st.cache_data
def load_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


@st.cache_data
def load_basis_csv(file) -> pd.DataFrame:
    df = pd.read_csv(file)
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def build_source_from_dataframe(df: pd.DataFrame) -> dict:
    source = {}
    for _, row in df.iterrows():
        company = str(row["company"]).strip()
        source[company] = {
            "E": {
                "탄소/배출 관리": float(row["e_carbon"]),
                "에너지 전환": float(row["e_energy"]),
                "자원/폐기물 효율": float(row["e_resource"]),
            },
            "S": {
                "노동/인권": float(row["s_labor"]),
                "안전/보건": float(row["s_safety"]),
                "다양성/포용": float(row["s_diversity"]),
            },
            "G": {
                "이사회 독립성": float(row["g_board"]),
                "윤리/컴플라이언스": float(row["g_ethics"]),
                "공시 투명성": float(row["g_disclosure"]),
            },
            "investment": float(row["investment"]),
            "fit": float(row["fit"]),
        }
    return source


def demo_to_template_df() -> pd.DataFrame:
    rows = []
    for company, row in DEMO.items():
        rows.append(
            {
                "company": company,
                "e_carbon": row["E"]["탄소/배출 관리"],
                "e_energy": row["E"]["에너지 전환"],
                "e_resource": row["E"]["자원/폐기물 효율"],
                "s_labor": row["S"]["노동/인권"],
                "s_safety": row["S"]["안전/보건"],
                "s_diversity": row["S"]["다양성/포용"],
                "g_board": row["G"]["이사회 독립성"],
                "g_ethics": row["G"]["윤리/컴플라이언스"],
                "g_disclosure": row["G"]["공시 투명성"],
                "investment": row["investment"],
                "fit": row["fit"],
            }
        )
    return pd.DataFrame(rows)


data_source = DEMO
source_label = "Demo"
csv_error = None
basis_summary_map = BASIS_SUMMARY_DEFAULT.copy()
basis_error = None

# ---------------- 사이드바 ----------------
with st.sidebar:
    st.header("설정")
    uploaded = st.file_uploader("CSV 데이터 업로드", type=["csv"])
    basis_uploaded = st.file_uploader("ESG 핵심요약 CSV(선택)", type=["csv"])
    use_relative_mode = st.toggle("상대 비교 점수 사용", value=True)
    st.caption(
        "필수 컬럼: company, e_carbon, e_energy, e_resource, s_labor, "
        "s_safety, s_diversity, g_board, g_ethics, g_disclosure, investment, fit"
    )
    st.download_button(
        "샘플 CSV 다운로드",
        demo_to_template_df().to_csv(index=False).encode("utf-8"),
        file_name="esg_template.csv",
        mime="text/csv",
    )

if uploaded is not None:
    try:
        df_raw = load_csv(uploaded)
        missing = [col for col in REQUIRED_COLUMNS if col not in df_raw.columns]
        if missing:
            csv_error = f"CSV 필수 컬럼 누락: {', '.join(missing)}"
        elif df_raw.empty:
            csv_error = "CSV에 데이터가 없습니다."
        else:
            cleaned = df_raw[REQUIRED_COLUMNS].copy()
            numeric_cols = [c for c in REQUIRED_COLUMNS if c != "company"]
            for col in numeric_cols:
                cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
            if cleaned[numeric_cols].isna().any().any():
                csv_error = "숫자 컬럼에 비어 있거나 숫자가 아닌 값이 있습니다."
            else:
                for col in numeric_cols:
                    cleaned[col] = cleaned[col].clip(0, 100)
                data_source = build_source_from_dataframe(cleaned)
                source_label = "CSV"
    except Exception as exc:
        csv_error = f"CSV 로드 실패: {exc}"

if basis_uploaded is not None:
    try:
        basis_df = load_basis_csv(basis_uploaded)
        required_basis = {"company", "basis_summary"}
        if not required_basis.issubset(set(basis_df.columns)):
            basis_error = "요약 CSV 필수 컬럼 누락: company, basis_summary"
        else:
            basis_summary_map = {
                str(row["company"]).strip(): str(row["basis_summary"]).strip()
                for _, row in basis_df.iterrows()
                if str(row["company"]).strip()
            }
    except Exception as exc:
        basis_error = f"요약 CSV 로드 실패: {exc}"

company_list = sorted(data_source.keys())

# 버튼 클릭 이후에도 결과가 유지되도록 저장
if "run_analysis" not in st.session_state:
    st.session_state.run_analysis = False

if "selected_companies" not in st.session_state:
    st.session_state.selected_companies = company_list[:2] if company_list else []

# 데이터 소스가 바뀌어 기존 선택이 유효하지 않을 때 보정
st.session_state.selected_companies = [
    c for c in st.session_state.selected_companies if c in company_list
]
if not st.session_state.selected_companies and company_list:
    st.session_state.selected_companies = company_list[:2]

with st.sidebar:
    with st.form("analysis_form"):
        companies = st.multiselect(
            "기업 선택",
            company_list,
            default=st.session_state.selected_companies,
        )
        submitted = st.form_submit_button("분석 실행")

if submitted:
    st.session_state.run_analysis = True
    st.session_state.selected_companies = companies

if csv_error:
    st.warning(f"{csv_error} 데모 데이터로 분석을 계속합니다.")
if basis_error:
    st.warning(f"{basis_error} 기본 ESG 요약을 사용합니다.")

# ---------------- 실행 ----------------
if st.session_state.run_analysis:
    selected = st.session_state.selected_companies

    if not selected:
        st.warning("기업을 1개 이상 선택해 주세요.")
        st.stop()

    data = []
    detail_rows = []
    for c in selected:
        if c not in data_source:
            continue
        row = data_source[c]
        e = weighted_score(row["E"], E_METRIC_WEIGHTS)
        s = weighted_score(row["S"], S_METRIC_WEIGHTS)
        g = weighted_score(row["G"], G_METRIC_WEIGHTS)
        total = round(
            e * PILLAR_WEIGHTS["E"] + s * PILLAR_WEIGHTS["S"] + g * PILLAR_WEIGHTS["G"]
        )
        data.append(
            {
                "company": c,
                "E": e,
                "S": s,
                "G": g,
                "Total": total,
                "investment": row["investment"],
                "fit": row["fit"],
            }
        )
        detail_rows.append(
            {
                "company": c,
                "E 탄소/배출": row["E"]["탄소/배출 관리"],
                "E 에너지": row["E"]["에너지 전환"],
                "E 자원효율": row["E"]["자원/폐기물 효율"],
                "S 노동/인권": row["S"]["노동/인권"],
                "S 안전/보건": row["S"]["안전/보건"],
                "S 다양성": row["S"]["다양성/포용"],
                "G 이사회": row["G"]["이사회 독립성"],
                "G 윤리": row["G"]["윤리/컴플라이언스"],
                "G 공시": row["G"]["공시 투명성"],
            }
        )

    df = pd.DataFrame(data)

    if df.empty:
        st.error("선택된 기업의 데이터를 불러오지 못했습니다.")
        st.stop()

    # 선택된 기업 집합 내 상대 점수(0~100)
    df["Relative_E"] = to_relative_0_100(df["E"])
    df["Relative_S"] = to_relative_0_100(df["S"])
    df["Relative_G"] = to_relative_0_100(df["G"])
    df["Relative_Total"] = to_relative_0_100(df["Total"])
    df["Rank"] = df["Relative_Total"].rank(method="min", ascending=False).astype(int)

    score_col = "Relative_Total" if use_relative_mode else "Total"
    top_company = df.sort_values(score_col, ascending=False).iloc[0]["company"]

    # -------- KPI --------
    st.caption(f"현재 데이터 소스: {source_label}")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("기업 수", len(df))

    with col2:
        if use_relative_mode:
            st.metric("평균 상대 점수", round(df["Relative_Total"].mean(), 1))
        else:
            st.metric("평균 ESG", round(df["Total"].mean()))

    with col3:
        st.metric("최고 기업", top_company)

    st.divider()

    # -------- 랭킹 --------
    st.subheader("🏆 ESG Ranking")
    if use_relative_mode:
        st.caption("현재 선택된 기업들끼리 상대 점수(0~100)로 비교합니다.")
        rank_df = df[
            ["Rank", "company", "Relative_Total", "Relative_E", "Relative_S", "Relative_G"]
        ].sort_values(["Rank", "company"], ascending=[True, True])
        st.dataframe(rank_df, use_container_width=True, hide_index=True)
    else:
        st.caption("절대 점수(Total) 기준 비교입니다.")
        st.dataframe(
            df[
                [
                    "company",
                    "Total",
                    "E",
                    "S",
                    "G",
                    "Relative_Total",
                    "Rank",
                    "investment",
                    "fit",
                ]
            ].sort_values("Total", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    st.divider()

    # -------- 기업별 핵심 내용 --------
    st.subheader("📝 기업별 ESG 핵심 내용")
    summary_df = pd.DataFrame(
        {
            "company": selected,
            "esg_summary": [
                basis_summary_map.get(c, "요약 데이터 없음 (basis CSV에서 추가 가능)")
                for c in selected
            ],
        }
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    st.divider()

    # -------- 스코어링 룰 --------
    st.subheader("📐 Scoring Rule")
    st.caption("총점 = E x 40% + S x 30% + G x 30%")

    rule_col1, rule_col2, rule_col3 = st.columns(3)
    with rule_col1:
        st.markdown("**E 가중치**")
        st.dataframe(
            pd.DataFrame(
                [{"metric": k, "weight": int(v * 100)} for k, v in E_METRIC_WEIGHTS.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    with rule_col2:
        st.markdown("**S 가중치**")
        st.dataframe(
            pd.DataFrame(
                [{"metric": k, "weight": int(v * 100)} for k, v in S_METRIC_WEIGHTS.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )
    with rule_col3:
        st.markdown("**G 가중치**")
        st.dataframe(
            pd.DataFrame(
                [{"metric": k, "weight": int(v * 100)} for k, v in G_METRIC_WEIGHTS.items()]
            ),
            hide_index=True,
            use_container_width=True,
        )

    with st.expander("기업별 세부 지표 보기"):
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True)

    st.divider()

    # -------- 포지셔닝 --------
    st.subheader("📍 ESG Positioning")

    chart = (
        alt.Chart(df)
        .mark_circle(size=150)
        .encode(
            x=alt.X("investment:Q", title="Investment Attractiveness"),
            y=alt.Y("fit:Q", title="Strategic Fit"),
            tooltip=["company:N", "Total:Q", "E:Q", "S:Q", "G:Q"],
        )
    )

    st.altair_chart(chart, use_container_width=True)
else:
    st.info("왼쪽에서 기업 선택 후 분석 실행을 눌러주세요.")
