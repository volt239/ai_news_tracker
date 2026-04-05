import html
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote_plus
from urllib.parse import urlparse

import feedparser
import matplotlib.pyplot as plt
import pandas as pd
import requests
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from wordcloud import WordCloud


st.set_page_config(
    page_title="AI Tool News Tracker",
    layout="wide",
)

st.title("AI Tool News Tracker")
st.caption("뉴스 기사에서 AI 툴 등장 빈도를 집계하고 급상승 툴을 확인합니다")

DEFAULT_TOOLS = [
    "ChatGPT",
    "Gemini",
    "Claude",
    "Copilot",
    "Perplexity",
    "Midjourney",
    "Runway",
    "Grok",
    "Notion AI",
    "Jasper",
    "Cursor",
    "Sora",
]

# 툴명 변형(별칭) 매칭. 필요 시 사이드바 입력값에 맞춰 확장 가능.
TOOL_ALIASES = {
    "ChatGPT": ["ChatGPT", "GPT-4o", "GPT-4.1", "OpenAI GPT"],
    "Gemini": ["Gemini", "Google Gemini", "Gemini Advanced"],
    "Claude": ["Claude", "Anthropic Claude"],
    "Copilot": ["Copilot", "GitHub Copilot", "Microsoft Copilot"],
    "Perplexity": ["Perplexity", "Perplexity AI"],
    "Midjourney": ["Midjourney"],
    "Runway": ["Runway", "RunwayML"],
    "Grok": ["Grok", "xAI Grok"],
    "Notion AI": ["Notion AI", "Notion"],
    "Jasper": ["Jasper", "Jasper AI"],
    "Cursor": ["Cursor", "Cursor AI"],
    "Sora": ["Sora", "OpenAI Sora"],
}

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)

# 언론사 중심 필터를 위한 비기사/블로그 플랫폼 제외 규칙.
EXCLUDED_SOURCE_KEYWORDS = {
    "브런치",
    "brunch",
    "tistory",
    "티스토리",
    "naver blog",
    "네이버 블로그",
    "velog",
    "medium",
    "substack",
}

EXCLUDED_SOURCE_DOMAINS = {
    "brunch.co.kr",
    "tistory.com",
    "blog.naver.com",
    "velog.io",
    "medium.com",
    "substack.com",
}

DEFAULT_LLM_MODEL = "gpt-5-mini"


def normalize_tool_input(text: str):
    tools = []
    for raw in re.split(r"[\n,]", text):
        cleaned = raw.strip()
        if cleaned and cleaned not in tools:
            tools.append(cleaned)
    return tools


def build_google_news_rss_url(query: str, hl: str, gl: str, ceid: str) -> str:
    encoded = quote_plus(query)
    return f"https://news.google.com/rss/search?q={encoded}&hl={hl}&gl={gl}&ceid={ceid}"


def safe_get_text(entry, key: str) -> str:
    value = entry.get(key, "")
    if not value:
        return ""
    return html.unescape(re.sub(r"<[^>]+>", " ", str(value))).strip()


def parse_entry_datetime(entry):
    parsed = None
    if getattr(entry, "published_parsed", None):
        parsed = entry.published_parsed
    elif getattr(entry, "updated_parsed", None):
        parsed = entry.updated_parsed

    if parsed is None:
        return None

    return datetime(*parsed[:6], tzinfo=timezone.utc)


def _build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.6,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update({"User-Agent": USER_AGENT})
    return session


@st.cache_data(ttl=600, show_spinner=False)
def fetch_rss_entries(url: str, timeout: int = 15):
    session = _build_session()
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    return feedparser.parse(response.content).entries


def alias_list_for_tool(tool: str):
    return TOOL_ALIASES.get(tool, [tool])


def compile_patterns(tool: str):
    patterns = []
    for alias in alias_list_for_tool(tool):
        escaped = re.escape(alias)
        # 문자 경계를 강제해 부분 문자열 오탐을 줄임.
        pattern = rf"(?<!\w){escaped}(?!\w)"
        patterns.append(re.compile(pattern, flags=re.IGNORECASE))
    return patterns


def count_mentions(text: str, compiled_patterns) -> int:
    return sum(len(pattern.findall(text)) for pattern in compiled_patterns)


def is_allowed_news_source(source: str, link: str) -> bool:
    source_norm = (source or "").strip().lower()
    if not source_norm:
        return False

    for keyword in EXCLUDED_SOURCE_KEYWORDS:
        if keyword in source_norm:
            return False

    domain = ""
    if link:
        try:
            domain = (urlparse(link).netloc or "").lower()
        except Exception:
            domain = ""
    if domain.startswith("www."):
        domain = domain[4:]
    if any(domain == blocked or domain.endswith(f".{blocked}") for blocked in EXCLUDED_SOURCE_DOMAINS):
        return False

    return True


def analyze_news_window(tools, window_start, window_end, max_articles_per_tool, hl, gl, ceid):
    article_rows = []

    def fetch_tool(tool):
        rss_url = build_google_news_rss_url(f'"{tool}"', hl=hl, gl=gl, ceid=ceid)
        entries = fetch_rss_entries(rss_url)
        return tool, entries

    results = []
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(tools)))) as pool:
        future_map = {pool.submit(fetch_tool, tool): tool for tool in tools}
        for fut in as_completed(future_map):
            tool = future_map[fut]
            try:
                t, entries = fut.result()
                results.append((t, entries, None))
            except Exception as exc:
                results.append((tool, [], exc))

    for tool, entries, error in results:
        if error:
            article_rows.append(
                {
                    "tool": tool,
                    "title": "[RSS 수집 실패]",
                    "source": "-",
                    "published_at": None,
                    "link": "",
                    "matched_mentions": 0,
                    "raw_text": str(error),
                }
            )
            continue

        compiled_patterns = compile_patterns(tool)
        seen_links_for_tool = set()
        kept = 0

        for entry in entries:
            if kept >= max_articles_per_tool:
                break

            published_at = parse_entry_datetime(entry)
            # 날짜 없는 항목은 구간 비교의 정확도를 위해 제외.
            if published_at is None:
                continue
            if published_at < window_start or published_at >= window_end:
                continue

            title = safe_get_text(entry, "title")
            summary = safe_get_text(entry, "summary") or safe_get_text(entry, "description")
            source = ""
            if entry.get("source"):
                source_obj = entry.get("source")
                if isinstance(source_obj, dict):
                    source = source_obj.get("title", "")
                else:
                    source = str(source_obj)

            link = entry.get("link", "")
            if not is_allowed_news_source(source=source, link=link):
                continue

            unique_key = link or title
            if unique_key in seen_links_for_tool:
                continue
            seen_links_for_tool.add(unique_key)

            raw_text = f"{title} {summary}".strip()
            mentions = count_mentions(raw_text, compiled_patterns)

            article_rows.append(
                {
                    "tool": tool,
                    "title": title,
                    "source": source,
                    "published_at": published_at,
                    "link": link,
                    "matched_mentions": mentions,
                    "raw_text": raw_text,
                }
            )
            kept += 1

    articles_df = pd.DataFrame(article_rows)
    if articles_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    valid_articles = articles_df[articles_df["title"] != "[RSS 수집 실패]"].copy()

    if not valid_articles.empty:
        summary_df = (
            valid_articles.groupby("tool", as_index=False)
            .agg(
                article_count=("title", "count"),
                mention_count=("matched_mentions", "sum"),
                latest_seen=("published_at", "max"),
            )
            .sort_values(["mention_count", "article_count"], ascending=False)
        )
    else:
        summary_df = pd.DataFrame(columns=["tool", "article_count", "mention_count", "latest_seen"])

    if "published_at" in articles_df.columns:
        articles_df["published_at_local"] = pd.to_datetime(articles_df["published_at"], utc=True, errors="coerce")
        articles_df["published_at_local"] = articles_df["published_at_local"].dt.tz_convert("Asia/Seoul")

    return summary_df, articles_df.sort_values("published_at", ascending=False, na_position="last")


def build_rising_table(recent_summary_df, base_summary_df):
    recent = (
        recent_summary_df[["tool", "mention_count"]]
        .rename(columns={"mention_count": "recent_mentions"})
        if not recent_summary_df.empty
        else pd.DataFrame(columns=["tool", "recent_mentions"])
    )
    base = (
        base_summary_df[["tool", "mention_count"]]
        .rename(columns={"mention_count": "base_mentions"})
        if not base_summary_df.empty
        else pd.DataFrame(columns=["tool", "base_mentions"])
    )
    merged = pd.merge(base, recent, on="tool", how="outer").fillna(0)
    if merged.empty:
        return merged

    merged["base_mentions"] = merged["base_mentions"].astype(int)
    merged["recent_mentions"] = merged["recent_mentions"].astype(int)
    merged["rising_score"] = ((merged["recent_mentions"] + 1) / (merged["base_mentions"] + 1)).round(2)
    merged["delta"] = merged["recent_mentions"] - merged["base_mentions"]
    return merged.sort_values(["rising_score", "recent_mentions"], ascending=False)


def make_wordcloud_text(articles_df):
    if articles_df.empty or "raw_text" not in articles_df.columns:
        return ""
    return " ".join(articles_df["raw_text"].dropna().astype(str).tolist())


def resolve_korean_font_path():
    # 로컬/클라우드 환경에서 자주 쓰는 한글 폰트 후보 경로.
    candidates = [
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",  # macOS
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",  # Ubuntu + fonts-nanum
        "/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Ubuntu + fonts-noto-cjk
        "/usr/share/fonts/opentype/noto/NotoSansCJKkr-Regular.otf",
        str(Path(__file__).resolve().parent / "fonts" / "NanumGothic.ttf"),  # 번들 폰트(선택)
    ]
    for path in candidates:
        if Path(path).exists():
            return path
    return None


def _extract_text_from_responses_api(resp_json: dict) -> str:
    if isinstance(resp_json.get("output_text"), str) and resp_json["output_text"].strip():
        return resp_json["output_text"].strip()

    chunks = []
    for item in resp_json.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                text = content.get("text", "")
                if text:
                    chunks.append(text)
    return "\n".join(chunks).strip()


def _safe_json_loads(text: str):
    if not text:
        return None
    raw = text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
        if not match:
            return None
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None


def call_openai_responses_json(api_key: str, model: str, system_prompt: str, user_payload: dict):
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {
                "role": "user",
                "content": [{"type": "input_text", "text": json.dumps(user_payload, ensure_ascii=False)}],
            },
        ],
        "text": {"format": {"type": "text"}},
    }
    response = requests.post(url, headers=headers, json=body, timeout=90)
    response.raise_for_status()
    resp_json = response.json()
    raw_text = _extract_text_from_responses_api(resp_json)
    parsed = _safe_json_loads(raw_text)
    if parsed is None:
        raise ValueError("LLM 응답을 JSON으로 해석하지 못했습니다.")
    return parsed


def build_llm_context_payload(articles_df: pd.DataFrame, summary_df: pd.DataFrame, rising_df: pd.DataFrame, base_summary_df: pd.DataFrame, max_articles: int):
    recent_articles = (
        articles_df[articles_df["title"] != "[RSS 수집 실패]"]
        .head(max_articles)
        .copy()
    )
    for col in ["published_at_local", "published_at"]:
        if col in recent_articles.columns:
            recent_articles[col] = recent_articles[col].astype(str)

    top_headlines_by_tool = {}
    if not recent_articles.empty:
        for tool, group in recent_articles.groupby("tool"):
            top_headlines_by_tool[tool] = group["title"].dropna().astype(str).head(5).tolist()

    return {
        "generated_at_kst": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "summary_table": summary_df.to_dict(orient="records"),
        "base_summary_table": base_summary_df.to_dict(orient="records"),
        "rising_table": rising_df.to_dict(orient="records"),
        "top_headlines_by_tool": top_headlines_by_tool,
        "articles_sample": recent_articles[
            ["tool", "title", "source", "published_at_local", "matched_mentions", "link"]
        ].to_dict(orient="records")
        if not recent_articles.empty
        else [],
    }


def run_llm_news_analysis(api_key: str, model: str, context_payload: dict):
    system_prompt = (
        "너는 AI 산업 뉴스 애널리스트다. 입력 JSON만 근거로 분석하라. "
        "과장 없이 팩트 중심으로 작성하고, 결과는 반드시 JSON으로만 반환하라."
    )
    user_payload = {
        "task": [
            "각 tool에 대해 한줄 요약(summary), 핵심포인트 3개(key_points), 감성(sentiment: positive|neutral|negative), 감성근거(sentiment_reason) 작성",
            "rising_table을 기반으로 최근 급상승 이유를 tool별로 작성(why_rising + evidence_headlines)",
            "기사들을 이슈 클러스터로 묶어 cluster_name, description, tools, headline_examples 작성",
            "전체 시사점 overall_takeaways 3~5개 작성",
        ],
        "output_schema": {
            "tool_summaries": [
                {
                    "tool": "string",
                    "summary": "string",
                    "key_points": ["string", "string", "string"],
                    "sentiment": "positive|neutral|negative",
                    "sentiment_reason": "string",
                }
            ],
            "rising_reasons": [
                {
                    "tool": "string",
                    "why_rising": "string",
                    "evidence_headlines": ["string"],
                }
            ],
            "issue_clusters": [
                {
                    "cluster_name": "string",
                    "description": "string",
                    "tools": ["string"],
                    "headline_examples": ["string"],
                }
            ],
            "overall_takeaways": ["string"],
        },
        "context": context_payload,
    }
    return call_openai_responses_json(api_key=api_key, model=model, system_prompt=system_prompt, user_payload=user_payload)


def run_llm_qa(api_key: str, model: str, context_payload: dict, question: str):
    system_prompt = (
        "너는 한국어 데이터 분석 어시스턴트다. 입력 JSON 데이터만 근거로 답하고, "
        "데이터에 없으면 없다고 말하라."
    )
    user_payload = {
        "question": question,
        "required_output": {
            "answer": "string",
            "evidence_headlines": ["string"],
            "confidence": "low|medium|high",
        },
        "context": context_payload,
    }
    return call_openai_responses_json(api_key=api_key, model=model, system_prompt=system_prompt, user_payload=user_payload)


with st.sidebar:
    st.header("설정")
    tool_text = st.text_area(
        "추적할 AI 툴 목록",
        value="\n".join(DEFAULT_TOOLS),
        height=250,
        help="줄바꿈 또는 쉼표로 여러 툴을 입력하세요.",
    )
    hours_back = st.selectbox("최근 분석 시간(시간)", [3, 6, 12, 24, 48, 168], index=2)
    compare_hours = st.selectbox("직전 비교 시간(시간)", [3, 6, 12, 24, 48, 168], index=3)
    max_articles_per_tool = st.slider("툴당 최대 기사 수", min_value=5, max_value=60, value=20, step=5)

    st.markdown("---")
    hl = st.selectbox("언어(hl)", ["ko", "en-US", "en-GB"], index=0)
    gl = st.selectbox("국가(gl)", ["KR", "US", "GB", "JP"], index=0)
    ceid = st.selectbox("에디션(ceid)", ["KR:ko", "US:en", "GB:en", "JP:ja"], index=0)

    st.markdown("---")
    st.subheader("LLM 분석")
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY", "")
    except Exception:
        secret_key = ""
    env_key = os.getenv("OPENAI_API_KEY", "")
    default_api_key = secret_key or env_key
    openai_api_key = st.text_input("OpenAI API Key", value=default_api_key, type="password")
    llm_model = st.text_input("모델", value=DEFAULT_LLM_MODEL)
    llm_max_articles = st.slider("LLM 분석용 기사 샘플 수", min_value=20, max_value=200, value=80, step=20)

    run_analysis = st.button("분석 시작", type="primary", use_container_width=True)


tools = normalize_tool_input(tool_text)
if not tools:
    st.warning("최소 1개 이상의 AI 툴 이름을 입력하세요.")
    st.stop()

for key in [
    "recent_summary_df",
    "recent_articles_df",
    "base_summary_df",
    "base_articles_df",
    "last_run_info",
    "llm_analysis",
    "llm_analysis_error",
    "llm_qa_result",
    "llm_qa_error",
]:
    if key not in st.session_state:
        st.session_state[key] = pd.DataFrame() if "info" not in key else None

if run_analysis:
    now = datetime.now(timezone.utc)
    recent_start = now - timedelta(hours=hours_back)
    base_end = recent_start
    base_start = base_end - timedelta(hours=compare_hours)

    with st.spinner("뉴스를 수집하고 빈도를 계산하는 중입니다..."):
        recent_summary_df, recent_articles_df = analyze_news_window(
            tools=tools,
            window_start=recent_start,
            window_end=now,
            max_articles_per_tool=max_articles_per_tool,
            hl=hl,
            gl=gl,
            ceid=ceid,
        )
        base_summary_df, base_articles_df = analyze_news_window(
            tools=tools,
            window_start=base_start,
            window_end=base_end,
            max_articles_per_tool=max_articles_per_tool,
            hl=hl,
            gl=gl,
            ceid=ceid,
        )

        st.session_state.recent_summary_df = recent_summary_df
        st.session_state.recent_articles_df = recent_articles_df
        st.session_state.base_summary_df = base_summary_df
        st.session_state.base_articles_df = base_articles_df
        st.session_state.llm_analysis = None
        st.session_state.llm_analysis_error = None
        st.session_state.llm_qa_result = None
        st.session_state.llm_qa_error = None
        st.session_state.last_run_info = {
            "ran_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "recent_hours": hours_back,
            "base_hours": compare_hours,
            "tool_count": len(tools),
            "recent_window": f"{recent_start:%Y-%m-%d %H:%M} ~ {now:%Y-%m-%d %H:%M} UTC",
            "base_window": f"{base_start:%Y-%m-%d %H:%M} ~ {base_end:%Y-%m-%d %H:%M} UTC",
        }

summary_df = st.session_state.recent_summary_df
articles_df = st.session_state.recent_articles_df
base_summary_df = st.session_state.base_summary_df
last_run_info = st.session_state.last_run_info

if last_run_info:
    st.success(
        f"마지막 분석: {last_run_info['ran_at']} | 최근 {last_run_info['recent_hours']}시간 vs 직전 {last_run_info['base_hours']}시간 | 추적 툴 {last_run_info['tool_count']}개"
    )
else:
    st.info("왼쪽 사이드바에서 설정을 고른 뒤 '분석 시작'을 눌러주세요.")

st.markdown(
    """
### 📌 분석 목적
뉴스에서 반복적으로 등장하는 AI 툴을 분석해 현재 시장에서 주목받는 도구를 파악하고,
우선 검토할 툴의 순서를 정하는 데 도움을 주는 탐색용 앱입니다.
"""
)

if summary_df.empty and articles_df.empty:
    st.stop()

valid_articles_df = articles_df[articles_df["title"] != "[RSS 수집 실패]"].copy() if not articles_df.empty else pd.DataFrame()

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("추적 AI 툴", len(tools))
with col2:
    st.metric("수집 기사 수(최근)", int(len(valid_articles_df)))
with col3:
    st.metric("총 언급 수(최근)", int(summary_df["mention_count"].sum()) if "mention_count" in summary_df else 0)
with col4:
    top_tool = summary_df.iloc[0]["tool"] if not summary_df.empty else "-"
    st.metric("가장 많이 언급된 툴", top_tool)

st.subheader("AI 툴별 언급 빈도 표 (최근 구간)")
display_summary = summary_df.copy()
if "latest_seen" in display_summary.columns:
    display_summary["latest_seen"] = pd.to_datetime(display_summary["latest_seen"], utc=True, errors="coerce").dt.tz_convert("Asia/Seoul")
    display_summary["latest_seen"] = display_summary["latest_seen"].dt.strftime("%Y-%m-%d %H:%M")
st.dataframe(display_summary, use_container_width=True, hide_index=True)

if not summary_df.empty:
    chart_df = summary_df[["tool", "mention_count"]].sort_values("mention_count", ascending=False).set_index("tool")
    st.subheader("AI 툴별 언급 수 차트")
    st.bar_chart(chart_df)

rising_df = build_rising_table(summary_df, base_summary_df)
st.subheader("🔥 급상승 AI 툴 비교")
st.caption(f"최근 {hours_back}시간과 그 직전 {compare_hours}시간을 분리 비교합니다.")
st.dataframe(rising_df, use_container_width=True, hide_index=True)

if not summary_df.empty:
    st.subheader("🏆 TOP 3 AI 툴")
    top3 = summary_df.head(3)[["tool", "mention_count", "article_count"]].reset_index(drop=True)
    st.dataframe(top3, use_container_width=True, hide_index=True)

    for tool in summary_df.head(3)["tool"].tolist():
        st.markdown(f"### {tool} 관련 주요 기사")
        tool_articles = articles_df[(articles_df["tool"] == tool) & (articles_df["title"] != "[RSS 수집 실패]")].head(3)
        if tool_articles.empty:
            st.write("관련 기사가 없습니다.")
        else:
            for _, row in tool_articles.iterrows():
                link = row.get("link", "")
                if link:
                    st.markdown(f"- [{row['title']}]({link}) | {row['source']}")
                else:
                    st.markdown(f"- **{row['title']}** | {row['source']}")

st.subheader("기사 상세 목록")
article_display_cols = [
    "tool",
    "title",
    "source",
    "published_at_local",
    "matched_mentions",
    "link",
]
article_display = articles_df.copy()
if "published_at_local" in article_display.columns:
    article_display["published_at_local"] = article_display["published_at_local"].dt.strftime("%Y-%m-%d %H:%M")

st.dataframe(
    article_display[article_display_cols] if not article_display.empty else article_display,
    use_container_width=True,
    hide_index=True,
)

st.subheader("LLM 인사이트")
st.caption("요약/급상승 원인/이슈 클러스터/감성 분석과 질문형 분석을 제공합니다.")

context_payload = build_llm_context_payload(
    articles_df=articles_df,
    summary_df=summary_df,
    rising_df=rising_df,
    base_summary_df=base_summary_df,
    max_articles=llm_max_articles,
)

llm_col1, llm_col2 = st.columns([1, 1])
with llm_col1:
    run_llm_analysis = st.button("LLM 종합 분석 실행", use_container_width=True)
with llm_col2:
    clear_llm_cache = st.button("LLM 결과 초기화", use_container_width=True)

if clear_llm_cache:
    st.session_state.llm_analysis = None
    st.session_state.llm_analysis_error = None
    st.session_state.llm_qa_result = None
    st.session_state.llm_qa_error = None

if run_llm_analysis:
    if not openai_api_key.strip():
        st.session_state.llm_analysis_error = "OpenAI API Key를 입력해 주세요."
        st.session_state.llm_analysis = None
    else:
        with st.spinner("LLM이 뉴스를 해석하는 중입니다..."):
            try:
                llm_analysis = run_llm_news_analysis(
                    api_key=openai_api_key.strip(),
                    model=llm_model.strip() or DEFAULT_LLM_MODEL,
                    context_payload=context_payload,
                )
                st.session_state.llm_analysis = llm_analysis
                st.session_state.llm_analysis_error = None
            except Exception as exc:
                st.session_state.llm_analysis = None
                st.session_state.llm_analysis_error = f"LLM 분석 실패: {exc}"

if st.session_state.llm_analysis_error:
    st.warning(st.session_state.llm_analysis_error)

llm_analysis = st.session_state.llm_analysis
if isinstance(llm_analysis, dict):
    tool_summaries = llm_analysis.get("tool_summaries", [])
    rising_reasons = llm_analysis.get("rising_reasons", [])
    issue_clusters = llm_analysis.get("issue_clusters", [])
    overall_takeaways = llm_analysis.get("overall_takeaways", [])

    if tool_summaries:
        st.markdown("#### 툴별 요약 + 감성")
        summary_rows = []
        for item in tool_summaries:
            summary_rows.append(
                {
                    "tool": item.get("tool", ""),
                    "summary": item.get("summary", ""),
                    "sentiment": item.get("sentiment", ""),
                    "sentiment_reason": item.get("sentiment_reason", ""),
                    "key_points": " | ".join(item.get("key_points", [])),
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if rising_reasons:
        st.markdown("#### 급상승 원인")
        for item in rising_reasons:
            st.markdown(f"- **{item.get('tool', '-')}:** {item.get('why_rising', '')}")
            evidence = item.get("evidence_headlines", [])
            if evidence:
                st.caption("근거 헤드라인: " + " / ".join(evidence[:3]))

    if issue_clusters:
        st.markdown("#### 이슈 클러스터")
        cluster_rows = []
        for item in issue_clusters:
            cluster_rows.append(
                {
                    "cluster_name": item.get("cluster_name", ""),
                    "description": item.get("description", ""),
                    "tools": ", ".join(item.get("tools", [])),
                    "headline_examples": " | ".join(item.get("headline_examples", [])[:3]),
                }
            )
        st.dataframe(pd.DataFrame(cluster_rows), use_container_width=True, hide_index=True)

    if overall_takeaways:
        st.markdown("#### 전체 시사점")
        for takeaway in overall_takeaways:
            st.markdown(f"- {takeaway}")

st.markdown("#### 질문형 분석 (한국어)")
user_question = st.text_input("질문 입력", placeholder="예: 지난 24시간에 Copilot 이슈는 긍정 뉴스가 많아?")
ask_llm = st.button("질문 분석 실행", use_container_width=True)

if ask_llm:
    if not openai_api_key.strip():
        st.session_state.llm_qa_error = "OpenAI API Key를 입력해 주세요."
        st.session_state.llm_qa_result = None
    elif not user_question.strip():
        st.session_state.llm_qa_error = "질문을 입력해 주세요."
        st.session_state.llm_qa_result = None
    else:
        with st.spinner("질문에 답변하는 중입니다..."):
            try:
                qa_result = run_llm_qa(
                    api_key=openai_api_key.strip(),
                    model=llm_model.strip() or DEFAULT_LLM_MODEL,
                    context_payload=context_payload,
                    question=user_question.strip(),
                )
                st.session_state.llm_qa_result = qa_result
                st.session_state.llm_qa_error = None
            except Exception as exc:
                st.session_state.llm_qa_result = None
                st.session_state.llm_qa_error = f"질문 분석 실패: {exc}"

if st.session_state.llm_qa_error:
    st.warning(st.session_state.llm_qa_error)

if isinstance(st.session_state.llm_qa_result, dict):
    st.info(st.session_state.llm_qa_result.get("answer", "답변이 없습니다."))
    evidence_headlines = st.session_state.llm_qa_result.get("evidence_headlines", [])
    if evidence_headlines:
        st.caption("근거 기사: " + " / ".join(evidence_headlines[:5]))
    confidence = st.session_state.llm_qa_result.get("confidence", "")
    if confidence:
        st.caption(f"신뢰도: {confidence}")

st.subheader("☁️ 뉴스 워드클라우드")
wc_text = make_wordcloud_text(valid_articles_df)
if wc_text.strip():
    font_path = resolve_korean_font_path()
    if font_path:
        wc = WordCloud(width=1200, height=500, background_color="white", font_path=font_path).generate(wc_text)
    else:
        st.warning("한글 폰트를 찾지 못해 워드클라우드 한글이 깨질 수 있습니다. 배포 환경에 Nanum/Noto 폰트를 설치해 주세요.")
        wc = WordCloud(width=1200, height=500, background_color="white").generate(wc_text)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    st.pyplot(fig)
else:
    st.info("워드클라우드를 만들 데이터가 없습니다.")

csv_bytes = display_summary.to_csv(index=False).encode("utf-8-sig")
st.download_button(
    "빈도표 CSV 다운로드",
    data=csv_bytes,
    file_name="ai_tool_news_frequency.csv",
    mime="text/csv",
)

with st.expander("해석 가이드"):
    st.markdown(
        """
- **article_count**: 최근 선택 구간에서 수집된 기사 수
- **mention_count**: 제목/요약에서 툴명(별칭 포함)이 등장한 횟수 합계
- **rising_score**: (최근 언급+1)/(직전 구간 언급+1)
- 이 앱은 감정 분석보다 빈도와 추세 비교에 초점을 둔 탐색 도구입니다.
        """
    )

with st.expander("주의사항"):
    st.markdown(
        """
1. 뉴스 언급량은 실제 제품 성능과 동일하지 않을 수 있습니다.
2. 제목/요약 기반이라 본문 전체 분석보다 단순합니다.
3. RSS 제공 방식이 바뀌면 수집 결과가 달라질 수 있습니다.
4. 동일 이벤트가 여러 매체에 반복 보도될 수 있습니다.
        """
    )
