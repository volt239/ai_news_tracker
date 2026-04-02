# M7 ESG Dashboard (Streamlit)

M7 기업 ESG를 CSV 기반으로 비교/랭킹/시각화하는 Streamlit 앱입니다.

## 핵심 기능

- CSV 업로드 기반 ESG 분석 (`company, e_carbon ... fit`)
- 상대 비교 모드(선택 기업 집합 내 0~100 정규화)
- KPI, 랭킹, 포지셔닝 차트
- 기업별 ESG 핵심 요약 표

## 로컬 실행

```bash
cd /Users/king/Documents/Playground
python3 -m pip install -r requirements.txt
streamlit run app.py
```

## 배포 (Streamlit Community Cloud)

1. 이 폴더를 GitHub 저장소로 push
2. [Streamlit Community Cloud](https://share.streamlit.io/) 로그인
3. `New app` 클릭
4. Repository 선택 후 아래 값 입력
   - **Branch:** `main` (또는 사용 브랜치)
   - **Main file path:** `app.py`
5. Deploy 클릭

배포가 끝나면 `https://<app-name>.streamlit.app` 링크가 생성됩니다.

## 업로드용 샘플 파일

- 점수 CSV: [m7_upload_ready_v2.csv](./m7_upload_ready_v2.csv)
- 요약 CSV(선택): [nvidia_upload_ready_v2_basis.csv](./nvidia_upload_ready_v2_basis.csv)

요약 CSV는 앱에서 `company,basis_summary` 컬럼 형식을 사용합니다.
