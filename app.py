import re
from math import exp
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ------------------------------------------------------------
# 310관 커피니 동적가격제 시뮬레이터
# ------------------------------------------------------------
# 새 모델 방향:
# 기존의 임의 가중치 모델 대신, 설문조사 데이터를 기반으로
# Logistic Regression 모델을 학습해 아메리카노 구매확률을 예측한다.
#
# 모델 입력값:
# 요일, 시간대, 시험기간 여부, 불쾌지수 단계, 가격
#
# 모델 출력값:
# 해당 조건과 가격에서 아메리카노를 구매할 확률
#
# 추천 방식:
# 후보 가격별 예상매출 = 가격 x 구매확률 x 예상 고객 수
# 예상매출이 가장 높은 가격을 추천한다.
# ------------------------------------------------------------


st.set_page_config(
    page_title="310관 커피니 동적가격제 시뮬레이터",
    page_icon="☕",
    layout="wide",
)


# ------------------------------------------------------------
# 1. 모델과 설문 변환에 사용할 기준값
# ------------------------------------------------------------

MENU_NAME = "아메리카노"
PRICE_CANDIDATES = [1600, 1800, 2000, 2200, 2400, 2600, 2800]
SURVEY_DATA_CANDIDATES = [
    Path("survey_result.csv"),
    Path("survey_result.xlsx"),
    Path("survey_result.xls"),
]
SAMPLE_DATA_PATH = Path("sample_data.csv")
AMERICANO_IMAGE_PATH = Path("assets") / "americano.svg"

DAYS = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]
TIME_SLOTS = ["8시~11시", "11시~14시", "14시~17시", "17시~19시", "19시 이후"]
EXAM_PERIODS = ["시험기간", "비시험기간"]
DI_LEVELS = ["낮음", "보통", "높음", "매우 높음"]

FEATURE_COLUMNS = ["요일", "시간대", "시험기간 여부", "불쾌지수 단계", "가격"]
TARGET_COLUMN = "구매여부"

# Google Form에서 상황별 질문이 열로 들어오는 경우를 long format으로 바꾸기 위해
# 상황 1~20의 조건을 코드 내부에 정의한다.
SITUATION_CONDITIONS = pd.DataFrame(
    [
        {"상황": 1, "요일": "월요일", "시간대": "8시~11시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "낮음", "가격": 1600},
        {"상황": 2, "요일": "월요일", "시간대": "11시~14시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "보통", "가격": 1800},
        {"상황": 3, "요일": "월요일", "시간대": "11시~14시", "시험기간 여부": "시험기간", "불쾌지수 단계": "높음", "가격": 2000},
        {"상황": 4, "요일": "화요일", "시간대": "8시~11시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "낮음", "가격": 2200},
        {"상황": 5, "요일": "화요일", "시간대": "14시~17시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "보통", "가격": 2400},
        {"상황": 6, "요일": "화요일", "시간대": "17시~19시", "시험기간 여부": "시험기간", "불쾌지수 단계": "높음", "가격": 2600},
        {"상황": 7, "요일": "수요일", "시간대": "11시~14시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "매우 높음", "가격": 2800},
        {"상황": 8, "요일": "수요일", "시간대": "14시~17시", "시험기간 여부": "시험기간", "불쾌지수 단계": "높음", "가격": 1600},
        {"상황": 9, "요일": "목요일", "시간대": "8시~11시", "시험기간 여부": "시험기간", "불쾌지수 단계": "보통", "가격": 1800},
        {"상황": 10, "요일": "목요일", "시간대": "19시 이후", "시험기간 여부": "비시험기간", "불쾌지수 단계": "낮음", "가격": 2000},
        {"상황": 11, "요일": "금요일", "시간대": "11시~14시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "높음", "가격": 2200},
        {"상황": 12, "요일": "금요일", "시간대": "14시~17시", "시험기간 여부": "시험기간", "불쾌지수 단계": "매우 높음", "가격": 2400},
        {"상황": 13, "요일": "토요일", "시간대": "11시~14시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "보통", "가격": 2600},
        {"상황": 14, "요일": "토요일", "시간대": "17시~19시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "낮음", "가격": 2800},
        {"상황": 15, "요일": "일요일", "시간대": "11시~14시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "낮음", "가격": 1600},
        {"상황": 16, "요일": "일요일", "시간대": "19시 이후", "시험기간 여부": "시험기간", "불쾌지수 단계": "보통", "가격": 1800},
        {"상황": 17, "요일": "수요일", "시간대": "8시~11시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "높음", "가격": 2000},
        {"상황": 18, "요일": "목요일", "시간대": "14시~17시", "시험기간 여부": "비시험기간", "불쾌지수 단계": "매우 높음", "가격": 2200},
        {"상황": 19, "요일": "금요일", "시간대": "17시~19시", "시험기간 여부": "시험기간", "불쾌지수 단계": "보통", "가격": 2400},
        {"상황": 20, "요일": "월요일", "시간대": "19시 이후", "시험기간 여부": "비시험기간", "불쾌지수 단계": "매우 높음", "가격": 2600},
    ]
)


# ------------------------------------------------------------
# 2. 데이터 전처리 함수
# ------------------------------------------------------------

def is_private_column(column_name: str) -> bool:
    """이메일, 전화번호, 이름 등 개인정보로 보이는 열을 자동 제외한다."""
    normalized = str(column_name).strip().lower()
    private_patterns = [
        "email",
        "e-mail",
        "메일",
        "이메일",
        "phone",
        "tel",
        "전화",
        "연락처",
        "휴대폰",
        "이름",
        "성명",
        "name",
    ]
    return any(pattern in normalized for pattern in private_patterns)


def remove_private_columns(df: pd.DataFrame) -> pd.DataFrame:
    """설문 원본에서 개인정보 가능성이 있는 열을 제거한다."""
    safe_columns = [col for col in df.columns if not is_private_column(col)]
    return df[safe_columns].copy()


def normalize_purchase_answer(value) -> int | None:
    """설문 응답값을 구매 여부 1/0으로 변환한다."""
    if pd.isna(value):
        return None

    # Excel/CSV에서 구매여부가 1, 0, 1.0, 0.0 같은 숫자로 들어오는 경우도 처리한다.
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if value == 1:
            return 1
        if value == 0:
            return 0

    text = str(value).strip().lower()

    positive_answers = {
        "구매하겠다",
        "구매 하겠다",
        "구매",
        "예",
        "yes",
        "y",
        "o",
        "○",
        "◯",
        "1",
        "1.0",
        "true",
    }
    negative_answers = {
        "구매하지 않겠다",
        "구매하지않겠다",
        "구매 안 하겠다",
        "구매안하겠다",
        "비구매",
        "아니오",
        "아니요",
        "no",
        "n",
        "x",
        "×",
        "0",
        "0.0",
        "false",
    }

    if text in positive_answers:
        return 1
    if text in negative_answers:
        return 0
    return None


def find_situation_column(columns: list[str], situation_number: int) -> str | None:
    """상황 번호에 해당하는 Google Form 질문 열을 찾는다.

    예: '상황 1', '상황1', '| 1 | 월요일 오후 12시 ...' 등
    """
    patterns = [
        re.compile(rf"상황\s*{situation_number}(?!\d)"),
        re.compile(rf"^\s*\|\s*{situation_number}\s*\|"),
    ]
    for column in columns:
        if any(pattern.search(str(column)) for pattern in patterns):
            return column
    return None


def parse_time_slot_from_text(text: str) -> str:
    """실제 설문 문항의 시간 표현을 모델 시간대 범주로 변환한다."""
    if any(keyword in text for keyword in ["오전 8", "오전 9", "오전 10", "오전 11"]):
        return "8시~11시"
    if any(keyword in text for keyword in ["오후 12", "오후 1"]):
        return "11시~14시"
    if any(keyword in text for keyword in ["오후 2", "오후 3", "오후 4"]):
        return "14시~17시"
    if any(keyword in text for keyword in ["오후 5", "오후 6"]):
        return "17시~19시"
    return "19시 이후"


def parse_exam_period_from_text(text: str) -> str:
    """중간기간/기말기간은 시험기간, 시험기간X는 비시험기간으로 변환한다."""
    compact_text = text.replace(" ", "")
    if "시험기간X" in compact_text:
        return "비시험기간"
    if "중간" in text or "기말" in text or "시험기간" in text:
        return "시험기간"
    return "비시험기간"


def parse_di_level_from_text(text: str) -> str:
    """실제 설문 문항의 날씨 표현을 불쾌지수 단계로 단순 매핑한다.

    실제 설문에는 기온/습도 기반 DI가 아니라 맑음, 흐림, 더움 같은 표현이 있으므로
    발표용 모델 입력 범주에 맞춰 다음처럼 변환한다.
    """
    if "더움" in text:
        return "매우 높음"
    if "맑음" in text:
        return "보통"
    if any(keyword in text for keyword in ["흐림", "비", "눈", "쌀쌀"]):
        return "낮음"
    return "보통"


def parse_price_from_text(text: str) -> int | None:
    """문항에 적힌 첫 번째 가격을 추출한다. 예: 2,800원 -> 2800"""
    match = re.search(r"(\d{1,3}(?:,\d{3})*)\s*원", text)
    if match is None:
        return None
    return int(match.group(1).replace(",", ""))


def parse_situation_from_column(column_name: str) -> dict | None:
    """'| 1 | 월요일 오후 12시 ... 2,800원' 형식의 실제 설문 열에서 조건을 추출한다."""
    text = str(column_name)
    situation_match = re.search(r"^\s*\|\s*(\d+)\s*\|", text)
    if situation_match is None:
        return None

    day = next((candidate for candidate in DAYS if candidate in text), None)
    price = parse_price_from_text(text)

    if day is None or price is None:
        return None

    return {
        "상황": int(situation_match.group(1)),
        "문항열": column_name,
        "요일": day,
        "시간대": parse_time_slot_from_text(text),
        "시험기간 여부": parse_exam_period_from_text(text),
        "불쾌지수 단계": parse_di_level_from_text(text),
        "가격": price,
    }


def parse_situation_conditions_from_columns(columns: list[str]) -> pd.DataFrame:
    """실제 설문 파일의 문항 열에서 상황 조건표를 만든다."""
    parsed_rows = []
    for column in columns:
        parsed = parse_situation_from_column(str(column))
        if parsed is not None:
            parsed_rows.append(parsed)
    return pd.DataFrame(parsed_rows)


def convert_wide_survey_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Google Form식 wide 응답 데이터를 모델 학습용 long 데이터로 변환한다."""
    df = remove_private_columns(df)
    rows = []
    parsed_situations = parse_situation_conditions_from_columns(list(df.columns))
    use_parsed_situations = not parsed_situations.empty
    situations = parsed_situations if use_parsed_situations else SITUATION_CONDITIONS

    for respondent_index, response_row in df.iterrows():
        for _, situation in situations.iterrows():
            situation_number = int(situation["상황"])
            situation_column = (
                situation["문항열"]
                if use_parsed_situations and "문항열" in situation
                else find_situation_column(list(df.columns), situation_number)
            )

            if situation_column is None:
                continue

            purchase = normalize_purchase_answer(response_row[situation_column])
            if purchase is None:
                continue

            rows.append(
                {
                    "응답자": respondent_index + 1,
                    "상황": situation_number,
                    "요일": situation["요일"],
                    "시간대": situation["시간대"],
                    "시험기간 여부": situation["시험기간 여부"],
                    "불쾌지수 단계": situation["불쾌지수 단계"],
                    "가격": int(situation["가격"]),
                    "구매여부": purchase,
                }
            )

    return pd.DataFrame(rows)


def normalize_long_survey(df: pd.DataFrame) -> pd.DataFrame:
    """이미 long format인 설문 데이터를 학습 가능한 형태로 정리한다."""
    df = remove_private_columns(df)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])

    if not required_columns.issubset(df.columns):
        return pd.DataFrame()

    long_df = df[list(required_columns)].copy()
    long_df["구매여부"] = long_df["구매여부"].apply(normalize_purchase_answer)
    long_df["가격"] = pd.to_numeric(long_df["가격"], errors="coerce")
    long_df = long_df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN])
    long_df["가격"] = long_df["가격"].astype(int)
    long_df["구매여부"] = long_df["구매여부"].astype(int)

    return long_df[FEATURE_COLUMNS + [TARGET_COLUMN]]


def prepare_training_data(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """업로드 데이터가 long인지 wide인지 판단하고 학습 데이터로 변환한다."""
    long_df = normalize_long_survey(raw_df)
    if not long_df.empty:
        return long_df, "업로드한 long format 데이터를 사용했습니다."

    wide_long_df = convert_wide_survey_to_long(raw_df)
    if not wide_long_df.empty:
        return wide_long_df, "업로드한 Google Form식 wide 데이터를 long format으로 변환했습니다."

    return pd.DataFrame(), "업로드 데이터에서 상황별 구매 응답을 찾지 못했습니다."


def read_csv_with_fallback(path: Path) -> pd.DataFrame:
    """CSV 인코딩이 달라도 읽을 수 있도록 여러 인코딩을 순서대로 시도한다."""
    last_error = None
    for encoding in ["utf-8-sig", "utf-8", "cp949", "euc-kr"]:
        try:
            return pd.read_csv(path, encoding=encoding)
        except Exception as error:
            last_error = error
    raise ValueError(f"CSV 파일을 읽지 못했습니다: {last_error}")


def load_data_file(path: Path) -> pd.DataFrame:
    """상대경로 데이터 파일을 확장자에 맞게 읽는다."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return read_csv_with_fallback(path)
    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError(f"지원하지 않는 데이터 파일 형식입니다: {path.name}")


def load_default_survey_data() -> tuple[pd.DataFrame, str]:
    """실제 설문 파일을 우선 사용하고, 없으면 sample_data.csv 또는 생성 예시 데이터를 사용한다."""
    for path in SURVEY_DATA_CANDIDATES:
        if path.exists():
            try:
                return load_data_file(path), f"실제 설문 파일 {path.name}을 사용했습니다."
            except Exception as error:
                st.warning(f"{path.name} 파일을 읽지 못했습니다: {error}")

    if SAMPLE_DATA_PATH.exists():
        try:
            return read_csv_with_fallback(SAMPLE_DATA_PATH), "실제 설문 파일이 없어 sample_data.csv를 사용했습니다."
        except Exception as error:
            st.warning(f"sample_data.csv를 읽지 못했습니다: {error}")

    return make_example_survey_data(), "데이터 파일이 없어 코드 내부 예시 데이터를 생성했습니다."


# ------------------------------------------------------------
# 3. 예시 설문 데이터 생성
# ------------------------------------------------------------

def estimate_example_probability(situation: pd.Series, respondent_index: int) -> float:
    """예시 데이터 생성을 위한 구매확률 규칙이다.

    실제 앱의 학습 모델은 이 규칙을 직접 사용하지 않는다.
    설문 파일이 없을 때만 시연 가능한 가상 응답을 만들기 위해 사용한다.
    """
    score = 0.0

    if situation["가격"] <= 1800:
        score += 1.4
    elif situation["가격"] <= 2200:
        score += 0.5
    elif situation["가격"] >= 2600:
        score -= 1.2

    if situation["시간대"] == "11시~14시":
        score += 0.8
    elif situation["시간대"] == "19시 이후":
        score -= 0.8

    if situation["시험기간 여부"] == "시험기간":
        score += 0.7

    if situation["불쾌지수 단계"] == "높음":
        score += 0.4
    elif situation["불쾌지수 단계"] == "매우 높음":
        score += 0.6

    if situation["요일"] in ["토요일", "일요일"]:
        score -= 0.7

    # 응답자별 성향 차이를 약간 부여해 구매/비구매가 모두 생기도록 한다.
    score += ((respondent_index % 7) - 3) * 0.2
    return 1 / (1 + exp(-score))


def make_example_survey_data(respondents: int = 60) -> pd.DataFrame:
    """설문 파일이 없어도 앱을 실행할 수 있도록 예시 wide 데이터를 만든다."""
    rows = []

    for respondent_index in range(respondents):
        row = {
            "이름": f"예시응답자{respondent_index + 1}",
            "이메일": f"example{respondent_index + 1}@cau.ac.kr",
        }

        for _, situation in SITUATION_CONDITIONS.iterrows():
            probability = estimate_example_probability(situation, respondent_index)

            # 완전 난수를 쓰지 않고, 응답자 번호와 상황 번호로 결정해 매 실행 결과가 안정적이게 한다.
            threshold = ((respondent_index * 17 + int(situation["상황"]) * 11) % 100) / 100
            answer = "구매하겠다" if threshold < probability else "구매하지 않겠다"
            row[f"상황 {int(situation['상황'])}"] = answer

        rows.append(row)

    return pd.DataFrame(rows)


# ------------------------------------------------------------
# 4. 모델 학습과 가격 시뮬레이션 함수
# ------------------------------------------------------------

def train_purchase_model(training_df: pd.DataFrame) -> tuple[Pipeline, float, str]:
    """Logistic Regression 모델을 학습하고 정확도를 계산한다."""
    X = training_df[FEATURE_COLUMNS]
    y = training_df[TARGET_COLUMN]

    categorical_features = ["요일", "시간대", "시험기간 여부", "불쾌지수 단계"]
    numeric_features = ["가격"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("category", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("number", StandardScaler(), numeric_features),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )

    class_counts = y.value_counts()
    can_split = len(training_df) >= 30 and len(class_counts) == 2 and class_counts.min() >= 2

    if can_split:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
        model.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, model.predict(X_test))
        accuracy_note = "검증 데이터 기준 정확도"
    else:
        model.fit(X, y)
        accuracy = accuracy_score(y, model.predict(X))
        accuracy_note = "학습 데이터 기준 정확도"

    return model, accuracy, accuracy_note


def simulate_prices(
    model: Pipeline,
    day: str,
    time_slot: str,
    exam_period: str,
    di_level: str,
    expected_customers: int,
) -> pd.DataFrame:
    """후보 가격별 구매확률과 예상매출을 계산한다."""
    simulation_df = pd.DataFrame(
        [
            {
                "요일": day,
                "시간대": time_slot,
                "시험기간 여부": exam_period,
                "불쾌지수 단계": di_level,
                "가격": price,
            }
            for price in PRICE_CANDIDATES
        ]
    )

    purchase_probabilities = model.predict_proba(simulation_df[FEATURE_COLUMNS])[:, 1]
    simulation_df["구매확률"] = purchase_probabilities
    simulation_df["예상 고객 수"] = expected_customers
    simulation_df["예상매출"] = simulation_df["가격"] * simulation_df["구매확률"] * expected_customers

    best_index = simulation_df["예상매출"].idxmax()
    simulation_df["추천 여부"] = False
    simulation_df.loc[best_index, "추천 여부"] = True

    return simulation_df


def format_won(value: float) -> str:
    """숫자를 원화 형식으로 표시한다."""
    return f"{value:,.0f}원"


# ------------------------------------------------------------
# 5. 화면 구성
# ------------------------------------------------------------

st.title("310관 커피니 동적가격제 시뮬레이터")

st.write(
    "중앙대학교 310관 커피니의 아메리카노 가격 후보를 비교하기 위해 "
    "설문 응답 기반 Logistic Regression 모델로 구매확률을 예측하고, "
    "후보 가격별 예상매출이 가장 높은 가격을 추천합니다."
)

st.info(
    "불쾌지수는 기온과 습도를 바탕으로 사람이 느끼는 불쾌감 정도를 나타내는 지표입니다. "
    "매경헬스 기사 기준으로 68 미만은 낮음, 68~75 미만은 보통, "
    "75~80 미만은 높음, 80 이상은 매우 높음 단계로 사용합니다."
)

st.divider()

st.subheader("학습 데이터")

raw_df, data_source = load_default_survey_data()
training_df, data_message = prepare_training_data(raw_df)

with st.expander("현재 사용 중인 설문 데이터 확인"):
    st.write(data_source)
    st.dataframe(raw_df.head(10), width="stretch", hide_index=True)

    parsed_situations = parse_situation_conditions_from_columns(list(raw_df.columns))
    if not parsed_situations.empty:
        st.write("실제 설문 문항에서 자동 추출한 상황 조건")
        st.dataframe(
            parsed_situations.drop(columns=["문항열"]),
            width="stretch",
            hide_index=True,
        )
    else:
        st.write("예시 데이터용 상황 1~20 조건")
        st.dataframe(SITUATION_CONDITIONS, width="stretch", hide_index=True)

if training_df.empty or training_df[TARGET_COLUMN].nunique() < 2:
    st.warning(
        "현재 설문 데이터만으로는 모델을 학습하기 어려워 예시 데이터를 사용합니다. "
        "구매하겠다와 구매하지 않겠다 응답이 모두 포함되어야 합니다."
    )
    fallback_raw_df = make_example_survey_data()
    training_df, data_message = prepare_training_data(fallback_raw_df)
    data_source = "예시 데이터"

st.caption(f"사용 데이터: {data_source} / {data_message}")

model, accuracy, accuracy_note = train_purchase_model(training_df)

purchase_rate = training_df[TARGET_COLUMN].mean()

st.subheader("모델 학습 결과")
result_col1, result_col2, result_col3 = st.columns(3)

with result_col1:
    st.metric("총 학습 데이터 수", f"{len(training_df):,}개")

with result_col2:
    st.metric("구매 응답 비율", f"{purchase_rate * 100:.1f}%")

with result_col3:
    st.metric("모델 정확도", f"{accuracy * 100:.1f}%")
    st.caption(accuracy_note)

with st.expander("학습 데이터 미리보기"):
    st.dataframe(training_df.head(30), width="stretch", hide_index=True)

st.divider()

st.subheader("시뮬레이션 입력")
input_col1, input_col2, input_col3, input_col4, input_col5 = st.columns(5)

with input_col1:
    selected_day = st.selectbox("요일", DAYS, index=DAYS.index("월요일"))

with input_col2:
    selected_time_slot = st.selectbox("시간대", TIME_SLOTS, index=TIME_SLOTS.index("11시~14시"))

with input_col3:
    selected_exam_period = st.radio("시험기간 여부", EXAM_PERIODS, index=EXAM_PERIODS.index("비시험기간"))

with input_col4:
    selected_di_level = st.selectbox("불쾌지수 단계", DI_LEVELS, index=DI_LEVELS.index("보통"))

with input_col5:
    expected_customers = st.number_input(
        "예상 고객 수",
        min_value=1,
        max_value=10000,
        value=1,
        step=1,
        help="발표용 기본값은 1명입니다. 향후 실제 예상 방문 고객 수를 넣어 확장할 수 있습니다.",
    )

simulation_results = simulate_prices(
    model=model,
    day=selected_day,
    time_slot=selected_time_slot,
    exam_period=selected_exam_period,
    di_level=selected_di_level,
    expected_customers=int(expected_customers),
)

recommended_row = simulation_results.loc[simulation_results["추천 여부"]].iloc[0]
recommended_price = int(recommended_row["가격"])
recommended_probability = float(recommended_row["구매확률"])
recommended_revenue = float(recommended_row["예상매출"])

display_results = simulation_results.copy()
display_results["구매확률"] = display_results["구매확률"].map(lambda value: f"{value * 100:.1f}%")
display_results["예상매출"] = display_results["예상매출"].map(format_won)
display_results["가격"] = display_results["가격"].map(format_won)
display_results["추천 여부"] = display_results["추천 여부"].map(lambda value: "추천" if value else "")

st.subheader("결과 출력")

product_col, detail_col = st.columns([0.9, 1.6])

with product_col:
    if AMERICANO_IMAGE_PATH.exists():
        st.image(str(AMERICANO_IMAGE_PATH), width=260)
    else:
        st.markdown("### 아메리카노")

    st.markdown(f"#### {MENU_NAME} 추천 결과")
    st.metric("추천 가격", format_won(recommended_price))
    st.metric("추천 가격 구매확률", f"{recommended_probability * 100:.1f}%")
    st.metric("추천 가격 예상매출", format_won(recommended_revenue))

with detail_col:
    st.success(
        f"{MENU_NAME} 후보 가격 중 {format_won(recommended_price)}의 예상매출이 가장 높습니다. "
        "교내 카페 시뮬레이션이므로 수요가 높아도 가격 인상 자체를 목표로 하기보다, "
        "설문 기반 구매확률과 예상매출을 함께 비교해 수용 가능한 가격을 찾는 전략으로 해석합니다."
    )

    st.write("후보 가격별 구매확률과 예상매출")
    st.dataframe(display_results, width="stretch", hide_index=True)

st.divider()

st.subheader("시각화")

chart_df = simulation_results.copy()
chart_df["가격_label"] = chart_df["가격"].map(lambda price: f"{price}원")
chart_df["추천"] = chart_df["추천 여부"].map(lambda value: "추천 가격" if value else "후보 가격")

probability_chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("가격_label:N", title="가격", sort=[f"{price}원" for price in PRICE_CANDIDATES]),
        y=alt.Y("구매확률:Q", title="구매확률", axis=alt.Axis(format="%")),
        color=alt.Color("추천:N", scale=alt.Scale(domain=["후보 가격", "추천 가격"], range=["#7aa6c2", "#d95f5f"])),
        tooltip=["가격", alt.Tooltip("구매확률:Q", format=".1%"), "추천"],
    )
    .properties(title="가격별 구매확률")
)

revenue_chart = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("가격_label:N", title="가격", sort=[f"{price}원" for price in PRICE_CANDIDATES]),
        y=alt.Y("예상매출:Q", title="예상매출"),
        color=alt.Color("추천:N", scale=alt.Scale(domain=["후보 가격", "추천 가격"], range=["#7aa6c2", "#d95f5f"])),
        tooltip=["가격", alt.Tooltip("예상매출:Q", format=",.0f"), "추천"],
    )
    .properties(title="가격별 예상매출")
)

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.altair_chart(probability_chart, width="stretch")

with chart_col2:
    st.altair_chart(revenue_chart, width="stretch")

st.caption("빨간색 막대는 예상매출이 가장 높은 추천 가격입니다.")
