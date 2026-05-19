import json
import os
import re
import sqlite3
import uuid
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

DB_PATH = os.getenv("DB_PATH", "crossfit_exercises.db")
TABLE_NAME = os.getenv("TABLE_NAME", "crossfit_exercises")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
TIMEZONE = os.getenv("TIMEZONE", "Europe/Berlin")
MAX_EXERCISES_FOR_PROMPT = int(os.getenv("MAX_EXERCISES_FOR_PROMPT", "80"))

REQUIRED_COLUMNS = {
    "name", "category", "movement_pattern", "equipment", "level",
    "primary_muscles", "skill_level", "intensity_type", "wod_formats",
    "default_sets", "default_reps", "default_duration_sec", "coaching_cues",
    "regression", "progression", "avoid_if", "tags",
}

PROMPT_EXERCISE_COLUMNS = [
    "name", "category", "movement_pattern", "equipment", "level",
    "primary_muscles", "skill_level", "intensity_type", "wod_formats",
    "default_sets", "default_reps", "default_duration_sec", "coaching_cues",
    "regression", "progression", "avoid_if", "tags",
]

WEEKDAY_MAP = {
    "Понедельник": 0,
    "Вторник": 1,
    "Среда": 2,
    "Четверг": 3,
    "Пятница": 4,
    "Суббота": 5,
    "Воскресенье": 6,
}

st.set_page_config(
    page_title="AI Fitness Planner",
    page_icon="🏋️",
    layout="wide",
)


def safe_table_name(table_name: str) -> str:
    """Защита от случайного SQL-инъекционного значения в TABLE_NAME из .env."""
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", table_name):
        raise ValueError(
            "Некорректное имя таблицы в TABLE_NAME. "
            "Используй только буквы, цифры и подчёркивание."
        )
    return table_name


def split_tokens(value) -> list[str]:
    """Превращает строку вида 'dumbbell, kettlebell' в список токенов."""
    if value is None:
        return []
    if pd.isna(value):
        return []
    text = str(value).strip().lower()
    if not text:
        return []
    return [token.strip() for token in re.split(r"[,;/|]+", text) if token.strip()]


def clean_records_for_json(df: pd.DataFrame) -> list[dict]:
    """Убирает NaN, чтобы json.dumps не ломался и не отправлял NaN в API."""
    safe_df = df.where(pd.notna(df), None)
    return safe_df.to_dict(orient="records")


def get_unique_tokens(df: pd.DataFrame, column: str, fallback: list[str]) -> list[str]:
    if column not in df.columns:
        return fallback
    values = set()
    for item in df[column].dropna().tolist():
        for token in split_tokens(item):
            values.add(token)
    result = sorted(values)
    return result if result else fallback


def get_default_values(options: list[str], preferred: list[str]) -> list[str]:
    return [item for item in preferred if item in options]


@st.cache_data(ttl=3600, show_spinner=False)
def load_exercises_from_db(db_path: str, table_name: str) -> pd.DataFrame:
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Файл БД не найден: {db_path}. "
            f"Положи crossfit_exercises.db рядом с app.py или укажи DB_PATH в .env."
        )

    table_name = safe_table_name(table_name)
    conn = sqlite3.connect(db_path)

    try:
        table_exists = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name = ?
            """,
            conn,
            params=(table_name,),
        )
        if table_exists.empty:
            raise RuntimeError(
                f"В базе {db_path} нет таблицы {table_name}. "
                f"Проверь, что ты импортировал SQL seed-файл."
            )
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    finally:
        conn.close()

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        raise RuntimeError("В таблице не хватает колонок: " + ", ".join(sorted(missing_columns)))
    return df


def filter_exercises(
    df: pd.DataFrame,
    level: str,
    equipment: list[str],
    avoid_list: list[str],
    goal: str,
) -> pd.DataFrame:
    filtered = df.copy()
    level_map = {
        "beginner": ["beginner"],
        "intermediate": ["beginner", "intermediate"],
        "advanced": ["beginner", "intermediate", "advanced"],
    }
    filtered = filtered[filtered["level"].isin(level_map.get(level, ["beginner"]))]

    selected_equipment = set(item.lower() for item in equipment)
    if "bodyweight" in selected_equipment:
        selected_equipment.add("none")
    if "none" in selected_equipment:
        selected_equipment.add("bodyweight")

    if selected_equipment:
        def equipment_match(row_equipment) -> bool:
            row_tokens = set(split_tokens(row_equipment))
            if not row_tokens:
                return True
            return bool(row_tokens & selected_equipment)
        filtered = filtered[filtered["equipment"].apply(equipment_match)]

    avoid_tokens = set(item.lower() for item in avoid_list)
    if avoid_tokens:
        def is_not_avoided(avoid_if_value) -> bool:
            row_avoid_tokens = split_tokens(avoid_if_value)
            for selected_avoid in avoid_tokens:
                for row_avoid in row_avoid_tokens:
                    if selected_avoid in row_avoid or row_avoid in selected_avoid:
                        return False
            return True
        filtered = filtered[filtered["avoid_if"].apply(is_not_avoided)]

    goal_lower = goal.lower()
    if "hyrox" in goal_lower:
        preferred_patterns = [
            "locomotion", "loaded_push", "loaded_pull", "carry", "squat_press",
            "lunge", "horizontal_pull", "hinge",
        ]
    elif "снижение веса" in goal_lower:
        preferred_patterns = [
            "full_body", "jump", "cyclical", "locomotion", "squat_press", "hinge", "carry",
        ]
    else:
        preferred_patterns = []

    if preferred_patterns:
        filtered = (
            filtered.assign(
                _goal_rank=filtered["movement_pattern"].apply(
                    lambda x: 0 if str(x).lower() in preferred_patterns else 1
                )
            )
            .sort_values(["_goal_rank", "level", "name"])
            .drop(columns=["_goal_rank"])
        )
    else:
        filtered = filtered.sort_values(["level", "category", "name"])

    return filtered.head(MAX_EXERCISES_FOR_PROMPT)


def build_plan_schema(allowed_exercise_names: list[str], days_per_week: int) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "overview", "weeks"],
        "properties": {
            "title": {"type": "string"},
            "overview": {"type": "string"},
            "weeks": {
                "type": "array",
                "minItems": 4,
                "maxItems": 4,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["week_number", "focus", "days"],
                    "properties": {
                        "week_number": {"type": "integer", "minimum": 1, "maximum": 4},
                        "focus": {"type": "string"},
                        "days": {
                            "type": "array",
                            "minItems": days_per_week,
                            "maxItems": days_per_week,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "required": [
                                    "day_number", "title", "warmup", "main_block", "cooldown", "coach_notes",
                                ],
                                "properties": {
                                    "day_number": {"type": "integer", "minimum": 1, "maximum": days_per_week},
                                    "title": {"type": "string"},
                                    "warmup": {
                                        "type": "array",
                                        "minItems": 2,
                                        "maxItems": 6,
                                        "items": {"type": "string"},
                                    },
                                    "main_block": {
                                        "type": "array",
                                        "minItems": 3,
                                        "maxItems": 8,
                                        "items": {
                                            "type": "object",
                                            "additionalProperties": False,
                                            "required": ["exercise", "sets", "reps_or_time", "rest_sec", "notes"],
                                            "properties": {
                                                "exercise": {"type": "string", "enum": allowed_exercise_names},
                                                "sets": {"type": "integer", "minimum": 1, "maximum": 10},
                                                "reps_or_time": {"type": "string"},
                                                "rest_sec": {"type": "integer", "minimum": 0, "maximum": 300},
                                                "notes": {"type": "string"},
                                            },
                                        },
                                    },
                                    "cooldown": {
                                        "type": "array",
                                        "minItems": 1,
                                        "maxItems": 5,
                                        "items": {"type": "string"},
                                    },
                                    "coach_notes": {"type": "string"},
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def extract_response_text(response) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return output_text

    output = getattr(response, "output", None)
    if not output:
        raise RuntimeError("OpenAI API не вернул output_text.")

    chunks = []
    for item in output:
        content = getattr(item, "content", None) or []
        for content_item in content:
            text = getattr(content_item, "text", None)
            if text:
                chunks.append(text)

    if not chunks:
        raise RuntimeError("Не удалось извлечь текст ответа OpenAI API.")
    return "\n".join(chunks)


def validate_generated_plan(plan: dict, allowed_exercise_names: list[str], days_per_week: int) -> None:
    allowed = set(allowed_exercise_names)
    weeks = plan.get("weeks", [])

    if len(weeks) != 4:
        raise ValueError("ИИ вернул не 4 недели плана.")

    invalid_exercises = []
    for week in weeks:
        days = week.get("days", [])
        if len(days) != days_per_week:
            raise ValueError(
                f"ИИ вернул неверное количество дней в неделе {week.get('week_number')}: "
                f"{len(days)} вместо {days_per_week}."
            )
        for day in days:
            for block in day.get("main_block", []):
                exercise_name = block.get("exercise")
                if exercise_name not in allowed:
                    invalid_exercises.append(exercise_name)

    if invalid_exercises:
        invalid_text = ", ".join(sorted(set(str(x) for x in invalid_exercises)))
        raise ValueError("ИИ использовал упражнения не из БД: " + invalid_text)


def generate_training_plan(user_profile: dict, exercises_df: pd.DataFrame) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("Не найден OPENAI_API_KEY. Добавь ключ в файл .env.")

    allowed_exercise_names = (
        exercises_df["name"].dropna().astype(str).drop_duplicates().tolist()
    )
    if not allowed_exercise_names:
        raise RuntimeError("После фильтрации не осталось упражнений для плана.")

    schema = build_plan_schema(
        allowed_exercise_names=allowed_exercise_names,
        days_per_week=user_profile["days_per_week"],
    )
    exercises_for_prompt = clean_records_for_json(exercises_df[PROMPT_EXERCISE_COLUMNS])

    system_prompt = """
Ты - профессиональный тренер по functional fitness/CrossFit-style тренировкам с огромным стажем.

Главные правила:
1. Используй только упражнения из списка exercises.
2. В поле exercise используй точное имя упражнения из БД без изменений.
3. Не придумывай упражнения, которых нет в списке.
4. Учитывай пол, возраст, уровень, цель, инвентарь, ограничения и дни тренировок.
5. Для beginner не ставь advanced-движения и слишком большой объём.
6. Если есть ограничения, избегай упражнений, где avoid_if конфликтует с ограничениями пользователя.
7. План должен быть на 4 недели.
8. В каждой неделе должно быть ровно days_per_week тренировочных дней.
9. Каждая тренировка должна включать warmup, main_block, cooldown и coach_notes.
10. Прогрессия должна быть реалистичной: неделя 1 легче, неделя 2-3 растёт нагрузка, неделя 4 закрепляет результат.
11. Не давай медицинских обещаний и диагнозов.
12. Ответ верни строго в JSON по схеме.
"""

    payload = {
        "user_profile": user_profile,
        "exercises": exercises_for_prompt,
        "task": "Составь месячный план тренировок на 4 недели.",
    }

    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "month_training_plan",
                "schema": schema,
                "strict": True,
            }
        },
        max_output_tokens=12000,
    )

    response_text = extract_response_text(response)
    plan = json.loads(response_text)
    validate_generated_plan(plan, allowed_exercise_names, user_profile["days_per_week"])
    return plan


def build_training_dates(start_date: date, selected_weekdays: list[str], total_sessions: int) -> list[date]:
    selected_weekday_numbers = {WEEKDAY_MAP[weekday] for weekday in selected_weekdays}
    result = []
    current_date = start_date
    guard = 0
    while len(result) < total_sessions and guard < 370:
        if current_date.weekday() in selected_weekday_numbers:
            result.append(current_date)
        current_date += timedelta(days=1)
        guard += 1
    return result


def group_dates_by_week(training_dates: list[date], days_per_week: int) -> list[list[date]]:
    return [training_dates[index:index + days_per_week] for index in range(0, len(training_dates), days_per_week)]


def get_plan_day_date(grouped_dates: list[list[date]], week_index: int, day_index: int) -> date | None:
    if week_index >= len(grouped_dates):
        return None
    if day_index >= len(grouped_dates[week_index]):
        return None
    return grouped_dates[week_index][day_index]


def plan_to_markdown(plan: dict, grouped_dates: list[list[date]]) -> str:
    lines = [f"# {plan['title']}", "", plan["overview"], ""]

    for week_index, week in enumerate(plan["weeks"]):
        lines.append(f"## Неделя {week['week_number']}: {week['focus']}")
        lines.append("")

        for day_index, day in enumerate(week["days"]):
            event_date = get_plan_day_date(grouped_dates, week_index, day_index)
            date_text = event_date.strftime("%d.%m.%Y") if event_date else "Без даты"
            lines.append(f"### {date_text} — {day['title']}")
            lines.append("")

            lines.append("**Разминка:**")
            for item in day["warmup"]:
                lines.append(f"- {item}")

            lines.append("")
            lines.append("**Основной блок:**")
            for block in day["main_block"]:
                lines.append(
                    f"- **{block['exercise']}**: "
                    f"{block['sets']} подхода × {block['reps_or_time']}; "
                    f"отдых {block['rest_sec']} сек. {block['notes']}"
                )

            lines.append("")
            lines.append("**Заминка:**")
            for item in day["cooldown"]:
                lines.append(f"- {item}")

            lines.append("")
            lines.append(f"**Заметки тренера:** {day['coach_notes']}")
            lines.append("")

    return "\n".join(lines)


def ics_escape(value) -> str:
    value = str(value)
    return value.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def fold_ics_line(line: str, limit: int = 73) -> list[str]:
    if len(line) <= limit:
        return [line]
    result = []
    while len(line) > limit:
        result.append(line[:limit])
        line = " " + line[limit:]
    result.append(line)
    return result


def plan_to_ics(plan: dict, grouped_dates: list[list[date]], training_time: time, session_minutes: int) -> str:
    tz = ZoneInfo(TIMEZONE)
    dtstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//AI Fitness Planner//Training Plan//RU",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:AI Training Plan",
    ]

    for week_index, week in enumerate(plan["weeks"]):
        for day_index, day in enumerate(week["days"]):
            event_date = get_plan_day_date(grouped_dates, week_index, day_index)
            if event_date is None:
                continue

            start_dt = datetime.combine(event_date, training_time).replace(tzinfo=tz)
            end_dt = start_dt + timedelta(minutes=session_minutes)

            description_parts = ["Разминка:"]
            description_parts.extend(day["warmup"])
            description_parts.append("")
            description_parts.append("Основной блок:")

            for block in day["main_block"]:
                description_parts.append(
                    f"{block['exercise']}: {block['sets']} x {block['reps_or_time']}; "
                    f"отдых {block['rest_sec']} сек. {block['notes']}"
                )

            description_parts.append("")
            description_parts.append("Заминка:")
            description_parts.extend(day["cooldown"])
            description_parts.append("")
            description_parts.append(f"Заметки тренера: {day['coach_notes']}")

            uid_seed = f"{plan['title']}-{week_index + 1}-{day_index + 1}-{event_date.isoformat()}"
            event_uid = str(uuid.uuid5(uuid.NAMESPACE_URL, uid_seed))

            raw_lines.extend([
                "BEGIN:VEVENT",
                f"UID:{event_uid}@ai-fitness-planner",
                f"DTSTAMP:{dtstamp}",
                f"DTSTART;TZID={TIMEZONE}:{start_dt.strftime('%Y%m%dT%H%M%S')}",
                f"DTEND;TZID={TIMEZONE}:{end_dt.strftime('%Y%m%dT%H%M%S')}",
                f"SUMMARY:{ics_escape('Тренировка: ' + day['title'])}",
                f"DESCRIPTION:{ics_escape(chr(10).join(description_parts))}",
                "END:VEVENT",
            ])

    raw_lines.append("END:VCALENDAR")
    folded_lines = []
    for line in raw_lines:
        folded_lines.extend(fold_ics_line(line))
    return "\r\n".join(folded_lines)


def render_plan(plan: dict, grouped_dates: list[list[date]]) -> None:
    st.header(plan["title"])
    st.write(plan["overview"])

    for week_index, week in enumerate(plan["weeks"]):
        with st.expander(f"Неделя {week['week_number']}: {week['focus']}", expanded=True):
            for day_index, day in enumerate(week["days"]):
                event_date = get_plan_day_date(grouped_dates, week_index, day_index)
                date_text = event_date.strftime("%d.%m.%Y") if event_date else "Без даты"
                st.subheader(f"{date_text} — {day['title']}")

                col1, col2, col3 = st.columns([1, 2, 1])

                with col1:
                    st.markdown("#### Разминка")
                    for item in day["warmup"]:
                        st.write(f"• {item}")

                with col2:
                    st.markdown("#### Основной блок")
                    for block in day["main_block"]:
                        st.markdown(
                            f"""
**{block['exercise']}**  
{block['sets']} подхода × {block['reps_or_time']}  
Отдых: {block['rest_sec']} сек.  
_{block['notes']}_
"""
                        )

                with col3:
                    st.markdown("#### Заминка")
                    for item in day["cooldown"]:
                        st.write(f"• {item}")

                st.info(day["coach_notes"])
                st.divider()


st.title("🏋️ AI-план тренировок на месяц")
st.caption(
    "Streamlit-приложение: разработано специально для hybrid-атлетов."
)

with st.sidebar:
    st.subheader("Настройки")
    st.write(f"**DB_PATH:** `{DB_PATH}`")
    st.write(f"**TABLE_NAME:** `{TABLE_NAME}`")
    st.write(f"**MODEL:** `{OPENAI_MODEL}`")

    if OPENAI_API_KEY:
        st.success("OPENAI_API_KEY найден в .env")
    else:
        st.warning("OPENAI_API_KEY не найден в .env")

try:
    all_exercises_df = load_exercises_from_db(DB_PATH, TABLE_NAME)
except Exception as error:
    st.error(str(error))
    st.markdown("Пример `.env`:")
    st.code(
        """OPENAI_API_KEY=твой_ключ
OPENAI_MODEL=gpt-4.1-mini
DB_PATH=crossfit_exercises.db
TABLE_NAME=crossfit_exercises
TIMEZONE=Europe/Berlin""",
        language="env",
    )
    st.stop()

equipment_options = get_unique_tokens(
    all_exercises_df,
    "equipment",
    fallback=[
        "bodyweight", "none", "dumbbell", "kettlebell", "barbell",
        "pullup_bar", "rings", "trx", "box", "jump_rope", "row_erg",
        "bike_erg", "wall_ball", "sled", "sandbag",
    ],
)

avoid_options = get_unique_tokens(
    all_exercises_df,
    "avoid_if",
    fallback=[
        "knee_pain", "shoulder_pain", "wrist_pain", "acute_back_pain",
        "acute_lower_back_pain", "achilles_pain", "high_impact_restriction",
    ],
)

with st.expander("Посмотреть упражнения в БД", expanded=False):
    st.write(f"Всего упражнений в таблице: **{len(all_exercises_df)}**")
    st.dataframe(
        all_exercises_df[["name", "category", "movement_pattern", "equipment", "level", "avoid_if"]],
        width="stretch",
    )

with st.form("training_profile_form"):
    st.subheader("Анкета пользователя")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Пол", ["Мужчина", "Женщина"])
        age = st.number_input("Возраст", min_value=14, max_value=80, value=30, step=1)

    with col2:
        level = st.selectbox(
            "Уровень подготовленности",
            ["beginner", "intermediate", "advanced"],
            format_func={
                "beginner": "Beginner / новичок",
                "intermediate": "Intermediate / средний",
                "advanced": "Advanced / продвинутый",
            }.get,
        )
        goal = st.selectbox(
            "Цель",
            [
                "Общая физическая форма",
                "Снижение веса",
                "Силовая выносливость",
                "Подготовка к CrossFit-style тренировкам",
                "Подготовка к HYROX-style тренировкам",
            ],
        )

    with col3:
        session_minutes = st.slider("Длительность тренировки, минут", 20, 90, 45, step=5)
        training_time = st.time_input("Время тренировки", value=time(18, 30))

    st.subheader("Расписание и условия")
    col4, col5 = st.columns(2)

    with col4:
        start_date = st.date_input("Дата начала", value=date.today())
        selected_weekdays = st.multiselect(
            "Дни тренировок",
            list(WEEKDAY_MAP.keys()),
            default=["Понедельник", "Среда", "Пятница"],
        )

    with col5:
        default_equipment = get_default_values(equipment_options, ["bodyweight", "dumbbell", "kettlebell"])
        equipment = st.multiselect("Доступный инвентарь", equipment_options, default=default_equipment)
        avoid = st.multiselect("Ограничения / что избегать", avoid_options, default=[])

    extra_notes = st.text_area(
        "Дополнительные пожелания",
        placeholder="Например: тренировки дома, не люблю бег, хочу больше круговых, нужно без прыжков...",
    )

    submitted = st.form_submit_button("Сгенерировать план на месяц", type="primary")

if submitted:
    if not selected_weekdays:
        st.error("Выбери хотя бы один день тренировки.")
        st.stop()

    days_per_week = len(selected_weekdays)
    total_sessions = days_per_week * 4

    user_profile = {
        "gender": gender,
        "age": int(age),
        "level": level,
        "goal": goal,
        "days_per_week": days_per_week,
        "selected_weekdays": selected_weekdays,
        "session_minutes": int(session_minutes),
        "training_time": training_time.strftime("%H:%M"),
        "start_date": start_date.isoformat(),
        "equipment": equipment,
        "avoid": avoid,
        "extra_notes": extra_notes,
    }

    with st.spinner("Фильтрую упражнения из БД..."):
        filtered_exercises_df = filter_exercises(
            df=all_exercises_df,
            level=level,
            equipment=equipment,
            avoid_list=avoid,
            goal=goal,
        )

    if filtered_exercises_df.empty:
        st.error(
            "После фильтров не осталось упражнений. "
            "Попробуй добавить инвентарь или убрать часть ограничений."
        )
        st.stop()

    with st.expander("Упражнения, которые отправятся ИИ-агенту", expanded=False):
        st.write(f"Найдено упражнений: **{len(filtered_exercises_df)}**")
        st.dataframe(
            filtered_exercises_df[[
                "name", "category", "movement_pattern", "equipment", "level",
                "regression", "progression", "avoid_if",
            ]],
            width="stretch",
        )

    try:
        with st.spinner("ИИ-агент составляет план..."):
            plan = generate_training_plan(user_profile=user_profile, exercises_df=filtered_exercises_df)

        training_dates = build_training_dates(
            start_date=start_date,
            selected_weekdays=selected_weekdays,
            total_sessions=total_sessions,
        )
        grouped_dates = group_dates_by_week(training_dates=training_dates, days_per_week=days_per_week)

        st.session_state["plan"] = plan
        st.session_state["user_profile"] = user_profile
        st.session_state["grouped_dates"] = grouped_dates
        st.session_state["training_time"] = training_time
        st.session_state["session_minutes"] = int(session_minutes)
        st.success("План готов!")

    except Exception as error:
        st.error(f"Не удалось сгенерировать план: {error}")
        st.stop()

if "plan" in st.session_state:
    plan = st.session_state["plan"]
    user_profile = st.session_state["user_profile"]
    grouped_dates = st.session_state["grouped_dates"]
    training_time = st.session_state["training_time"]
    session_minutes = st.session_state["session_minutes"]

    st.divider()
    render_plan(plan, grouped_dates)

    markdown_file = plan_to_markdown(plan, grouped_dates)
    json_file = json.dumps(
        {"user_profile": user_profile, "plan": plan},
        ensure_ascii=False,
        indent=2,
        default=str,
    )
    ics_file = plan_to_ics(
        plan=plan,
        grouped_dates=grouped_dates,
        training_time=training_time,
        session_minutes=session_minutes,
    )

    st.subheader("Экспорт плана тренировок")
    download_col1, download_col2, download_col3 = st.columns(3)

    with download_col1:
        st.download_button(
            label="📄 Скачать Markdown",
            data=markdown_file,
            file_name="month_training_plan.md",
            mime="text/markdown",
            width="stretch",
        )

    with download_col2:
        st.download_button(
            label="🧾 Скачать JSON",
            data=json_file,
            file_name="month_training_plan.json",
            mime="application/json",
            width="stretch",
        )

    with download_col3:
        st.download_button(
            label="📅 Скачать календарь .ics",
            data=ics_file,
            file_name="month_training_plan.ics",
            mime="text/calendar",
            width="stretch",
        )

st.caption(
    "*Обратите внимание: При травмах, боли или заболеваниях "
    "лучше согласовать нагрузку с вашим медицинским специалистом.*"
)
