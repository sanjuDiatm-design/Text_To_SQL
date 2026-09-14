import gc
import os
import re
import sqlite3
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "student.db")
UPLOADED_DB_PATH = os.path.join(os.path.dirname(__file__), "uploaded_custom.db")
MODEL_NAME = "openai/gpt-oss-20b"

# ----------------- Helper & Schema Functions -----------------

def clean_sql_query(sql_query: str) -> str:
    """Removes markdown code fences, backticks, and cleans whitespace."""
    sql_query = sql_query.strip()
    if sql_query.startswith("```"):
        lines = sql_query.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        sql_query = "\n".join(lines).strip()
    return sql_query


def is_valid_sql(query: str) -> bool:
    """Validates if the returned query is a recognized SQLite query, not an error or gibberish signal."""
    if not query:
        return False
    clean = query.strip().upper()
    if clean.startswith("INVALID_QUERY") or clean.startswith("ERROR") or "NO QUERY" in clean or "CANNOT" in clean:
        return False
    # Must start with standard data-retrieval SQL keywords
    valid_starters = ("SELECT", "WITH", "PRAGMA", "EXPLAIN")
    return any(clean.startswith(starter) for starter in valid_starters)


def render_dataframe(df: pd.DataFrame, **kwargs):
    """Renders a dataframe cleanly without deprecation warnings across Streamlit versions."""
    try:
        st.dataframe(df, width="stretch", **kwargs)
    except TypeError:
        st.dataframe(df, use_container_width=True, **kwargs)


def get_db_schema(db_path: str) -> dict:
    """Inspects SQLite database and returns tables, column names, column types, and row counts."""
    schema_info = {}
    if not os.path.exists(db_path):
        return schema_info

    try:
        conn = sqlite3.connect(db_path)
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
            tables = [row[0] for row in cursor.fetchall()]

            for table in tables:
                cursor.execute(f'PRAGMA table_info("{table}")')
                cols = [(row[1], row[2] if row[2] else "TEXT") for row in cursor.fetchall()]
                cursor.execute(f'SELECT COUNT(*) FROM "{table}"')
                count = cursor.fetchone()[0]
                schema_info[table] = {
                    "columns": cols,
                    "count": count
                }
        finally:
            conn.close()
    except Exception as e:
        st.error(f"Error inspecting schema: {e}")

    return schema_info


def format_schema_for_prompt(schema_info: dict) -> str:
    """Formats schema details into a clear prompt string for the LLM."""
    if not schema_info:
        return "No tables or schema available."

    lines = []
    for table, meta in schema_info.items():
        col_list = [f"- {col_name} ({col_type})" for col_name, col_type in meta["columns"]]
        lines.append(f"Table: {table} ({meta['count']} records)\nColumns:\n" + "\n".join(col_list))
    return "\n\n".join(lines)


def get_sql_query(user_query: str, schema_context: str) -> str:
    """Generates SQL query using Groq model with deterministic temperature=0.0."""
    groq_sys_prompt = ChatPromptTemplate.from_template("""
        You are an expert in converting natural English questions into valid SQLite SQL queries!
        The database contains the following tables and schemas:
        {schema_context}

        STRICT RULES:
        1. If the user question is random gibberish, meaningless characters (e.g. 'gardfGHUH', 'asdkjh', 'xyz123'), nonsensical, or cannot be understood as a question about this database, output EXACTLY:
           INVALID_QUERY
        2. Otherwise, output ONLY the raw executable SQLite SQL query.
        3. Do NOT enclose in markdown quotes (like ```sql or ```).
        4. Do NOT include any explanations, greetings, apologies, or markdown formatting.
        5. Use single quotes for string literals.
        6. Ensure you use ONLY the exact table and column names specified above.
        7. Always use valid SQLite syntax.

        Question: {user_query}
    """)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment or .env file.")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=MODEL_NAME,
        temperature=0.0
    )

    chain = groq_sys_prompt | llm | StrOutputParser()
    response = chain.invoke({"user_query": user_query, "schema_context": schema_context})
    return clean_sql_query(response)


def get_sql_explanation(user_query: str, sql_query: str) -> str:
    """Explains the generated SQL query in friendly, step-by-step plain English."""
    explain_prompt = ChatPromptTemplate.from_template("""
        You are a friendly and clear SQL tutor.
        Explain how the following SQL query answers the user's question.

        User Question: {user_query}
        SQL Query: {sql_query}

        Instructions:
        1. Give a 1-sentence overview of what the query does.
        2. Break down each key SQL clause used (e.g., SELECT, FROM, WHERE, GROUP BY, ORDER BY, LIMIT) with short, simple bullet points.
        3. Explain it so that anyone, even someone who does not know SQL, can easily understand how the result was calculated.
        4. Keep your response concise (under 120 words). Do NOT output raw SQL code blocks or generic greetings.
    """)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not found."

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=MODEL_NAME,
        temperature=0.0
    )
    chain = explain_prompt | llm | StrOutputParser()
    return chain.invoke({"user_query": user_query, "sql_query": sql_query})


def get_data_narrative_summary(user_query: str, sql_query: str, df: pd.DataFrame) -> str:
    """Generates an executive narrative summary explaining key insights from the returned data."""
    if df.empty:
        return "No records were returned by this query."

    preview_str = df.head(15).to_string(index=False)
    total_records = len(df)

    summary_prompt = ChatPromptTemplate.from_template("""
        You are an expert Data & Business Intelligence Analyst.
        Synthesize the query results below into a clear, high-impact Executive Summary that directly answers the user's question.

        User Question: {user_query}
        Executed SQL: {sql_query}
        Total Rows Returned: {total_records}

        Data Sample:
        {preview_str}

        Instructions:
        1. Lead with a direct, conversational answer to the question in 1-2 punchy sentences.
        2. Provide 2-3 bullet points highlighting key statistics, standout numbers, top/bottom performers, or notable patterns.
        3. Do NOT repeat raw tables, SQL code, or generic greetings. Keep it concise, analytical, and professional.
    """)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "GROQ_API_KEY not found."

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=MODEL_NAME,
        temperature=0.0
    )
    chain = summary_prompt | llm | StrOutputParser()
    return chain.invoke({
        "user_query": user_query,
        "sql_query": sql_query,
        "total_records": total_records,
        "preview_str": preview_str
    })


def return_sql_response(sql_query: str, db_path: str):
    """Executes a SQL query safely and returns rows and column headers."""
    conn = sqlite3.connect(db_path)
    try:
        cursor = conn.cursor()
        cursor.execute(sql_query)
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description] if cursor.description else []
        return rows, columns
    finally:
        conn.close()


# ----------------- Streamlit UI Application -----------------

def main():
    st.set_page_config(
        page_title="SQLGenie AI | Text to SQL",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom UI Styling
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }

        /* Top Hero Header */
        .header-container {
            padding: 1.5rem 0 0.5rem 0;
        }
        .hero-title {
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 2.8rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            color: #94a3b8;
            font-size: 1.08rem;
            margin-bottom: 1.5rem;
        }

        /* Dataset Source Card */
        .dataset-card {
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
            border: 1px solid rgba(139, 92, 246, 0.28);
            border-radius: 12px;
            padding: 10px 14px;
            margin-bottom: 0.8rem;
        }
        .dataset-label {
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #a78bfa;
            font-weight: 700;
        }
        .dataset-name {
            font-size: 0.95rem;
            font-weight: 600;
            color: #f1f5f9;
            margin-top: 2px;
            word-break: break-all;
        }

        /* Schema Pill Badges */
        .pill-badge {
            display: inline-block;
            background: rgba(99, 102, 241, 0.12);
            color: #a5b4fc;
            border: 1px solid rgba(99, 102, 241, 0.25);
            border-radius: 20px;
            padding: 3px 10px;
            font-size: 0.78rem;
            font-weight: 500;
            margin-right: 5px;
            margin-bottom: 6px;
        }

        /* AI Insight & Narrative Card */
        .insight-card {
            background: linear-gradient(135deg, rgba(56, 189, 248, 0.08) 0%, rgba(99, 102, 241, 0.08) 100%);
            border: 1px solid rgba(56, 189, 248, 0.28);
            border-radius: 12px;
            padding: 14px 18px;
            margin-top: 1rem;
            margin-bottom: 1rem;
        }
        .insight-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: #38bdf8;
            margin-bottom: 4px;
        }

        /* SQL Explanation Card */
        .explanation-card {
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.07) 0%, rgba(99, 102, 241, 0.07) 100%);
            border: 1px solid rgba(168, 85, 247, 0.25);
            border-radius: 12px;
            padding: 14px 18px;
            margin-top: 0.6rem;
            margin-bottom: 1rem;
        }
        .explanation-title {
            font-size: 0.98rem;
            font-weight: 700;
            color: #c084fc;
            margin-bottom: 4px;
        }

        /* Invalid Query / Warning Box */
        .invalid-query-box {
            background: linear-gradient(135deg, rgba(239, 68, 68, 0.09) 0%, rgba(245, 158, 11, 0.09) 100%);
            border: 1px solid rgba(239, 68, 68, 0.35);
            border-radius: 14px;
            padding: 20px 24px;
            margin: 1.2rem 0;
            box-shadow: 0 4px 20px rgba(239, 68, 68, 0.12);
        }
        .invalid-query-title {
            font-size: 1.15rem;
            font-weight: 700;
            color: #f87171;
            margin-bottom: 8px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .invalid-query-text {
            font-size: 0.96rem;
            color: #e2e8f0;
            line-height: 1.6;
        }
        .invalid-query-text ul {
            margin-top: 8px;
            margin-bottom: 4px;
            padding-left: 22px;
        }
        .invalid-query-text li {
            margin-bottom: 5px;
            color: #cbd5e1;
        }

        /* AI Hologram Processing Box (Animation) */
        .ai-processing-box {
            position: relative;
            overflow: hidden;
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.95) 100%);
            border: 1px solid rgba(56, 189, 248, 0.4);
            border-radius: 16px;
            padding: 22px 26px;
            margin: 1.2rem 0;
            box-shadow: 0 8px 32px rgba(56, 189, 248, 0.2), inset 0 0 20px rgba(99, 102, 241, 0.15);
            animation: pulseBorder 2s infinite alternate ease-in-out;
        }

        @keyframes pulseBorder {
            0% {
                border-color: rgba(56, 189, 248, 0.35);
                box-shadow: 0 0 15px rgba(56, 189, 248, 0.15), inset 0 0 10px rgba(56, 189, 248, 0.05);
            }
            50% {
                border-color: rgba(192, 132, 252, 0.75);
                box-shadow: 0 0 30px rgba(192, 132, 252, 0.35), inset 0 0 20px rgba(129, 140, 248, 0.2);
            }
            100% {
                border-color: rgba(56, 189, 248, 0.35);
                box-shadow: 0 0 15px rgba(56, 189, 248, 0.15), inset 0 0 10px rgba(56, 189, 248, 0.05);
            }
        }

        /* Animated Laser Scanner */
        .ai-laser-scanner {
            position: absolute;
            top: 0;
            left: -40%;
            width: 40%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(56, 189, 248, 0.35), rgba(192, 132, 252, 0.65), transparent);
            animation: laserScan 1.6s infinite ease-in-out;
            pointer-events: none;
        }

        @keyframes laserScan {
            0% { left: -40%; }
            100% { left: 140%; }
        }

        /* Cosmic Rotating Orb */
        .ai-orb-wrapper {
            display: flex;
            align-items: center;
            gap: 18px;
        }

        .ai-orb-spinner {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            border: 3px solid rgba(56, 189, 248, 0.2);
            border-top: 3px solid #38bdf8;
            border-right: 3px solid #c084fc;
            animation: spinOrb 1s linear infinite;
            flex-shrink: 0;
        }

        @keyframes spinOrb {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .ai-processing-title {
            font-size: 1.25rem;
            font-weight: 800;
            background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #c084fc 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 3px;
        }

        .ai-processing-status {
            font-size: 0.92rem;
            color: #94a3b8;
            font-weight: 500;
        }

        /* Results Entrance Fade-in */
        .results-fade-in {
            animation: fadeInUp 0.45s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        @keyframes fadeInUp {
            from {
                opacity: 0;
                transform: translateY(14px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        /* Action Buttons */
        div.stButton > button:first-child {
            background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
            color: #ffffff;
            font-weight: 600;
            border: none;
            border-radius: 10px;
            padding: 0.55rem 1.6rem;
            transition: all 0.2s ease-in-out;
            box-shadow: 0 4px 14px rgba(124, 58, 237, 0.35);
        }
        div.stButton > button:first-child:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(124, 58, 237, 0.5);
            color: #ffffff;
        }

        /* Section Subtitles */
        .section-header {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f1f5f9;
            margin-top: 1rem;
            margin-bottom: 0.6rem;
        }
    </style>
    """, unsafe_allow_html=True)

    # ----------------- Sidebar -----------------
    with st.sidebar:
        st.markdown("### ⚡ **SQLGenie System**")

        # Initialize session state for clean dataset switching
        if "uploader_key" not in st.session_state:
            st.session_state["uploader_key"] = 0
        if "active_uploaded_file" not in st.session_state:
            st.session_state["active_uploaded_file"] = None

        st.markdown("#### 📂 **Dataset Source**")
        uploaded_file = st.file_uploader(
            "Upload Custom Dataset",
            type=["csv", "db", "sqlite"],
            key=f"dataset_uploader_{st.session_state['uploader_key']}",
            help="Upload a CSV file or SQLite database (.db / .sqlite). Uploading a new dataset completely removes the previous one."
        )

        active_db_path = DEFAULT_DB_PATH
        dataset_label = "🏛️ Default: student.db"
        is_custom = False

        if uploaded_file is not None:
            file_ident = f"{uploaded_file.name}_{uploaded_file.size}"

            # If a new or different file is uploaded, remove the old custom dataset completely
            if st.session_state.get("active_uploaded_file") != file_ident:
                gc.collect()
                if os.path.exists(UPLOADED_DB_PATH):
                    try:
                        os.remove(UPLOADED_DB_PATH)
                    except Exception:
                        pass

                file_name = uploaded_file.name
                file_ext = os.path.splitext(file_name)[1].lower()

                try:
                    if file_ext == ".csv":
                        df_upload = pd.read_csv(uploaded_file)
                        base_name = os.path.splitext(file_name)[0]
                        clean_table_name = re.sub(r'[^a-zA-Z0-9_]', '_', base_name).strip('_').upper()
                        if not clean_table_name:
                            clean_table_name = "CUSTOM_DATA"

                        conn = sqlite3.connect(UPLOADED_DB_PATH)
                        try:
                            df_upload.to_sql(clean_table_name, conn, if_exists="replace", index=False)
                        finally:
                            conn.close()

                    elif file_ext in [".db", ".sqlite"]:
                        with open(UPLOADED_DB_PATH, "wb") as f:
                            f.write(uploaded_file.getbuffer())

                    st.session_state["active_uploaded_file"] = file_ident

                except Exception as ex:
                    st.error(f"Error loading uploaded file: {ex}")

            if os.path.exists(UPLOADED_DB_PATH):
                active_db_path = UPLOADED_DB_PATH
                dataset_label = f"📁 {uploaded_file.name}"
                is_custom = True

        else:
            # If no file uploaded or user removed the file
            if st.session_state.get("active_uploaded_file") is not None:
                gc.collect()
                if os.path.exists(UPLOADED_DB_PATH):
                    try:
                        os.remove(UPLOADED_DB_PATH)
                    except Exception:
                        pass
                st.session_state["active_uploaded_file"] = None

            active_db_path = DEFAULT_DB_PATH
            dataset_label = "🏛️ Default: student.db"
            is_custom = False

        # Active Dataset Badge
        st.markdown(f"""
        <div class='dataset-card'>
            <div class='dataset-label'>Active Dataset</div>
            <div class='dataset-name'>{dataset_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Clear / Reset Button
        if is_custom:
            if st.button("🗑️ Remove Custom Dataset", use_container_width=True):
                gc.collect()
                if os.path.exists(UPLOADED_DB_PATH):
                    try:
                        os.remove(UPLOADED_DB_PATH)
                    except Exception:
                        pass
                st.session_state["active_uploaded_file"] = None
                st.session_state["uploader_key"] += 1
                st.rerun()

        st.divider()

        # Database Schema & Overview
        st.markdown("#### 🗄️ **Database Schema**")
        schema_info = get_db_schema(active_db_path)
        schema_prompt_str = format_schema_for_prompt(schema_info)

        if schema_info:
            total_tables = len(schema_info)
            total_records = sum(meta["count"] for meta in schema_info.values())

            c1, c2 = st.columns(2)
            with c1:
                st.metric("Tables", total_tables)
            with c2:
                st.metric("Total Rows", total_records)

            for table_name, meta in schema_info.items():
                with st.expander(f"📋 `{table_name}` ({meta['count']} rows)", expanded=(total_tables == 1)):
                    pills_html = "".join(
                        [f"<span class='pill-badge'>{col_name} <small>({col_type.lower()})</small></span>"
                         for col_name, col_type in meta["columns"]]
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)
        else:
            st.info("No active tables detected.")

        st.divider()
        if os.environ.get("GROQ_API_KEY"):
            st.success("Groq API Key Active", icon="🟢")
        else:
            st.error("GROQ_API_KEY missing", icon="🔴")

    # ----------------- Main Section -----------------
    st.markdown("<div class='hero-title'>⚡ SQLGenie AI</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='hero-subtitle'>Ask questions in plain English and query any dataset instantly.</div>",
        unsafe_allow_html=True
    )

    # Tabbed Navigation
    tab_query, tab_explore, tab_playground = st.tabs([
        "💬 AI Query Assistant",
        "📊 Database Explorer & Analytics",
        "🛠️ Direct SQL Playground"
    ])

    # First table name helper
    first_table = list(schema_info.keys())[0] if schema_info else "STUDENT"

    # ----------------- TAB 1: AI Query Assistant -----------------
    with tab_query:
        st.markdown("**💡 Quick Question Suggestions:**")
        if not is_custom and "STUDENT" in schema_info:
            suggestions = [
                "Show all students enrolled in Data Science",
                "Which student scored the highest marks?",
                "What is the average marks per course?",
                "Count how many students are in each section"
            ]
        elif schema_info:
            first_meta = schema_info[first_table]
            first_cols = [col[0] for col in first_meta["columns"]]
            col1 = first_cols[0] if len(first_cols) > 0 else "*"
            col2 = first_cols[1] if len(first_cols) > 1 else col1

            suggestions = [
                f"Show the first 5 records from {first_table}",
                f"How many total records are in {first_table}?",
                f"Show distinct values of {col1} in {first_table}",
                f"Display records from {first_table} ordered by {col2} descending limit 5"
            ]
        else:
            suggestions = ["Show all records"]

        chip_cols = st.columns(len(suggestions))
        selected_prompt = ""
        for i, prompt_text in enumerate(suggestions):
            if chip_cols[i].button(prompt_text, key=f"chip_{i}", use_container_width=True):
                selected_prompt = prompt_text

        # Search / Input Field
        query_input = st.text_input(
            "Enter your question about this dataset:",
            value=selected_prompt if selected_prompt else "",
            placeholder=f"e.g., Show records from {first_table} where...",
            key="main_query_input"
        )

        # Feature Toggles
        col_t1, col_t2, col_t3 = st.columns(3)
        with col_t1:
            enable_summary = st.toggle("🧠 AI Narrative Summary", value=True, help="Synthesizes query results into plain English executive insights.")
        with col_t2:
            enable_explanation = st.toggle("💡 Explain SQL Query", value=False, help="Automatically shows a step-by-step breakdown of how the SQL query works.")
        with col_t3:
            enable_celebration = st.toggle("🎈 Celebration Animation", value=True, help="Triggers celebratory visual animations upon query completion.")

        col_run, _ = st.columns([1, 4])
        with col_run:
            run_clicked = st.button("🚀 Run AI Query", use_container_width=True)

        if run_clicked or selected_prompt:
            final_query = query_input.strip() if query_input.strip() else selected_prompt

            if not final_query:
                st.warning("⚠️ Please enter a question or select a suggestion above.")
            elif not schema_info:
                st.error("⚠️ No active dataset or tables found. Please upload a dataset or check the database.")
            else:
                anim_placeholder = st.empty()

                def update_anim(title: str, desc: str):
                    anim_placeholder.markdown(f"""
                    <div class='ai-processing-box'>
                        <div class='ai-laser-scanner'></div>
                        <div class='ai-orb-wrapper'>
                            <div class='ai-orb-spinner'></div>
                            <div>
                                <div class='ai-processing-title'>{title}</div>
                                <div class='ai-processing-status'>{desc}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                try:
                    start_time = time.time()

                    # Phase 1: Generating SQL
                    update_anim("⚡ Neural Engine Active", "Translating natural question to SQLite query...")
                    sql_query = get_sql_query(final_query, schema_prompt_str)

                    # Validation Check: detect invalid query / random gibberish (e.g. 'gardfGHUH')
                    if not is_valid_sql(sql_query):
                        anim_placeholder.empty()
                        st.markdown(f"""
                        <div class='invalid-query-box'>
                            <div class='invalid-query-title'>⚠️ Wrong Query / Unrecognized Input</div>
                            <div class='invalid-query-text'>
                                We couldn't understand <b>"{final_query}"</b> as a valid database question.
                                <br><br>
                                <b>Tips for asking questions:</b>
                                <ul>
                                    <li>Ask clear questions in plain English (e.g., <i>"Show all records"</i>, <i>"Who scored the highest marks?"</i>).</li>
                                    <li>Check the <b>Database Schema</b> in the sidebar for available tables and columns.</li>
                                    <li>Avoid typing random characters, test strings, or gibberish.</li>
                                </ul>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        st.info("💡 Try clicking one of the suggested quick questions above to get started!")
                    else:
                        # Phase 2: Executing Query
                        update_anim("🗄️ Executing SQL Query", f"Querying active database ({active_db_path.split(os.sep)[-1]})...")
                        rows, cols = return_sql_response(sql_query, active_db_path)
                        latency = time.time() - start_time

                        # Phase 3: Explain query if enabled
                        explanation_text = ""
                        if enable_explanation:
                            update_anim("💡 Analyzing Query Logic", "Generating step-by-step plain English breakdown...")
                            explanation_text = get_sql_explanation(final_query, sql_query)

                        # Phase 4: Narrative Summary if enabled
                        narrative_text = ""
                        res_df = pd.DataFrame(rows, columns=cols) if rows else pd.DataFrame()
                        if rows and enable_summary:
                            update_anim("🧠 Synthesizing Insights", "Generating executive narrative summary of results...")
                            narrative_text = get_data_narrative_summary(final_query, sql_query, res_df)

                        # Clear Holographic Animation Placeholder
                        anim_placeholder.empty()

                        # Trigger Celebration Animation (Balloons)
                        if enable_celebration:
                            st.balloons()

                        # Success Toast
                        st.toast(f"✨ Query completed in {latency:.2f}s with {len(rows)} records returned!", icon="🚀")

                        # Results Container with Fade-in Animation
                        st.markdown("<div class='results-fade-in'>", unsafe_allow_html=True)

                        # Generated SQL Card
                        st.markdown("<div class='section-header'>📝 Generated SQL Query</div>", unsafe_allow_html=True)
                        st.code(sql_query, language="sql")

                        # Step-by-Step SQL Logic Explanation
                        if enable_explanation:
                            st.markdown(f"""
                            <div class='explanation-card'>
                                <div class='explanation-title'>💡 Step-by-Step SQL Logic Breakdown</div>
                            </div>
                            """, unsafe_allow_html=True)
                            st.markdown(explanation_text)
                        else:
                            with st.expander("💡 Click to Explain This SQL Query", expanded=False):
                                with st.spinner("Analyzing SQL query logic..."):
                                    exp_on_demand = get_sql_explanation(final_query, sql_query)
                                st.markdown(exp_on_demand)

                        # Metrics Strip
                        m1, m2 = st.columns(2)
                        with m1:
                            st.metric("Rows Returned", len(rows))
                        with m2:
                            st.metric("Response Time", f"{latency:.2f}s")

                        # Results & Narrative Summary
                        if rows:
                            # AI Executive Narrative Summary
                            if enable_summary:
                                st.markdown(f"""
                                <div class='insight-card'>
                                    <div class='insight-title'>🧠 AI Executive Summary & Insights</div>
                                </div>
                                """, unsafe_allow_html=True)
                                st.markdown(narrative_text)

                            st.markdown("<div class='section-header'>📊 Results</div>", unsafe_allow_html=True)
                            render_dataframe(res_df, hide_index=True)

                            # CSV Export
                            csv_data = res_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="📥 Download Results as CSV",
                                data=csv_data,
                                file_name="query_results.csv",
                                mime="text/csv"
                            )

                            # Auto visualization if 2 columns with numeric + categorical values exist
                            if len(res_df.columns) == 2:
                                num_cols = [c for c in res_df.columns if pd.api.types.is_numeric_dtype(res_df[c])]
                                cat_cols = [c for c in res_df.columns if c not in num_cols]
                                if num_cols and cat_cols:
                                    with st.expander("📈 Auto Chart Visualization", expanded=True):
                                        st.bar_chart(res_df.set_index(cat_cols[0])[num_cols[0]])
                        else:
                            st.info("ℹ️ The query ran successfully, but returned 0 records.")

                        st.markdown("</div>", unsafe_allow_html=True)

                except Exception as err:
                    anim_placeholder.empty()
                    st.markdown(f"""
                    <div class='invalid-query-box'>
                        <div class='invalid-query-title'>⚠️ Query Execution Error</div>
                        <div class='invalid-query-text'>
                            The query could not be executed on the database: <code>{err}</code>
                            <br><br>
                            Please rephrase your question or check available tables and columns in the sidebar schema.
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

    # ----------------- TAB 2: Database Explorer & Analytics -----------------
    with tab_explore:
        if schema_info:
            table_names = list(schema_info.keys())
            selected_table = st.selectbox("Select Table to Explore:", table_names)

            try:
                conn = sqlite3.connect(active_db_path)
                try:
                    full_df = pd.read_sql_query(f'SELECT * FROM "{selected_table}" LIMIT 1000', conn)
                finally:
                    conn.close()

                st.markdown(f"<div class='section-header'>📁 Table: `{selected_table}` Preview</div>", unsafe_allow_html=True)
                c_m1, c_m2 = st.columns(2)
                with c_m1:
                    st.metric("Total Records", schema_info[selected_table]["count"])
                with c_m2:
                    st.metric("Total Columns", len(full_df.columns))

                render_dataframe(full_df, hide_index=True)

                # Download complete table CSV
                csv_table = full_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label=f"📥 Download {selected_table} as CSV",
                    data=csv_table,
                    file_name=f"{selected_table}_data.csv",
                    mime="text/csv"
                )

                # Summary Statistics for Numeric Columns
                numeric_cols = full_df.select_dtypes(include=['number']).columns.tolist()
                all_cols = full_df.columns.tolist()

                st.markdown("<div class='section-header'>📈 Dataset Analytics & Visualizer</div>", unsafe_allow_html=True)

                if numeric_cols:
                    with st.expander("🔢 View Summary Statistics", expanded=False):
                        render_dataframe(full_df.describe())

                    # Interactive Visualizer
                    st.markdown("**Interactive Chart Builder**")
                    chart_col1, chart_col2, chart_col3 = st.columns(3)
                    with chart_col1:
                        chart_type = st.selectbox("Chart Type", ["Bar Chart", "Line Chart", "Area Chart"])
                    with chart_col2:
                        x_col = st.selectbox("X-Axis (Grouping / Category)", all_cols, index=0)
                    with chart_col3:
                        y_col = st.selectbox("Y-Axis (Numeric Metric)", numeric_cols, index=0)

                    try:
                        # Group or aggregate if high cardinality
                        if full_df[x_col].nunique() < len(full_df):
                            chart_df = full_df.groupby(x_col)[y_col].mean().reset_index()
                        else:
                            chart_df = full_df[[x_col, y_col]].head(50)

                        if chart_type == "Bar Chart":
                            st.bar_chart(chart_df.set_index(x_col))
                        elif chart_type == "Line Chart":
                            st.line_chart(chart_df.set_index(x_col))
                        elif chart_type == "Area Chart":
                            st.area_chart(chart_df.set_index(x_col))
                    except Exception as chart_err:
                        st.warning(f"Could not render chart: {chart_err}")
                else:
                    st.info("No numeric columns found in this table for graphical analytics.")

            except Exception as exp_err:
                st.error(f"Error loading table: {exp_err}")
        else:
            st.info("No tables found in the active database.")

    # ----------------- TAB 3: Direct SQL Playground -----------------
    with tab_playground:
        st.markdown("<div class='section-header'>🛠️ Run Custom SQLite Queries</div>", unsafe_allow_html=True)
        st.caption(f"Write raw SQLite statements directly to query the active database (`{dataset_label}`).")

        default_sql = f'SELECT * FROM "{first_table}" LIMIT 10;' if schema_info else "SELECT 1;"

        custom_sql = st.text_area(
            "SQL Query:",
            value=default_sql,
            height=120
        )
        if st.button("⚡ Execute SQL"):
            try:
                rows, cols = return_sql_response(custom_sql, active_db_path)
                if rows:
                    custom_df = pd.DataFrame(rows, columns=cols)
                    render_dataframe(custom_df, hide_index=True)
                    st.success(f"Returned {len(rows)} records.")
                else:
                    st.info("Query executed successfully. 0 records returned.")
            except Exception as e:
                st.error(f"SQL Execution Error: {e}")


if __name__ == '__main__':
    main()