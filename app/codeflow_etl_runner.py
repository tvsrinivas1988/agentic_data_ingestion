import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import traceback
import os
from dotenv import load_dotenv
import time

load_dotenv()

st.set_page_config(page_title="ETL Job Runner", page_icon="⚙️", layout="wide")

# ---- DB Connection ----
db_url = os.getenv("DATABASE_URL")
engine = create_engine(db_url)

st.title("⚙️ ETL Job Runner Dashboard")

# ---- Fetch ETL jobs from audit.etl_code_store ----
with engine.connect() as conn:
    query = text("""
        SELECT id, source_schema, source_table, target_schema, target_table, generated_code, last_modified
        FROM audit.etl_code_store
        ORDER BY source_schema, target_table
    """)
    df = pd.read_sql(query, conn)

if df.empty:
    st.warning("No ETL code found in audit.etl_code_store.")
    st.stop()

# ---- Derive function names ----
df["func_name"] = df.apply(lambda x: f"{x['source_schema']}_{x['target_schema']}_{x['target_table']}", axis=1)

# ---- Multi-select for sequential run ----
st.sidebar.header("Select ETL Jobs to Run Sequentially")
selected_funcs = st.sidebar.multiselect(
    "Choose ETL Jobs (in desired order):",
    df["func_name"].tolist(),
    default=[]
)

if selected_funcs:
    st.info(f"🧩 You selected {len(selected_funcs)} ETL job(s) to run sequentially:")
    for i, func in enumerate(selected_funcs, start=1):
        st.write(f"{i}. {func}")
else:
    st.warning("Select at least one ETL job to enable Run button.")

# ---- Run selected sequentially ----
if st.button("▶️ Run Selected ETLs in Sequence", use_container_width=True, disabled=not selected_funcs):
    progress_bar = st.progress(0)
    log_area = st.empty()

    total_jobs = len(selected_funcs)
    success_count = 0
    failure_count = 0
    log_messages = []

    for i, func_name in enumerate(selected_funcs, start=1):
        st.write(f"🚀 Running {func_name} ({i}/{total_jobs})...")
        start_time = datetime.now()

        try:
            func_row = df[df["func_name"] == func_name].iloc[0]
            code = func_row["generated_code"]

            # Execute function in isolated namespace
            namespace = {}
            exec(code, namespace)

            # Derive actual callable name from schema-table mapping
            callable_name = func_name
            if callable_name in namespace:
                namespace[callable_name]()
            else:
                # fallback if function not defined
                raise ValueError(f"Function '{callable_name}' not found in code block.")

            end_time = datetime.now()
            duration = int((end_time - start_time).total_seconds())
            message = f"✅ {func_name} executed successfully in {duration} sec."

            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO audit.etl_execution_log 
                    (func_name, source_schema, source_table, target_schema, target_table,
                     execution_start, execution_end, status, message, duration_seconds)
                    VALUES (:func_name, :src_schema, :src_table, :tgt_schema, :tgt_table,
                            :start, :end, 'SUCCESS', :message, :duration)
                """), {
                    "func_name": func_name,
                    "src_schema": func_row["source_schema"],
                    "src_table": func_row["source_table"],
                    "tgt_schema": func_row["target_schema"],
                    "tgt_table": func_row["target_table"],
                    "start": start_time,
                    "end": end_time,
                    "message": message,
                    "duration": duration
                })

            success_count += 1
            log_messages.append(message)

        except Exception as e:
            end_time = datetime.now()
            duration = int((end_time - start_time).total_seconds())
            error_msg = traceback.format_exc()
            message = f"❌ {func_name} failed: {str(e)}"

            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO audit.etl_execution_log 
                    (func_name, source_schema, source_table, target_schema, target_table,
                     execution_start, execution_end, status, message, duration_seconds)
                    VALUES (:func_name, :src_schema, :src_table, :tgt_schema, :tgt_table,
                            :start, :end, 'FAILED', :message, :duration)
                """), {
                    "func_name": func_name,
                    "src_schema": func_row["source_schema"],
                    "src_table": func_row["source_table"],
                    "tgt_schema": func_row["target_schema"],
                    "tgt_table": func_row["target_table"],
                    "start": start_time,
                    "end": end_time,
                    "message": error_msg[:4000],
                    "duration": duration
                })

            failure_count += 1
            log_messages.append(message)

        progress_bar.progress(i / total_jobs)
        log_area.text("\n".join(log_messages))
        time.sleep(0.5)

    st.success(f"🎯 Sequence completed — {success_count} succeeded, {failure_count} failed.")
    st.divider()

# ---- Execution Log with Filters ----
st.subheader("🕒 Execution History")

with engine.connect() as conn:
    logs_df = pd.read_sql(
        text("""
            SELECT func_name, status, duration_seconds, message,
                   execution_start, execution_end
            FROM audit.etl_execution_log
            ORDER BY execution_start DESC
            LIMIT 100
        """),
        conn
    )

# --- Filters ---
col1, col2 = st.columns(2)
with col1:
    func_filter = st.selectbox("🔍 Filter by Function", ["All"] + sorted(logs_df["func_name"].unique().tolist()))
with col2:
    status_filter = st.selectbox("📊 Filter by Status", ["All", "SUCCESS", "FAILED"])

filtered_df = logs_df.copy()
if func_filter != "All":
    filtered_df = filtered_df[filtered_df["func_name"] == func_filter]
if status_filter != "All":
    filtered_df = filtered_df[filtered_df["status"] == status_filter]

# --- Add color badges ---
def color_status(status):
    if status == "SUCCESS":
        return "🟢 SUCCESS"
    elif status == "FAILED":
        return "🔴 FAILED"
    else:
        return "⚪ UNKNOWN"

filtered_df["status"] = filtered_df["status"].apply(color_status)

st.dataframe(
    filtered_df[["func_name", "status", "duration_seconds", "execution_start", "execution_end", "message"]],
    use_container_width=True
)
