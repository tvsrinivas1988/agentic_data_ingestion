import streamlit as st
import pandas as pd
import json
import sys
import os
import tempfile
import importlib.util
from dotenv import load_dotenv

# Add parent folder for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.db import (
    list_schemas,
    list_tables,
    preview_table,
    get_existing_code,
    save_generated_code,
    get_engine
)
from graph.codegen_flow import generate_sttm_node, codegen_node_func

# ---------------------------------------------
# Streamlit UI Setup
# ---------------------------------------------
st.set_page_config(page_title="CodeFlow AI", layout="wide")
st.title("💬 CodeFlow AI")

# ---------------------------------------------
# Initialize session state
# ---------------------------------------------
state_keys = [
    "sttm_generated", "final_sttm", "generated_code",
    "etl_engine", "load_type", "step1_done", "step2_done",
    "step3_done", "step4_done"
]
for key in state_keys:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------
# Sidebar Navigation
# ---------------------------------------------
st.sidebar.title("Workflow Steps")
steps = [
    "1️⃣ Select Tables",
    "2️⃣ Generate STTM",
    "3️⃣ Generate ETL Code",
    "4️⃣ Persist STTM & Code",
    "5️⃣ Execute ETL"
]

completed = [
    st.session_state.step1_done,
    st.session_state.step2_done,
    st.session_state.step3_done,
    st.session_state.step4_done,
    False
]

# Sidebar selection: disable future steps until previous is completed
for i, step in enumerate(steps):
    if i == 0 or completed[i-1]:
        st.sidebar.button(step, key=f"sidebar_step_{i}")

# ---------------------------------------------
# STEP 1: Source & Target Selection
# ---------------------------------------------
with st.expander("Step 1: Select Source & Target Tables", expanded=True):
    table_type = st.selectbox("Table Type", ["Fact Table", "Dimension Table"])
    scd_type = None
    if table_type == "Dimension Table":
        scd_type = st.selectbox("SCD Type", ["SCD1", "SCD2", "SCD3"])

    schemas = list_schemas()
    source_schema = st.selectbox("Source Schema", schemas)
    target_schema = st.selectbox("Target Schema", schemas)
    source_tables = list_tables(source_schema)
    target_tables = list_tables(target_schema)

    source_table = st.selectbox("Source Table", source_tables)
    target_table = st.selectbox("Target Table", target_tables)

    load_type = st.selectbox("Load Type", ["Full", "Incremental"])
    st.session_state.load_type = load_type

    if st.button("➡️ Proceed to Step 2"):
        st.session_state.step1_done = True

# ---------------------------------------------
# STEP 2: Generate / Review STTM
# ---------------------------------------------
if st.session_state.step1_done:
    with st.expander("Step 2: Generate / Review STTM", expanded=True):
        engine = get_engine()
        existing_sttm = get_existing_code(source_schema, source_table, target_schema, target_table, engine)

        if existing_sttm and not st.session_state.final_sttm:
            st.session_state.final_sttm = existing_sttm.get("sttm")
            st.session_state.generated_code = existing_sttm.get("generated_code")
            st.session_state.sttm_generated = True

        if st.button("🤖 Generate Draft STTM (LLM + Audit Columns)"):
            with st.spinner("Generating STTM using LLM..."):
                state = {
                    "source_schema": source_schema,
                    "source_table": source_table,
                    "target_schema": target_schema,
                    "target_table": target_table
                }
                state = generate_sttm_node(state)
                draft_sttm = state.get("sttm")

                if draft_sttm:
                    st.session_state.final_sttm = draft_sttm
                    st.session_state.sttm_generated = True
                    st.success("✅ STTM generated successfully!")
                else:
                    st.error("⚠️ Failed to generate STTM.")

        if st.session_state.sttm_generated:
            edited_sttm_str = st.text_area(
                "STTM JSON",
                json.dumps(st.session_state.final_sttm, indent=2),
                height=400
            )
            try:
                st.session_state.final_sttm = json.loads(edited_sttm_str)
            except Exception:
                st.error("⚠️ Invalid JSON. Please correct before proceeding.")

            if st.button("➡️ Proceed to Step 3"):
                st.session_state.step2_done = True

# ---------------------------------------------
# STEP 3: Generate Python ETL Code
# ---------------------------------------------
if st.session_state.step2_done:
    with st.expander("Step 3: Generate Python ETL Code", expanded=True):
        if st.button("🚀 Generate ETL Code"):
            with st.spinner("Generating ETL code using .env settings..."):
                load_dotenv()
                db_env = {
                    "user": os.getenv("DB_USER"),
                    "host": os.getenv("DB_HOST"),
                    "port": os.getenv("DB_PORT"),
                    "database": os.getenv("DB_NAME")
                }

                codegen_state = {
                    "sttm": st.session_state.final_sttm,
                    "table_type": table_type,
                    "scd_type": scd_type,
                    "etl_engine": "Pandas",
                    "load_type": st.session_state.load_type,
                    "db_env": db_env
                }

                codegen_state = codegen_node_func(codegen_state, preview=False)
                st.session_state.generated_code = codegen_state.get("code")

                if st.session_state.generated_code:
                    st.success("✅ ETL code generated successfully!")
                    st.code(st.session_state.generated_code, language="python")
                    st.session_state.step3_done = True

# ---------------------------------------------
# STEP 4: Persist STTM + ETL Code
# ---------------------------------------------
if st.session_state.step3_done:
    with st.expander("Step 4: Persist STTM + ETL Code", expanded=True):
        if st.button("💾 Persist STTM + Code"):
            try:
                engine = get_engine()
                save_generated_code(
                    engine,
                    source_schema,
                    target_schema,
                    source_table,
                    target_table,
                    st.session_state.generated_code,
                    st.session_state.final_sttm,
                    table_type,
                    scd_type,
                    st.session_state["etl_engine"]
                )
                st.success("✅ STTM and ETL Code saved successfully.")
                st.session_state.step4_done = True
            except Exception as e:
                st.error(f"❌ Error while saving to database: {e}")

# ---------------------------------------------
# STEP 5: Execute Saved ETL
# ---------------------------------------------
if st.session_state.step4_done:
    with st.expander("Step 5: Execute Saved ETL", expanded=True):
        if st.button("▶️ Execute Saved ETL from DB"):
            try:
                engine = get_engine()
                existing_code = get_existing_code(
                    source_schema, source_table, target_schema, target_table, engine
                )
                if not existing_code:
                    st.error("⚠️ No saved code found. Please generate and save first.")
                else:
                    code_to_run = existing_code["generated_code"]

                    with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as tmpfile:
                        tmpfile.write(code_to_run)
                        temp_path = tmpfile.name

                    spec = importlib.util.spec_from_file_location("generated_etl", temp_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    func_name = f"{source_schema}_{target_schema}_{target_table}".lower()
                    df_result = getattr(module, func_name)()

                    if df_result is not None:
                        st.dataframe(df_result.head(20), use_container_width=True)
                    else:
                        query = f"SELECT * FROM {target_schema}.{target_table} LIMIT 20"
                        df_result = pd.read_sql(query, engine)
                        st.dataframe(df_result, use_container_width=True)

                    st.success(f"✅ ETL executed successfully for {target_schema}.{target_table}")
                    os.remove(temp_path)
            except Exception as e:
                st.error(f"❌ Error during ETL execution: {e}")
