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
st.set_page_config(page_title="Agentic Code Generator", layout="wide")
st.title("💬 Agentic Code Generator (LLM-driven STTM)")

# ---------------------------------------------
# Initialize session state
# ---------------------------------------------
for key in ["sttm_generated", "final_sttm", "generated_code", "etl_engine", "load_type"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ---------------------------------------------
# Step 1: Basic Setup
# ---------------------------------------------
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
##st.session_state.load_type = load_type

# ---------------------------------------------
# Step 2: Generate STTM
# ---------------------------------------------
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

# ---------------------------------------------
# Step 3: Show & Edit STTM (only after generation)
# ---------------------------------------------
if st.session_state.sttm_generated:
    st.subheader("📘 Review / Edit Generated STTM")
    edited_sttm_str = st.text_area(
        "STTM JSON",
        json.dumps(st.session_state.final_sttm, indent=2),
        height=400
    )

    try:
        st.session_state.final_sttm = json.loads(edited_sttm_str)
    except Exception:
        st.error("⚠️ Invalid JSON. Please correct before proceeding.")

# ---------------------------------------------
# Step 4: Generate ETL Code (after STTM is ready)
# ---------------------------------------------
if st.session_state.final_sttm:
    if st.button("🚀 Generate Python ETL Code"):
        with st.spinner("Generating ETL code using .env settings..."):
            # Load .env for DB details
            load_dotenv()
            db_user = os.getenv("DB_USER")
            db_host = os.getenv("DB_HOST")
            db_port = os.getenv("DB_PORT")
            db_name = os.getenv("DB_NAME")

            # Pass all info to codegen
            codegen_state = {
                "sttm": st.session_state.final_sttm,
                "table_type": table_type,
                "scd_type": scd_type,
                "etl_engine": "Pandas",
                "load_type": load_type,
                "db_env": {
                    "user": db_user,
                    "host": db_host,
                    "port": db_port,
                    "database": db_name
                }
            }

            codegen_state = codegen_node_func(codegen_state, preview=False)
            st.session_state.generated_code = codegen_state.get("code")

            if st.session_state.generated_code:
                st.success("✅ ETL code generated successfully!")
                st.code(st.session_state.generated_code, language="python")

# ---------------------------------------------
# Step 5: Save Code & STTM to Database
# ---------------------------------------------
if st.session_state.generated_code:
    if st.button("💾 Persist STTM + ETL Code to Database"):
        try:
            engine = get_engine()
            save_generated_code(
            engine,
            source_schema,
            target_schema,
            source_table,
            target_table,
            st.session_state["generated_code"],
            st.session_state["final_sttm"],
            table_type,
            scd_type,
            st.session_state["etl_engine"]
        )
            st.success("✅ STTM and ETL Code saved to database successfully.")
        except Exception as e:
            st.error(f"❌ Error while saving to database: {e}")

# ---------------------------------------------
# Step 6: Execute Saved Code (from DB)
# ---------------------------------------------
if st.button("▶️ Execute Saved ETL from DB"):
    try:
        engine = get_engine()
        existing_code = get_existing_code(source_schema, source_table, target_schema, target_table, engine)
        if not existing_code:
            st.error("⚠️ No saved code found. Please generate and save first.")
        else:
            code_to_run = existing_code["generated_code"]

            # Write code to temporary file for import
            with tempfile.NamedTemporaryFile("w", delete=False, suffix=".py") as tmpfile:
                tmpfile.write(code_to_run)
                temp_path = tmpfile.name

            spec = importlib.util.spec_from_file_location("generated_etl", temp_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            func_name = f"{source_schema}_{target_schema}_{target_table}".lower()
            df_result = getattr(module, func_name)(engine)

            st.success(f"✅ ETL executed successfully for {target_schema}.{target_table}")
            st.dataframe(df_result.head(20))

    except Exception as e:
        st.error(f"❌ Error during ETL execution: {e}")
