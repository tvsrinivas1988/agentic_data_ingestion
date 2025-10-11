import streamlit as st
import pandas as pd
import json
import sys
import os
from dotenv import load_dotenv

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()

# Add parent folder to sys.path so 'utils' can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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

st.set_page_config(page_title="Agentic Code Generator", layout="wide")
st.title("💬 Agentic Code Generator (LLM-driven STTM)")

# --------------------------
# Initialize session state
# --------------------------
for key in ["sttm_generated", "final_sttm", "generated_code",
            "etl_engine", "load_type"]:
    if key not in st.session_state:
        st.session_state[key] = None

# --------------------------
# Table Type + SCD
# --------------------------
table_type = st.selectbox("Table Type", ["Fact Table", "Dimension Table"])
scd_type = None
if table_type == "Dimension Table":
    scd_type = st.selectbox("SCD Type", ["SCD1", "SCD2", "SCD3"])

# --------------------------
# Schema Selection
# --------------------------
schemas = list_schemas()
source_schema = st.selectbox("Source Schema", schemas, key="src_schema")
target_schema = st.selectbox("Target Schema", schemas, key="tgt_schema")

# --------------------------
# Table Selection
# --------------------------
source_tables = list_tables(source_schema)
target_tables = list_tables(target_schema)
source_table = st.selectbox("Source Table", source_tables, key="src_table")
target_table = st.selectbox("Target Table", target_tables, key="tgt_table")

# --------------------------
# Load Type Selection
# --------------------------
load_type = st.selectbox("Load Type", ["Full", "Incremental"], key="load_type")


# --------------------------
# Table Preview Function
# --------------------------
def render_table_preview(schema_name, table_name):
    preview = preview_table(schema_name, table_name)
    columns = [col["column_name"] for col in preview["columns"]]
    if preview["sample_rows"]:
        st.dataframe(pd.DataFrame(preview["sample_rows"]), use_container_width=True)
    else:
        st.table(pd.DataFrame(columns=columns))
    return len(preview["sample_rows"])

col1, col2 = st.columns(2)
with col1:
    st.markdown(f"**Source Table: {source_schema}.{source_table}**")
    src_row_count = render_table_preview(source_schema, source_table)
with col2:
    st.markdown(f"**Target Table: {target_schema}.{target_table}**")
    _ = render_table_preview(target_schema, target_table)

# --------------------------
# ETL Engine Selection
# --------------------------
etl_engine_option = st.radio(
    "Choose ETL engine (or Auto based on row count):",
    ["Pandas", "PySpark", "Auto"],
    index=2
)
if etl_engine_option == "Auto":
    etl_engine = "Pandas" if src_row_count < 1_000_000 else "PySpark"
else:
    etl_engine = etl_engine_option
st.session_state.etl_engine = etl_engine
st.info(f"Selected ETL Engine: {etl_engine}")

# --------------------------
# Initialize SQLAlchemy engine
# --------------------------
engine = get_engine()

# --------------------------
# Check for existing code
# --------------------------
existing_code = get_existing_code(source_schema, source_table, target_schema, target_table, engine)
if existing_code:
    st.info("⚡ Existing ETL code found. Loading into editor...")
    st.session_state.generated_code = existing_code["generated_code"]
    st.session_state.etl_engine = existing_code["etl_engine"]
    if not st.session_state.final_sttm:
        st.session_state.final_sttm = {
            "source_schema": source_schema,
            "source_table": source_table,
            "target_schema": target_schema,
            "target_table": target_table,
            "mappings": []
        }

# --------------------------
# Generate Draft STTM
# --------------------------
if st.button("🤖 Generate Draft STTM (LLM + Audit Columns)"):
    state = {
        "source_schema": source_schema,
        "source_table": source_table,
        "target_schema": target_schema,
        "target_table": target_table
    }
    with st.spinner("Generating draft STTM..."):
        state = generate_sttm_node(state)
        draft_sttm = state.get("sttm", {})
        if draft_sttm:
            st.session_state.sttm_generated = draft_sttm
            st.session_state.final_sttm = draft_sttm
        else:
            st.error("⚠️ STTM generation failed.")

# --------------------------
# Display and Edit STTM
# --------------------------
if st.session_state.sttm_generated or st.session_state.final_sttm:
    edited_sttm_str = st.text_area(
        "Review/Edit STTM JSON (Audit & Process Columns Included)",
        json.dumps(st.session_state.final_sttm, indent=2),
        height=500
    )
    try:
        st.session_state.final_sttm = json.loads(edited_sttm_str)
    except Exception:
        st.error("⚠️ Invalid JSON. Please fix before generating code.")

# --------------------------
# Generate ETL Code (always from latest STTM)
# --------------------------
if st.button("🚀 Generate Python ETL Code") and st.session_state.final_sttm:
    codegen_state = {
        "sttm": st.session_state.final_sttm,
        "table_type": table_type,
        "scd_type": scd_type,
        "etl_engine": st.session_state.etl_engine,
        "load_type": load_type
    }
    with st.spinner("Generating full Python ETL code..."):
        codegen_state = codegen_node_func(codegen_state, preview=False)
        st.session_state.generated_code = codegen_state.get("code")
    st.success("✅ Python ETL code generated and ready for backend execution.")

# --------------------------
# Display Generated Code
# --------------------------
if st.session_state.generated_code:
    st.markdown("### Generated ETL Code")
    st.code(st.session_state.generated_code, language="python")

# --------------------------
# Execute ETL Backend
# --------------------------
if st.session_state.generated_code:
    if st.button("🚀 Execute ETL & Write to Target Table"):
        with st.spinner("Running ETL job on backend..."):
            try:
                local_vars = {}
                exec(st.session_state.generated_code, globals(), local_vars)
                func_name = f"{source_schema}_{target_schema}_{target_table}".lower()
                # Execute ETL in memory
                df_preview = local_vars[func_name]()  # Pass engine inside generated code
                st.success(f"✅ Data successfully written to {target_schema}.{target_table}")
                st.markdown("### Sample Target Data After ETL")
                st.dataframe(df_preview.head(20))
            except Exception as e:
                st.error(f"❌ Error during ETL execution: {e}")

# --------------------------
# Persist Generated Code
# --------------------------
if st.session_state.generated_code:
    if st.button("💾 Persist Generated ETL Code"):
        try:
            save_generated_code(
                source_schema, source_table, target_schema, target_table,
                table_type, scd_type, st.session_state.etl_engine,
                st.session_state.generated_code,
                engine
            )
            st.success("✅ Generated code saved to database.")
        except Exception as e:
            st.error(f"❌ Error saving code: {e}")
