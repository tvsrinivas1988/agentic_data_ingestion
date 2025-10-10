import streamlit as st
from graph.codegen_flow import app as codegen_graph
from utils.db import list_tables, preview_table
import pandas as pd

st.set_page_config(page_title="Agentic Code Generator", page_icon="🤖", layout="wide")

st.title("💬 Agentic Code Generator")
st.markdown(
    """
    This chatbot generates **PySpark or Pandas** transformation code based on your uploaded STTM and selected database tables.
    It previews both source and target tables before generating the code.
    """
)

# Step 1: Upload STTM
uploaded_sttm = st.file_uploader("📄 Upload your STTM file (CSV or TXT)", type=["csv", "txt"])
structured_request = ""
if uploaded_sttm is not None:
    structured_request = uploaded_sttm.read().decode("utf-8")

# Step 2: Select Fact / Dimension and SCD type
table_type = st.selectbox("🧱 Select Table Type", ["Fact Table", "Dimension Table"])
scd_type = None
if table_type == "Dimension Table":
    scd_type = st.selectbox("📜 Select SCD Type", ["SCD1", "SCD2", "SCD3"])

# Step 3: Choose source & target tables from database
st.subheader("🔗 Select Source and Target Tables from Database")

tables = list_tables()
source_table = st.selectbox("Source Table", tables, key="src")
target_table = st.selectbox("Target Table", tables, key="tgt")

# Step 4: Preview data (before generating code)
if source_table and target_table:
    st.subheader("👀 Preview Data (First 5 Rows)")

    # Split into two columns
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**📘 Source Table:** `{source_table}`")
        try:
            src_preview = preview_table(source_table)
            src_df = pd.DataFrame(src_preview["sample_rows"])
            st.dataframe(src_df, use_container_width=True)
            st.caption(f"Columns: {list(src_df.columns)}")
        except Exception as e:
            st.error(f"Error previewing source table: {e}")

    with col2:
        st.markdown(f"**📗 Target Table:** `{target_table}`")
        try:
            tgt_preview = preview_table(target_table)
            tgt_df = pd.DataFrame(tgt_preview["sample_rows"])
            st.dataframe(tgt_df, use_container_width=True)
            st.caption(f"Columns: {list(tgt_df.columns)}")
        except Exception as e:
            st.error(f"Error previewing target table: {e}")

# Step 5: Generate code
st.markdown("---")
if st.button("🚀 Generate Code"):
    if not structured_request:
        st.warning("Please upload your STTM file first.")
    else:
        st.info("Fetching schema and generating transformation code...")
        state = {
            "structured_request": structured_request,
            "table_type": table_type,
            "scd_type": scd_type,
            "source_table": source_table,
            "target_table": target_table,
        }

        with st.spinner("Running agent workflow..."):
            result = codegen_graph.invoke(state)
            code = result.get("generated_code", "")

        if code:
            st.success("✅ Code generated successfully!")
            st.code(code, language="python")
        else:
            st.error("⚠️ No code generated. Please check your STTM or DB connection.")
