from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key  = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-5", temperature=0.2,api_key=api_key)


def generate_code(state):
    """
    Generate Python code (PySpark or Pandas) to transform data from source to target.
    Uses schema-qualified table names.
    """
    sttm = state.get("structured_request", "")
    table_type = state.get("table_type")
    scd_type = state.get("scd_type")
    source_preview = state.get("source_preview", {})
    target_preview = state.get("target_preview", {})

    # Use schema-qualified table names
    source_table = state.get("source_table")  # "schema.table"
    target_table = state.get("target_table")  # "schema.table"

    src_cols = source_preview.get("columns", [])
    tgt_cols = target_preview.get("columns", [])

    # Determine SCD logic
    scd_logic = ""
    if table_type == "Dimension Table" and scd_type:
        scd_logic = {
            "SCD1": "Overwrite existing records (no history).",
            "SCD2": "Expire old records and insert new ones with start_date/end_date.",
            "SCD3": "Maintain both current and previous value columns."
        }.get(scd_type, "")
    else:
        scd_logic = "Typical Fact Table joins and aggregations."

    # Construct prompt for LLM
    prompt = f"""
You are a senior data engineer. Generate Python code to transform data 
from source table to target table using the STTM below.

STTM:
{sttm}

Table Type: {table_type}
SCD Type: {scd_type or 'N/A'}
Logic: {scd_logic}

Source Table: {source_table}
Target Table: {target_table}

Source Columns: {src_cols}
Target Columns: {tgt_cols}

Requirements:
- Always produce Python code (PySpark or Pandas)
- Read from source_df, write to target_df
- Apply transformations as per STTM
- No explanations, only clean code
- Include schema-qualified table names when reading/writing
"""

    code = llm.invoke(prompt).strip()
    return {"generated_code": code}
