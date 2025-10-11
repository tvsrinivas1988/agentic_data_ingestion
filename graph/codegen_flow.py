import json
from utils.db import get_table_columns, get_connection
from openai import OpenAI
# -------------------------
# STTM Generator Node
# -------------------------
import json
import openai
from dotenv import load_dotenv
import os
load_dotenv()



from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json

from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json
from utils.db import get_table_columns  # <-- make sure this exists in your utils.db

def generate_sttm_node(state):
    """
    Generates an STTM JSON for ETL mapping from source to target.
    Ensures all target columns are real (from database metadata).
    LLM only infers transformations, not new columns.
    """
    src_schema = state.get("source_schema")
    src_table = state.get("source_table")
    tgt_schema = state.get("target_schema")
    tgt_table = state.get("target_table")

    # Fetch schema metadata from DB
    source_columns = get_table_columns(src_schema, src_table)
    target_columns = get_table_columns(tgt_schema, tgt_table)

    # Extract just the column names for prompt clarity
    src_cols = [c["column_name"] for c in source_columns]
    tgt_cols = [c["column_name"] for c in target_columns]

    # Build LLM model
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.2,api_key=os.getenv('OPENAI_API_KEY'))

    # Strict instruction to use only target columns
    messages = [
        SystemMessage(content=(
            "You are an expert ETL engineer. "
            "Your task is to build a JSON Source-To-Target Mapping (STTM). "
            "You must only use the target columns that exist in the provided schema metadata. "
            "Do not invent or hallucinate new target columns. "
            "You may, however, infer transformations or use constants for audit fields like timestamps."
    
        )),
        HumanMessage(content=f"""
        Given:
        - Source table: `{src_schema}.{src_table}`
        - Target table: `{tgt_schema}.{tgt_table}`

        Source columns:
        {json.dumps(src_cols, indent=2)}

        Target columns:
        {json.dumps(tgt_cols, indent=2)}

        Generate an STTM JSON with entries like this:
        {{
            "target_column": "<column_name_in_target>",
            "source_columns": ["col1", "col2", ...],
            "transformation": "<expression or null>"
        }}

        Rules:
        - Map only columns present in the target list.
        - Use similar names for direct mappings (case-insensitive) except fou audit columns.
        - For audit fields like 'audit_insrt_dt', 'audit_updt_dt', 'load_ts', use CURRENT_TIMESTAMP.
        - Output must be valid JSON (list of mappings).
        -INT schema tables are always SCD1 for dimensions
        -DWH Schema tables are always SCD2 for dimensions.
        -Process_ind in the INT Schema will be I for new records & U for updated records & D for deleted records.
        -When the source schema is INT and target schema is DWH pick up records which have processind in I and U.
        -DWH tables will have is_active column which is a flag .Its set to Y for currently active records and N for inactive records.
        -EFF_START_DT & EFF_END_DT are the dates in between which the records were active/currently active.

        """)
    ]

    response = model.invoke(messages)
    reply = response.content.strip()

    # Safe JSON extraction
    try:
        sttm = json.loads(reply)
    except Exception:
        import re
        match = re.search(r"\[[\s\S]*\]", reply)
        sttm = json.loads(match.group(0)) if match else []

    # Double-check target column validity (safety filter)
    valid_tgt_cols = set(tgt_cols)
    filtered_sttm = [m for m in sttm if m.get("target_column") in valid_tgt_cols]

    state["sttm"] = {
        "source_schema": src_schema,
        "source_table": src_table,
        "target_schema": tgt_schema,
        "target_table": tgt_table,
        "mappings": filtered_sttm
    }

    return state
# -------------------------
# Code Generation Node
# -------------------------
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage
import json

def codegen_node_func(state, preview=False):
    """
    Generates Python ETL code dynamically using the latest STTM mapping.
    """
    import_statements = set(["import pandas as pd"])
    sttm = state.get("sttm", {})
    etl_engine = state.get("etl_engine", "Pandas")
    load_type = state.get("load_type", "Full")

    src_schema = sttm.get("source_schema")
    src_table = sttm.get("source_table")
    tgt_schema = sttm.get("target_schema")
    tgt_table = sttm.get("target_table")

    func_name = f"{src_schema}_{tgt_schema}_{tgt_table}".lower()

    # ---- 🧠 Use the LLM to generate transformation logic dynamically ----
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

    # Make sure we use *actual* source & target columns from DB (avoid hallucination)
    src_columns = state.get("source_columns", [])
    tgt_columns = state.get("target_columns", [])

    sttm_json = json.dumps(sttm, indent=2)

    prompt = f"""
    You are a precise ETL code generator.
    Generate a clean Python function using pandas that:
    - Reads from table {src_schema}.{src_table}
    - Applies transformations from this STTM: {sttm_json}
    - Writes to table {tgt_schema}.{tgt_table}
    - Only use these source columns: {src_columns}
    - Only use these target columns: {tgt_columns}
    - Load type is: {load_type}
    - Use datetime.now() for current_timestamp if needed.
    - No placeholders or mock data.
    - Func_name is {func_name}
    -Use the .env file to create database connection.

    Output only valid executable Python code.
    """

    response = llm.invoke([
        SystemMessage(content="You are a strict Python ETL code generator."),
        HumanMessage(content=prompt)
    ])

    generated_code = response.content.strip()

    # Attach generated code back to the state
    state["code"] = generated_code
    state["sttm_used_for_codegen"] = sttm  # keep reference for traceability

    return state

# -------------------------
# Preview + Write Nodes
# -------------------------
def execute_sttm_preview(sttm, engine, etl_engine="Pandas"):
    """
    Reads source table, applies STTM transformations in-memory, and returns a Pandas DataFrame.
    Does NOT write to target table.
    """
    import pandas as pd

    src_schema = sttm["source_schema"]
    src_table = sttm["source_table"]

    # Read source table
    df = pd.read_sql(f"SELECT * FROM {src_schema}.{src_table}", engine)

    # Apply STTM mappings
    for mapping in sttm.get("mappings", []):
        tgt_col = mapping["target_column"]
        if not mapping["source_columns"]:
            # Default values for audit/process columns
            default_map = {
                "process_ind": "'I'",
                "AUDIT_INSRT_DT": pd.Timestamp.now(),
                "AUDIT_INSRT_NM": "agentic_user",
                "AUDIT_UPDT_DT": pd.Timestamp.now(),
                "AUDIT_UPDT_NM": "agentic_user",
                "AUDIT_BATCH_ID": "BATCH_001"
            }
            df[tgt_col] = default_map.get(tgt_col, None)
        elif len(mapping["source_columns"]) == 1:
            src_col = mapping["source_columns"][0].split('.')[-1]
            df[tgt_col] = df[src_col]
        else:
            cols = [c.split('.')[-1] for c in mapping["source_columns"]]
            df[tgt_col] = df[cols].astype(str).agg('_'.join, axis=1)

    return df

def write_to_target_table(df, target_schema, target_table, engine, etl_engine="Pandas"):
    """
    Writes the DataFrame to the target table.
    Supports Pandas and PySpark.
    """
    if etl_engine == "Pandas":
        df.to_sql(target_table, engine, schema=target_schema, if_exists='append', index=False)

    elif etl_engine == "PySpark":
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName('ETLJob').getOrCreate()
        sdf = spark.createDataFrame(df)
        sdf.write.format('jdbc').options(
            url=str(engine.url),
            dbtable=f'{target_schema}.{target_table}',
            user=engine.url.username,
            password=engine.url.password
        ).mode('append').save()
