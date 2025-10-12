import json
from utils.db import get_table_columns, get_connection,list_schemas,list_tables
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

    # Include schema overview for context
    all_schemas = {s: list_tables(s) for s in list_schemas()}

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
            "Include join details if the column comes from another schema/table."
            "Use the key 'join_logic' and 'lookup_table' for such cases."

        )),
        HumanMessage(content=f"""
        Given:
        - Primary source table: `{src_schema}.{src_table}`
        - Target table: `{tgt_schema}.{tgt_table}`

        Available schemas and tables:
        {json.dumps(all_schemas, indent=2)}

        Source columns:
        {json.dumps(src_cols, indent=2)}

        Target columns:
        {json.dumps(tgt_cols, indent=2)}

        Generate an STTM JSON with entries like this for simple table:
        {{
            "target_column": "<column_name_in_target>",
            "source_columns": ["col1", "col2", ...],
            "transformation": "<expression or null>",
            "pkey" : "<Primary Keys for the target table>"
        }}

        Generate an STTM JSON with entries like this for complex table that need joins:
        
        {{
            "target_column": "product_name",
            "pkey" : "<Primary Keys for the target table>",
            "source_tables": [
        {{
            "schema": "ref",
            "table": "product_master",
            "join_type": "inner",
            "join_logic": "stg.sales_txn.product_id = ref.product_master.product_id"
        }}
        ],
            "source_columns": ["product_id"],
            "transformation": "<expression or null>"
            
            
        }}
        

        Rules:
        - Map only columns present in the target list.
        - Use similar names for direct mappings (case-insensitive) except for audit columns.
        - For audit fields like 'audit_insrt_dt', 'audit_updt_dt', 'load_ts', use CURRENT_TIMESTAMP.
        - If you use more than one table, include the "source_tables" field in the mapping like this:
        - If join is needed, specify both lookup_table and join_logic
        - Output must be valid JSON (list of mappings).
        -INT schema tables are always SCD1 for dimensions
        -DWH Schema tables are always SCD2 for dimensions.
        -Process_ind in the INT Schema will be I for new records & U for updated records & D for deleted records.
        -When the source schema is INT and target schema is DWH pick up records which have processind in I and U.
        -When looking up records from DWH always use the filter IS_ACTIVE='Y'
        -When looking up records from INT always use the filter PROCESS_IND in ('I','U')
        -DWH tables will have is_active column which is a flag .Its set to Y for currently active records and N for inactive records.
        -EFF_START_DT & EFF_END_DT are the dates in between which the records were active/currently active.
        -Provide the Primary_key details as well by profiling if needed.

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

    ##print(filtered_sttm)
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
    scd_type =state.get("scd_type",'NA')
    

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
    - If there are joins referring to other table in STTM , ensure to read those as well to apply those transformations.
    - Writes to table {tgt_schema}.{tgt_table}
    - Only use these source columns: {src_columns}
    - Only use these target columns: {tgt_columns}
    - Load type is: {load_type}
    - Process_ind should be derived based on the rule  and assumed.
    - Primary key is present in STTM as pkey
    - Use datetime.now() for current_timestamp if needed.
    - No placeholders or mock data.
    - Func_name is {func_name}
    -Use the scd_type value in {scd_type} to generate code for Dimensions.
    -Use the .env file to create database connection.
    -Apply the following logic :
        -If load type is incremental and scd type is SCD1 apply insert else update to the table based on primary key defined as pkey.
        -If load type is full and scd type is SCD1  apply truncate and load to the table .
        -If load type is incremental or full and scd type is SCD2 apply insert and  update to the table based on primary key.Also  end date the previous records and mark them inactive
    -Make sure to add single quote data values in sql where clauses 
    -All dynamically executed SQL statements must use sqlalchemy.text() with parameterized values.”

    ### Safe SQL Execution (Critical)
    - All SQL statements (INSERT, UPDATE, DELETE, TRUNCATE) **must** be executed using:
        from sqlalchemy import text
        conn.execute(text(sql), parameters_dict)
    - Do **not** execute plain strings directly with conn.execute().
    - Use parameterized queries with placeholders (e.g. :brand_id, :brand_name) instead of string concatenation.
    - If multiple rows are inserted, use pandas to_sql() where possible, or loop efficiently.
    -Create code as per the load type. Dont pass any paramters to the function call.
    -Dont assume scd type and load type.Its available as {scd_type} and {load_type}

    ###Code sql
    When generating ETL or data ingestion scripts that update records in a database:
    - Always use `with engine.begin() as conn:` for transactional operations.
    - Never use `with engine.connect()` without explicitly committing, since SQLAlchemy 2.x does not auto-commit.
    - Make sure that inserts done with pandas `.to_sql()` and manual updates both persist consistently.
    
    ### Performance
    - Never query inside a per-row loop.
    - If you need to check for existing keys (e.g., brand_id), fetch all keys from the target table once and compare in-memory.


    Output only valid executable Python code & call the code.
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
