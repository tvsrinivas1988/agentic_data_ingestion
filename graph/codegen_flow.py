import json
from utils.db import get_table_columns, get_connection

# -------------------------
# STTM Generator Node
# -------------------------
def generate_sttm_node(state):
    """
    Generates draft STTM based on source/target table metadata,
    and adds audit/process columns automatically.
    """
    src_schema = state["source_schema"]
    src_table = state["source_table"]
    tgt_schema = state["target_schema"]
    tgt_table = state["target_table"]

    src_cols = [col["column_name"] for col in get_table_columns(src_schema, src_table)]
    tgt_cols = [col["column_name"] for col in get_table_columns(tgt_schema, tgt_table)]

    # Default STTM structure
    sttm = {
        "source_schema": src_schema,
        "source_table": src_table,
        "target_schema": tgt_schema,
        "target_table": tgt_table,
        "load_type": "full",
        "primary_key": tgt_cols[0] if tgt_cols else None,
        "mappings": [],
        "join_logic": []
    }

    # Add mappings for matching columns
    for col in tgt_cols:
        if col in src_cols:
            sttm["mappings"].append({
                "target_column": col,
                "source_columns": [col],
                "transformation": col,
                "notes": "Direct mapping"
            })

    # Inject audit/process columns if not present in mappings
    for audit_col in ["process_ind","AUDIT_INSRT_DT","AUDIT_INSRT_NM",
                      "AUDIT_UPDT_DT","AUDIT_UPDT_NM","AUDIT_BATCH_ID"]:
        if audit_col not in [m["target_column"] for m in sttm["mappings"]]:
            default_transform = {
                "process_ind": "'I'",
                "AUDIT_INSRT_DT": "current_timestamp()",
                "AUDIT_INSRT_NM": "'agentic_user'",
                "AUDIT_UPDT_DT": "current_timestamp()",
                "AUDIT_UPDT_NM": "'agentic_user'",
                "AUDIT_BATCH_ID": "'BATCH_001'"
            }
            sttm["mappings"].append({
                "target_column": audit_col,
                "source_columns": [],
                "transformation": default_transform[audit_col],
                "notes": "Audit/process column"
            })

    state["sttm"] = sttm
    return state

# -------------------------
# Code Generation Node
# -------------------------
def codegen_node_func(state, preview=False):
    """
    Generate Python ETL code from STTM.
    The generated code is ready to be executed on backend.
    """
    import_statements = set()
    
    sttm = state.get("sttm", {})
    etl_engine = state.get("etl_engine", "Pandas")
    load_type = state.get("load_type", "Full")  # Full or Incremental
    
    src_schema = sttm.get("source_schema")
    src_table = sttm.get("source_table")
    tgt_schema = sttm.get("target_schema")
    tgt_table = sttm.get("target_table")
    func_name = f"{src_schema}_{tgt_schema}_{tgt_table}".lower()
    
    code_lines = []

    if etl_engine == "Pandas":
        import_statements.add("import pandas as pd")
        import_statements.add("from datetime import datetime")

        code_lines.append(f"def {func_name}(engine):")
        code_lines.append(f"    # Read source table")
        code_lines.append(f"    df = pd.read_sql('SELECT * FROM {src_schema}.{src_table}', engine)")
        
        # Apply mappings
        code_lines.append(f"    # Apply mappings as per STTM")
        for mapping in sttm.get("mappings", []):
            tgt_col = mapping["target_column"]  # keep exact case
            trans = mapping.get("transformation")
            src_cols = mapping.get("source_columns", [])
            
            if not src_cols:
                if trans == "current_timestamp":
                    code_lines.append(f"    df['{tgt_col}'] = datetime.now()")
                else:
                    code_lines.append(f"    df['{tgt_col}'] = {trans}")
            elif len(src_cols) == 1:
                src_col = src_cols[0].split('.')[-1]
                code_lines.append(f"    df['{tgt_col}'] = df['{src_col}']")
            else:
                cols = [c.split('.')[-1] for c in src_cols]
                cols_str = ", ".join([f"'{c}'" for c in cols])
                code_lines.append(f"    df['{tgt_col}'] = df[[{cols_str}]].astype(str).agg('_'.join, axis=1)")
        
        # Write to target
        code_lines.append(f"    # Write to target table")
        if not preview:
            if load_type.lower() == "full":
                if_exists_mode = "replace"
            else:
                if_exists_mode = "append"
            code_lines.append(f"    df.to_sql('{tgt_table}', engine, schema='{tgt_schema}', if_exists='{if_exists_mode}', index=False)")
            code_lines.append(f"    print('ETL job completed for {tgt_schema}.{tgt_table} ({load_type} Load)')")
        
        code_lines.append("    return df  # Always return df for inspection if needed")

    # Combine imports + code
    code = "\n".join(sorted(import_statements)) + "\n\n" + "\n".join(code_lines)
    state["code"] = code
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
