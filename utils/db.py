import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import psycopg2

load_dotenv()

# --------------------------
# Connection / Engine
# --------------------------
def get_connection():
    """Return a psycopg2 connection (for simple reads)."""
    return psycopg2.connect(
        host=os.getenv("PG_HOST"),
        port=os.getenv("PG_PORT", "5432"),
        database=os.getenv("PG_DB"),
        user=os.getenv("PG_USER"),
        password=os.getenv("PG_PASS"),
        sslmode="require"
    )

def get_engine():
    """Return a SQLAlchemy engine (for ETL, inserts, etc.)."""
    user = os.getenv("PG_USER")
    password = os.getenv("PG_PASS")
    host = os.getenv("PG_HOST")
    port = os.getenv("PG_PORT", "5432")
    db = os.getenv("PG_DB")
    url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}"
    return create_engine(url, future=True)

# --------------------------
# Schema & Table Listings
# --------------------------
def list_schemas(engine=None):
    """Return list of non-system schemas."""
    if engine:
        query = "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog','information_schema') ORDER BY schema_name"
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
    else:
        conn = get_connection()
        query = "SELECT schema_name FROM information_schema.schemata WHERE schema_name NOT IN ('pg_catalog','information_schema') ORDER BY schema_name"
        df = pd.read_sql(query, conn)
        conn.close()
    return df["schema_name"].tolist()

def list_tables(schema_name, engine=None):
    """Return list of tables for a schema."""
    query = f"SELECT table_name FROM information_schema.tables WHERE table_schema='{schema_name}' ORDER BY table_name"
    if engine:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
    else:
        conn = get_connection()
        df = pd.read_sql(query, conn)
        conn.close()
    return df["table_name"].tolist()

# --------------------------
# Table Preview / Metadata
# --------------------------
def preview_table(schema_name, table_name, engine=None):
    """Return columns + sample rows."""
    col_query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='{schema_name}' AND table_name='{table_name}' ORDER BY ordinal_position"
    row_query = f"SELECT * FROM {schema_name}.{table_name} LIMIT 5"

    if engine:
        with engine.connect() as conn:
            col_df = pd.read_sql(col_query, conn)
            try:
                row_df = pd.read_sql(row_query, conn)
                sample_rows = row_df.to_dict(orient="records")
            except Exception:
                sample_rows = []
    else:
        conn = get_connection()
        col_df = pd.read_sql(col_query, conn)
        try:
            row_df = pd.read_sql(row_query, conn)
            sample_rows = row_df.to_dict(orient="records")
        except Exception:
            sample_rows = []
        conn.close()

    return {"columns": col_df.to_dict(orient="records"), "sample_rows": sample_rows}

def get_table_columns(schema_name, table_name, engine=None):
    """Return list of columns with types."""
    query = f"SELECT column_name, data_type FROM information_schema.columns WHERE table_schema='{schema_name}' AND table_name='{table_name}' ORDER BY ordinal_position"
    if engine:
        with engine.connect() as conn:
            df = pd.read_sql(query, conn)
    else:
        conn = get_connection()
        df = pd.read_sql(query, conn)
        conn.close()
    return df.to_dict(orient="records")

# --------------------------
# Row Count
# --------------------------
def get_table_rowcount(schema_name, table_name, engine):
    """Return number of rows in table (requires engine)."""
    query = text(f"SELECT COUNT(*) AS cnt FROM {schema_name}.{table_name}")
    with engine.connect() as conn:
        result = conn.execute(query)
        return result.scalar()

# --------------------------
# ETL Code Persistence
# --------------------------
from sqlalchemy import text

def get_existing_code(src_schema, src_table, tgt_schema, tgt_table, engine):
    query = text("""
        SELECT generated_code, NULL as sttm_json, etl_engine, NULL as load_type
        FROM audit.etl_code_store
        WHERE source_schema = :src_schema
          AND source_table = :src_table
          AND target_schema = :tgt_schema
          AND target_table = :tgt_table
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {
            "src_schema": src_schema,
            "src_table": src_table,
            "tgt_schema": tgt_schema,
            "tgt_table": tgt_table
        })
        
        row = result.mappings().first()   # <-- ✅ ensures dict-like access
        if not row:
            return None

        return {
            "generated_code": row["generated_code"],
            "sttm_json": row["sttm_json"],
            "etl_engine": row["etl_engine"],
            "load_type": row["load_type"]
        }


def save_generated_code(src_schema, src_table, tgt_schema, tgt_table,
                        table_type, scd_type, etl_engine, code, engine):
    """Insert/update generated ETL code in DB (requires engine)."""
    query = text("""
        INSERT INTO audit.etl_code_store
        (source_schema, source_table, target_schema, target_table,
         table_type, scd_type, etl_engine, generated_code)
        VALUES (:src_schema,:src_table,:tgt_schema,:tgt_table,
                :table_type,:scd_type,:etl_engine,:code)
        ON CONFLICT (source_schema, source_table, target_schema, target_table)
        DO UPDATE SET generated_code = EXCLUDED.generated_code,
                      etl_engine = EXCLUDED.etl_engine,
                      table_type = EXCLUDED.table_type,
                      scd_type = EXCLUDED.scd_type,
                      last_modified = CURRENT_TIMESTAMP
    """)
    with engine.connect() as conn:
        conn.execute(query, {
            "src_schema": src_schema,
            "src_table": src_table,
            "tgt_schema": tgt_schema,
            "tgt_table": tgt_table,
            "table_type": table_type,
            "scd_type": scd_type,
            "etl_engine": etl_engine,
            "code": code
        })
        conn.commit()
