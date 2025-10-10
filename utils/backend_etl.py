import os
import importlib.util
from utils.db import get_engine, save_generated_code

def run_generated_etl(source_schema, source_table, target_schema, target_table, generated_code,
                      table_type, scd_type, etl_engine, load_type="Full", persist_code=True):
    """
    Executes the generated ETL code on the backend and optionally persists it to DB.
    """
    engine = get_engine()

    # Save generated code to a temporary file
    temp_file = f"generated_etl_{source_schema}_{target_schema}_{target_table}.py".lower()
    with open(temp_file, "w") as f:
        f.write(generated_code)

    # Dynamically import the generated module
    spec = importlib.util.spec_from_file_location("generated_etl", temp_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Execute the ETL function
    func_name = f"{source_schema}_{target_schema}_{target_table}".lower()
    getattr(module, func_name)(engine)

    # Persist generated code to DB if requested
    if persist_code:
        save_generated_code(
            source_schema, source_table,
            target_schema, target_table,
            table_type, scd_type, etl_engine,
            generated_code,
            engine
        )

    return f"ETL completed for {target_schema}.{target_table}"
