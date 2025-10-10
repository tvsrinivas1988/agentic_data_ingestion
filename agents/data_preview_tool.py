from utils.db import preview_table

def preview_source_and_target(state):
    """
    Fetch first 5 rows and structure for source and target tables.
    Accepts schema-qualified table names (e.g., "myschema.mytable").
    Previews are stored in runtime state only, NOT in schema.
    """
    source_full = state.get("source_table")  # e.g., "schema.table"
    target_full = state.get("target_table")

    # Split schema and table
    source_schema, source_table = source_full.split(".", 1)
    target_schema, target_table = target_full.split(".", 1)

    state["source_preview"] = preview_table(source_schema, source_table)
    state["target_preview"] = preview_table(target_schema, target_table)

    return state
