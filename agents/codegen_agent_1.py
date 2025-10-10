from langchain_openai import ChatOpenAI
import json
import os
from dotenv import load_dotenv

load_dotenv()
api_key  = os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model="gpt-5", temperature=0.2,api_key=api_key)

def generate_code(state):
    """
    Generate PySpark code from STTM JSON structure.
    """
    sttm = state["structured_request"]

    if isinstance(sttm, dict):
        sttm_json = json.dumps(sttm, indent=2)
    else:
        sttm_json = sttm

    prompt = f"""
    You are a data engineer. Generate executable PySpark code that:
    1. Loads all source tables with aliases from 'source_tables'.
    2. Applies joins defined in 'join_logic'.
    3. Maps each 'mappings.target_column' with transformations.
    4. Handles 'load_type': 'incremental' -> append, 'full' -> overwrite.
    5. Writes results to 'target_table'.
    6. Uses only valid Python/PySpark code; no explanations.

    STTM:
    {sttm_json}
    """

    code = llm.invoke(prompt).strip()
    return {"generated_code": code, "engine": "pyspark"}
