from utils.db import insert_record
import json
def persist_to_db(state):
    sttm = json.dumps(state["structured_request"])
    code = state["generated_code"]
    engine = state["engine"]
    validation_status = state["validation_status"]

    record_id = insert_record(sttm, code, engine, validation_status)
    return {"record_id": record_id}
