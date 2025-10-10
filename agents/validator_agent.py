import ast

def validate_code(state):
    code = state["generated_code"]
    try:
        ast.parse(code)
        return {"validation_status": "passed"}
    except SyntaxError as e:
        return {"validation_status": f"failed: {e}"}
