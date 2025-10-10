import io, sys

def execute_code(state):
    code = state["generated_code"]
    buffer = io.StringIO()
    sys.stdout = buffer
    try:
        exec_globals = {}
        exec(code, exec_globals)
        result = "executed successfully"
        output = buffer.getvalue()
    except Exception as e:
        result = f"execution failed: {e}"
        output = buffer.getvalue()
    finally:
        sys.stdout = sys.__stdout__
    return {"execution_result": result, "execution_output": output}
