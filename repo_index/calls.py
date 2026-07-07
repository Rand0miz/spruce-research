import ast

def extract_calls(file_path):
    """
    Parse one file's AST. Return a list of edges:
    {caller_function, called_name, file}
    caller_function is None for calls made outside any function (module level).
    called_name is the raw name as written (e.g. "foo" or "obj.method") --
    resolving it to a specific symbol elsewhere is a separate concern.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    tree = ast.parse(source, filename=file_path)
    calls = []

    #track the current function name as we traverse the AST
    class CallVisitor(ast.NodeVisitor):
        def __init__(self):
            self.current_function = None
        
        def visit_FunctionDef(self, node):
            prev = self.current_function
            self.current_function = node.name
            self.generic_visit(node)
            self.current_function = prev
        
        visit_AsyncFunctionDef = visit_FunctionDef  # handle async functions the same way

        def visit_Call(self, node):
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                called_name = node.func.attr
            else:
                called_name = None  # could be a complex expression, ignore for now

            if called_name:
                calls.append({
                    "caller_function": self.current_function,
                    "called_name": called_name,
                    "file": file_path
                })
            self.generic_visit(node)

    CallVisitor().visit(tree)
    return calls