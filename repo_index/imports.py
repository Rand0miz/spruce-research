import ast

def extract_imports(file_path):
    """
    Parse one file's AST. Return a list of edges:
    {from_file, imported_module}
    Covers both `import x` and `from x import y`.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    tree = ast.parse(source, filename=file_path)
    imports = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "from_file": file_path,
                    "imported_module": alias.name
                })
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module if node.module else ""
            imports.append({
                "from_file": file_path,
                "imported_module": module_name
            })

    return imports