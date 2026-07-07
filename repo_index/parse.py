import ast, os

EXCLUDED_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "env",
    "node_modules",
    "venv",
}

def find_python_files(repo_path):
    """Return a list of Python source files in the repository.

    Common virtual environments, caches, and build outputs are skipped so the
    indexer stays focused on project code instead of crawling thousands of
    dependency files.
    """
    python_files = []
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
        for filename in files:
            if filename.endswith(".py"):
                python_files.append(os.path.join(root, filename))
    return python_files

def extract_symbols(file_path):
    """
    Parse one file's AST. Return a list of symbols:
    {name, kind, lineno, file}
    kind is "function" or "class".
    """
    with open(file_path, "r", encoding="utf-8") as f:
        source = f.read()
    
    tree = ast.parse(source, filename=file_path)
    symbols = []

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            symbols.append({
                "name": node.name,
                "kind": "function",
                "lineno": node.lineno,
                "file": file_path
            })
        elif isinstance(node, ast.ClassDef):
            symbols.append({
                "name": node.name,
                "kind": "class",
                "lineno": node.lineno,
                "file": file_path
            })

    return symbols