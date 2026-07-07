import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repo_index.parse import find_python_files, extract_symbols
from repo_index.imports import extract_imports
from repo_index.calls import extract_calls

def build_index(repo_path):
    """
    Point at a repo, get back three plain data structures:
    symbols, import_edges, call_edges.
    No model or selector code touches this -- kept isolated per the task.
    """
    py_files = find_python_files(repo_path)

    symbols = []
    import_edges = []
    call_edges = []

    for file_path in py_files:
        symbols.extend(extract_symbols(file_path))
        import_edges.extend(extract_imports(file_path))
        call_edges.extend(extract_calls(file_path))
    
    return {
        "symbols": symbols,
        "import_edges": import_edges,
        "call_edges": call_edges
    }