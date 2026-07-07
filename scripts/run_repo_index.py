import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from repo_index.build_index import build_index

repo_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
index = build_index(repo_path)

print(f"symbols found: {len(index['symbols'])}")
print(f"import edges: {len(index['import_edges'])}")
print(f"call edges: {len(index['call_edges'])}")

print("\nfirst 5 symbols:")
for s in index["symbols"][:5]:
    print(" ", s)

print("\nfirst 5 import edges:")
for e in index["import_edges"][:5]:
    print(" ", e)

print("\nfirst 5 call edges:")
for e in index["call_edges"][:5]:
    print(" ", e)