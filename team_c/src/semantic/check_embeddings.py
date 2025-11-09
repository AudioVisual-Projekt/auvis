import argparse, os, json, numpy as np
from numpy.linalg import norm

def cos(a,b): 
    d = (norm(a)*norm(b)) + 1e-12
    return float(a @ b / d)

p = argparse.ArgumentParser()
p.add_argument("--dir", type=str, default="../../data-bin")
args = p.parse_args()

E = np.load(os.path.join(args.dir, "E.npy"))
with open(os.path.join(args.dir, "ids.json"), "r", encoding="utf-8") as f:
    ids = json.load(f)

print("E-Shape:", E.shape, "| IDs:", len(ids))
assert E.shape[0] == len(ids), "Anzahl Embeddings ≠ Anzahl IDs"

if E.shape[0] > 1:
    print(f"cos(ids[0], ids[1]) = {cos(E[0], E[1]):.3f}")
print("✓ Basiskontrolle erfolgreich.")
