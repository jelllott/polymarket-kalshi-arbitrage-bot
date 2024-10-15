"""Aggregate per-pruner JSONs into a single table.

TODO: pretty-print as markdown.
"""
import json, os, sys, glob

def main(outdir):
    rows = []
    for p in sorted(glob.glob(os.path.join(outdir, "*.json"))):
        with open(p) as f:
            rows.append({"file": os.path.basename(p), **json.load(f)})
    print(json.dumps(rows, indent=2))

if __name__ == "__main__":
    main(sys.argv[1])
