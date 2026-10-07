"""Inspect input image/mask geometry without modifying any source files."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/"src"))
from bridge_tuning.config import write_json
from bridge_tuning.data import check_geometry, read_manifest

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv",required=True);p.add_argument("--output",required=True)
    a=p.parse_args();rows=[]
    for item in read_manifest(a.csv):
        try:check_geometry([item]);rows.append({"id":item["id"],"compatible":True})
        except ValueError as error:rows.append({"id":item["id"],"compatible":False,"reason":str(error)})
    report={"n_cases":len(rows),"n_compatible":sum(r["compatible"]for r in rows),"source_modified":False,"cases":rows}
    write_json(a.output,report);print(f"Compatible grids: {report['n_compatible']}/{len(rows)}")
    raise SystemExit(0 if all(r["compatible"]for r in rows) else 2)
