"""Check distributed file hashes without importing model dependencies."""
import argparse
import hashlib
from pathlib import Path

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-only",action="store_true",help="Check the source-only archive inventory")
    a=p.parse_args();root=Path(__file__).resolve().parent
    manifest=root/("SOURCE_SHA256SUMS.txt" if a.source_only else "SHA256SUMS.txt")
    checked=0
    for line in manifest.read_text().splitlines():
        expected,name=line.split("  ",1);path=(root/name).resolve()
        if not path.is_relative_to(root)or not path.is_file():raise ValueError(f"Missing or invalid file: {name}")
        digest=hashlib.sha256()
        with path.open("rb")as f:
            for block in iter(lambda:f.read(8*1024*1024),b""):digest.update(block)
        if digest.hexdigest()!=expected:raise ValueError(f"SHA256 mismatch: {name}")
        checked+=1
    print(f"SHA256 verified for {checked} files")
