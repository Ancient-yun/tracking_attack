"""Inspect an anonymous public Google Drive folder snapshot, without credentials."""
import json
import pathlib
import re
import sys


def entries(html: str):
    match = re.search(r"window\['_DRIVE_ivd'\] = '([^']+)';", html)
    if not match:
        raise RuntimeError("Public folder metadata absent (permission/login or changed HTML)")
    escaped = match.group(1)
    decoded = re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m[1], 16)), escaped)
    decoded = decoded.replace(r"\/", "/").replace(r"\'", "'")
    payload = json.loads(decoded)
    return [{"id": x[0], "name": x[2], "mime": x[3], "bytes": x[13]} for x in payload[0]]


if __name__ == "__main__":
    for file in sys.argv[1:]:
        p = pathlib.Path(file)
        print(p.name)
        print(json.dumps(entries(p.read_text(encoding="utf-8")), indent=2, ensure_ascii=False))
