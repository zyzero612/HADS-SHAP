from __future__ import annotations

import hashlib
from pathlib import Path


EXPECTED = {
    "data/raw/beijing_multi_site_air_quality.zip": "d1b9261c54132f04c374f762f1e5e512af19f95c95fd6bfa1e8ac7e927e3b0b8",
    "data/raw/gas_sensor_array_drift.zip": "91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    failed = False
    for name, expected in EXPECTED.items():
        path = Path(name)
        if not path.exists():
            print(f"MISSING  {name}")
            failed = True
            continue
        observed = sha256(path)
        status = "OK" if observed == expected else "MISMATCH"
        print(f"{status:8s} {name}  {observed}")
        failed |= observed != expected
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
