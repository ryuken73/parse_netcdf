#!/usr/bin/env python3
"""기상청 RDR binary 파일과 예상 PNG 산출물을 점검한다.

이 스크립트는 netcdf 프로젝트나 native geospatial dependency를 import하지 않도록
Python standard library만 사용한다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


HEADER_OFFSET_BYTES = 1024
RDR_NX = 2305
RDR_NY = 2881
DTYPE_BYTES = 2
EXPECTED_SIZE_BYTES = HEADER_OFFSET_BYTES + RDR_NX * RDR_NY * DTYPE_BYTES

EXPECTED_SUFFIXES = [
    "_step1.png",
    "_step1_equi.png",
    "_step1_equi_normal.png",
    "_step5.png",
    "_step10.png",
]


def expected_outputs(bin_path: Path, out_root: Path) -> list[Path]:
    save_dir = out_root / bin_path.parent.name
    return [save_dir / f"{bin_path.stem}{suffix}" for suffix in EXPECTED_SUFFIXES]


def build_report(bin_file: Path, out_root: Path | None) -> dict:
    report = {
        "bin_file": str(bin_file),
        "exists": bin_file.exists(),
        "expected_size_bytes": EXPECTED_SIZE_BYTES,
        "header_offset_bytes": HEADER_OFFSET_BYTES,
        "rdr_nx": RDR_NX,
        "rdr_ny": RDR_NY,
        "dtype": "int16",
    }

    if bin_file.exists():
        actual_size = bin_file.stat().st_size
        report["actual_size_bytes"] = actual_size
        report["size_ok"] = actual_size == EXPECTED_SIZE_BYTES
        report["data_bytes_after_header"] = max(0, actual_size - HEADER_OFFSET_BYTES)
        report["expected_data_bytes_after_header"] = RDR_NX * RDR_NY * DTYPE_BYTES
    else:
        report["actual_size_bytes"] = None
        report["size_ok"] = False

    if out_root is not None:
        outputs = []
        for output in expected_outputs(bin_file, out_root):
            outputs.append(
                {
                    "path": str(output),
                    "exists": output.exists(),
                    "size_bytes": output.stat().st_size if output.exists() else None,
                }
            )
        report["output_root"] = str(out_root)
        report["expected_outputs"] = outputs
        report["outputs_complete"] = all(item["exists"] and item["size_bytes"] > 0 for item in outputs)

    return report


def print_text(report: dict) -> None:
    print(f"bin_file: {report['bin_file']}")
    print(f"exists: {report['exists']}")
    print(f"expected_size_bytes: {report['expected_size_bytes']}")
    print(f"actual_size_bytes: {report['actual_size_bytes']}")
    print(f"size_ok: {report['size_ok']}")

    if "expected_outputs" in report:
        print("expected_outputs:")
        for item in report["expected_outputs"]:
            status = "ok" if item["exists"] and item["size_bytes"] > 0 else "missing"
            print(f"  {status}: {item['path']} ({item['size_bytes']})")
        print(f"outputs_complete: {report['outputs_complete']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="기상청 RDR binary 크기와 예상 PNG 산출물을 점검한다.")
    parser.add_argument("bin_file", help="기상청 RDR .bin 파일 경로")
    parser.add_argument("--out-root", help="예상 산출물을 확인할 OUT_PATH_RDR root")
    parser.add_argument("--json", action="store_true", help="기계가 읽기 쉬운 JSON으로 출력한다")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    bin_file = Path(args.bin_file)
    out_root = Path(args.out_root) if args.out_root else None
    report = build_report(bin_file, out_root)

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_text(report)

    if not report["exists"]:
        return 1
    if not report["size_ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
