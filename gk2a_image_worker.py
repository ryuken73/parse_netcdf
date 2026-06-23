import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from config import get_config


SAVE_IMAGE_STEPS = {
    "ea": [1, 5, 10],
    "fd": [1, 5, 10],
    "ko": [1, 5, 10],
}

IMAGE_SIZE = {
    "ea": {
        1: (1500, 1300),
        5: (1200, 1040),
        10: (900, 780),
    },
    "fd": {
        1: (3192, 3192),
        5: (2048, 2048),
        10: (1400, 1400),
    },
    "ko": {
        1: (800, 800),
        5: (700, 700),
        10: (600, 600),
    },
}

BOUNDS = {
    "ea": [76.81183423919347, 11.369317564542508, 175.08747983767321, 61.93104770869447],
    "fd": [30, -80, 220, 80],
    "ko": [113.99641, 29.312252, 138.003582, 45.728965],
}


APP_DIR = Path(__file__).resolve().parent


def get_nc_coverage(nc_file):
    basename = Path(nc_file).stem
    parts = basename.split("_")
    if len(parts) < 5:
        raise ValueError(f"Unexpected GK2A file name: {nc_file}")
    coverage = parts[4][:2]
    if coverage not in SAVE_IMAGE_STEPS:
        raise ValueError(f"Unsupported GK2A coverage '{coverage}': {nc_file}")
    return coverage


def get_json_fname(out_dir, nc_file, step):
    basename = Path(nc_file).stem
    return str(Path(out_dir) / f"{basename}_step{step}.json")


def get_save_dir(nc_file):
    config = get_config()
    sub_dir = Path(nc_file).parent.name
    return Path(config.OUT_PATH) / sub_dir


def expected_outputs(nc_file):
    nc_coverage = get_nc_coverage(nc_file)
    save_dir = get_save_dir(nc_file)
    highest_step = min(SAVE_IMAGE_STEPS[nc_coverage])
    outputs = []

    high_stem = Path(get_json_fname(save_dir, nc_file, highest_step)).stem
    outputs.extend(
        [
            save_dir / f"{high_stem}_mono.png",
            save_dir / f"{high_stem}_color.png",
            save_dir / f"{high_stem}_mono_equi.png",
            save_dir / f"{high_stem}_color_equi.png",
        ]
    )

    for step in SAVE_IMAGE_STEPS[nc_coverage]:
        if step == highest_step:
            continue
        stem = Path(get_json_fname(save_dir, nc_file, step)).stem
        outputs.extend(
            [
                save_dir / f"{stem}_mono.png",
                save_dir / f"{stem}_color.png",
            ]
        )

    return outputs


def is_processed(nc_file):
    return all(path.exists() and path.stat().st_size > 0 for path in expected_outputs(nc_file))


def process_file(nc_file):
    # Import heavy NetCDF/GDAL dependencies only in the per-file worker process.
    from parseWithVectorNC import (
        generate_image_from_data_fast,
        read_ir105_ea_fast_with_vector,
        read_ir105_fd_fast_with_vector,
        resize_image,
    )
    from to_epsg3857_keep_size import convert_to_equi_rectangle

    nc_file = str(Path(nc_file).resolve())
    conversion_file = str(APP_DIR / "ir105_conversion_c.txt")

    print(f"processing file {nc_file}", flush=True)
    save_dir = get_save_dir(nc_file)
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"save file to {save_dir}", flush=True)

    nc_coverage = get_nc_coverage(nc_file)
    highest_step = min(SAVE_IMAGE_STEPS[nc_coverage])
    out_file = get_json_fname(save_dir, nc_file, highest_step)
    attr_to_get = "image_pixel_values"

    print("start read ir105 fast", flush=True)
    if nc_coverage in ("ea", "ko"):
        parse_result = read_ir105_ea_fast_with_vector(nc_file, highest_step, attr_to_get, conversion_file)
    else:
        parse_result = read_ir105_fd_fast_with_vector(nc_file, highest_step, attr_to_get, conversion_file)

    if parse_result is None or len(parse_result) == 0:
        raise RuntimeError(f"No parsed data returned for {nc_file}")

    high_quality_image_name_mono = str(save_dir / f"{Path(out_file).stem}_mono.png")
    high_quality_image_name_color = str(save_dir / f"{Path(out_file).stem}_color.png")
    image_size = IMAGE_SIZE[nc_coverage][highest_step]
    bounds = BOUNDS[nc_coverage]

    generate_image_from_data_fast(
        parse_result,
        high_quality_image_name_mono,
        image_size,
        bounds,
        color_mode="gray",
    )
    print("saved high quality image[mono]:", high_quality_image_name_mono, flush=True)

    generate_image_from_data_fast(
        parse_result,
        high_quality_image_name_color,
        image_size,
        bounds,
        color_mode="color",
    )
    print("saved high quality image[color]:", high_quality_image_name_color, flush=True)

    high_quality_image_name_mono_equi = str(save_dir / f"{Path(out_file).stem}_mono_equi.png")
    high_quality_image_name_color_equi = str(save_dir / f"{Path(out_file).stem}_color_equi.png")
    convert_to_equi_rectangle(nc_coverage, high_quality_image_name_mono, high_quality_image_name_mono_equi)
    convert_to_equi_rectangle(nc_coverage, high_quality_image_name_color, high_quality_image_name_color_equi)
    print(
        "saved high quality image[equi]:",
        high_quality_image_name_mono_equi,
        high_quality_image_name_color_equi,
        flush=True,
    )

    print("start downgrade image quality", flush=True)
    for step in SAVE_IMAGE_STEPS[nc_coverage]:
        if step == highest_step:
            continue

        step_out_file = get_json_fname(save_dir, nc_file, step)
        out_image_name_mono = str(save_dir / f"{Path(step_out_file).stem}_mono.png")
        out_image_name_color = str(save_dir / f"{Path(step_out_file).stem}_color.png")
        resize_image(high_quality_image_name_mono, out_image_name_mono, IMAGE_SIZE[nc_coverage][step])
        resize_image(high_quality_image_name_color, out_image_name_color, IMAGE_SIZE[nc_coverage][step])
        print("save mono image:", out_image_name_mono, flush=True)
        print("save color image:", out_image_name_color, flush=True)

    print(f"completed file {nc_file}", flush=True)


def discover_batch_files(batch_root, pending_only=True):
    batch_root = Path(batch_root)
    files = sorted(batch_root.rglob("*.nc")) if batch_root.is_dir() else [batch_root]
    selected = []

    for nc_file in files:
        try:
            if pending_only and is_processed(nc_file):
                continue
            selected.append(str(nc_file))
        except Exception as exc:
            print(f"skip {nc_file}: {exc}", flush=True)

    return selected


def run_worker_subprocess(nc_file, retry_count, retry_delay):
    cmd = [sys.executable, str(APP_DIR / "gk2a_image_worker.py"), "--file", str(nc_file)]
    env = os.environ.copy()
    cwd = str(APP_DIR)

    for attempt in range(1, retry_count + 2):
        print(f"worker attempt {attempt}/{retry_count + 1}: {nc_file}", flush=True)
        result = subprocess.run(cmd, cwd=cwd, env=env)
        if result.returncode == 0:
            return 0
        if attempt <= retry_count:
            time.sleep(retry_delay)

    return result.returncode


def run_batch(batch_root, max_workers, retry_count, retry_delay, pending_only):
    max_workers = max(1, max_workers)
    files = discover_batch_files(batch_root, pending_only=pending_only)
    print(f"batch candidates: {len(files)}", flush=True)
    if not files:
        return 0

    failed = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_worker_subprocess, nc_file, retry_count, retry_delay): nc_file
            for nc_file in files
        }
        for future in as_completed(futures):
            nc_file = futures[future]
            try:
                returncode = future.result()
            except Exception as exc:
                print(f"batch failed: {nc_file} - {exc}", flush=True)
                failed.append(nc_file)
                continue

            if returncode != 0:
                print(f"batch failed: {nc_file} - returncode {returncode}", flush=True)
                failed.append(nc_file)
            else:
                print(f"batch completed: {nc_file}", flush=True)

    if failed:
        print("batch failed files:", flush=True)
        for nc_file in failed:
            print(nc_file, flush=True)
        return 1

    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="GK2A image worker")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--file", help="Process one GK2A NetCDF file")
    mode.add_argument("--batch", help="Process a file or directory of GK2A NetCDF files")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.getenv("GK2A_IMAGE_MAX_WORKERS", "2")),
        help="Maximum parallel subprocess workers for batch mode",
    )
    parser.add_argument(
        "--retry-count",
        type=int,
        default=int(os.getenv("GK2A_IMAGE_RETRY_COUNT", "1")),
        help="Retries per file",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=float(os.getenv("GK2A_IMAGE_RETRY_DELAY", "30")),
        help="Seconds to wait between retries",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="In batch mode, process all files instead of only files with missing outputs",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.file:
        process_file(args.file)
        return 0

    return run_batch(
        args.batch,
        max_workers=args.max_workers,
        retry_count=args.retry_count,
        retry_delay=args.retry_delay,
        pending_only=not args.force,
    )


if __name__ == "__main__":
    raise SystemExit(main())
