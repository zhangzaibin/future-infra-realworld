#!/usr/bin/env python3
"""Split LeRobot v3 chunked videos into one file per episode and camera."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq


@dataclass(frozen=True)
class SplitJob:
    episode_index: int
    camera: str
    source: Path
    output: Path
    start_s: float
    end_s: float
    expected_frames: int
    tasks: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split LeRobot v3 chunked videos into episode-level MP4 files."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace episode videos that already exist in the output directory.",
    )
    return parser.parse_args()


def load_jobs(dataset_root: Path, output_dir: Path) -> list[SplitJob]:
    info_path = dataset_root / "meta" / "info.json"
    with info_path.open(encoding="utf-8") as file:
        info = json.load(file)

    camera_keys = [
        key
        for key, feature in info["features"].items()
        if feature.get("dtype") == "video"
    ]
    if not camera_keys:
        raise RuntimeError(f"No video features found in {info_path}")

    episode_files = sorted((dataset_root / "meta" / "episodes").glob("chunk-*/*.parquet"))
    if not episode_files:
        raise FileNotFoundError("No episode metadata parquet files found")

    columns = ["episode_index", "tasks", "length"]
    for key in camera_keys:
        columns.extend(
            [
                f"videos/{key}/chunk_index",
                f"videos/{key}/file_index",
                f"videos/{key}/from_timestamp",
                f"videos/{key}/to_timestamp",
            ]
        )

    episodes = pq.read_table(episode_files, columns=columns).to_pylist()
    jobs: list[SplitJob] = []
    video_template = info["video_path"]

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        expected_frames = int(episode["length"])
        tasks = tuple(episode.get("tasks") or ())
        episode_dir = output_dir / f"episode_{episode_index:06d}"

        for key in camera_keys:
            prefix = f"videos/{key}"
            chunk_index = int(episode[f"{prefix}/chunk_index"])
            file_index = int(episode[f"{prefix}/file_index"])
            start_s = float(episode[f"{prefix}/from_timestamp"])
            end_s = float(episode[f"{prefix}/to_timestamp"])
            camera = key.rsplit(".", 1)[-1]
            source = dataset_root / video_template.format(
                video_key=key,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            jobs.append(
                SplitJob(
                    episode_index=episode_index,
                    camera=camera,
                    source=source,
                    output=episode_dir / f"{camera}.mp4",
                    start_s=start_s,
                    end_s=end_s,
                    expected_frames=expected_frames,
                    tasks=tasks,
                )
            )

    missing = sorted({job.source for job in jobs if not job.source.is_file()})
    if missing:
        preview = "\n".join(str(path) for path in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} source video files:\n{preview}")
    return jobs


def split_one(job: SplitJob, overwrite: bool) -> tuple[SplitJob, str]:
    if job.output.exists() and not overwrite:
        return job, "skipped"

    job.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = job.output.with_name(f".{job.output.stem}.part.mp4")
    temporary.unlink(missing_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{job.start_s:.9f}",
        "-i",
        str(job.source),
        "-frames:v",
        str(job.expected_frames),
        "-map",
        "0:v:0",
        "-an",
        "-c",
        "copy",
        "-avoid_negative_ts",
        "make_zero",
        str(temporary),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        if not temporary.is_file() or temporary.stat().st_size == 0:
            raise RuntimeError(f"ffmpeg produced an empty file: {temporary}")
        os.replace(temporary, job.output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return job, "written"


def write_manifest(output_dir: Path, jobs: list[SplitJob]) -> None:
    manifest = output_dir / "manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "episode_index",
                "camera",
                "video_path",
                "expected_frames",
                "duration_s",
                "task",
            ]
        )
        for job in sorted(jobs, key=lambda item: (item.episode_index, item.camera)):
            writer.writerow(
                [
                    job.episode_index,
                    job.camera,
                    job.output.relative_to(output_dir),
                    job.expected_frames,
                    f"{job.end_s - job.start_s:.9f}",
                    " | ".join(job.tasks),
                ]
            )


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if output_dir == dataset_root or dataset_root in output_dir.parents:
        raise ValueError("Output directory must be outside the source dataset")

    jobs = load_jobs(dataset_root, output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Splitting {len(jobs)} videos for "
        f"{len({job.episode_index for job in jobs})} episodes into {output_dir}"
    )

    written = 0
    skipped = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(split_one, job, args.overwrite) for job in jobs]
        for completed, future in enumerate(as_completed(futures), start=1):
            job, status = future.result()
            written += status == "written"
            skipped += status == "skipped"
            if completed % 25 == 0 or completed == len(jobs):
                print(
                    f"[{completed}/{len(jobs)}] written={written} skipped={skipped} "
                    f"latest=episode_{job.episode_index:06d}/{job.camera}.mp4",
                    flush=True,
                )

    write_manifest(output_dir, jobs)
    print(f"Done. written={written}, skipped={skipped}, manifest={output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
