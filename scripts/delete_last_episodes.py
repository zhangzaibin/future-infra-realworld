#!/usr/bin/env python3
"""
删除 LeRobot 数据集中末尾的 N 条 episode。

用法:
    python delete_last_episodes.py --dataset ./datasets/pick_and_place --last 1
    python delete_last_episodes.py --dataset ./datasets/pick_and_place --last 3

只支持删除末尾连续的 episode。从中间删除请使用 lerobot-edit-dataset。
"""

import argparse, json, subprocess, shutil
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser(description="Delete last N episodes from a LeRobot dataset")
    parser.add_argument("--dataset", required=True, help="Path to dataset root directory")
    parser.add_argument("--last", type=int, required=True, help="Number of episodes to delete from the end")
    parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    root = Path(args.dataset).resolve()
    meta_dir = root / "meta"
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    data_dir = root / "data" / "chunk-000"
    video_dir = root / "videos"
    info_path = meta_dir / "info.json"

    # ---- 读取 info ----
    info = json.loads(info_path.read_text())
    fps = info["fps"]
    total_ep = info["total_episodes"]
    first_del = total_ep - args.last  # 0-indexed

    if first_del < 0:
        print(f"错误: 要删除 {args.last} 条，但总共只有 {total_ep} 条")
        return 1

    del_set = set(range(first_del, total_ep))
    print(f"数据集: {root}")
    print(f"当前 episode 总数: {total_ep}")
    print(f"待删除: ep {first_del} ~ ep {total_ep - 1}（共 {args.last} 条）")
    print()

    # ---- 扫描 camera keys ----
    cam_keys = [d.name for d in sorted(video_dir.iterdir()) if d.is_dir()]
    print(f"相机: {cam_keys}")

    # ---- 读取所有 meta episodes，建立映射 ----
    meta_files = sorted(episodes_dir.glob("file-*.parquet"))
    data_files = sorted(data_dir.glob("file-*.parquet"))

    # {ep: {cam_key: (chunk, file, from_ts, to_ts)}}
    ep_video = defaultdict(dict)
    all_meta_eps = set()

    for mf in meta_files:
        pf = pq.ParquetFile(str(mf))
        cols = ["episode_index", "length"]
        for cam in cam_keys:
            cols += [f"videos/{cam}/chunk_index", f"videos/{cam}/file_index",
                     f"videos/{cam}/from_timestamp", f"videos/{cam}/to_timestamp"]
        for rg_idx in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg_idx, columns=cols)
            ep_arr = t.column("episode_index").to_pylist()
            for i, ep in enumerate(ep_arr):
                all_meta_eps.add(ep)
                for cam in cam_keys:
                    try:
                        ci = t.column(f"videos/{cam}/chunk_index")[i].as_py()
                        fi = t.column(f"videos/{cam}/file_index")[i].as_py()
                        ft = t.column(f"videos/{cam}/from_timestamp")[i].as_py()
                        tt = t.column(f"videos/{cam}/to_timestamp")[i].as_py()
                        ep_video[ep][cam] = (ci, fi, ft, tt)
                    except Exception:
                        pass

    # ---- 检查是否所有待删除 ep 都在 meta 中 ----
    missing = del_set - all_meta_eps
    if missing:
        print(f"警告: 以下 episode 在 meta 中不存在，将跳过: {sorted(missing)}")
        del_set -= missing

    if not del_set:
        print("没有需要删除的 episode")
        return 0

    # ---- 分析视频文件: 哪些需要整文件删除 vs 裁剪 ----
    video_to_delete = set()      # {(cam, chunk, file)}
    video_to_trim = {}           # {(cam, chunk, file): cut_off_ts}

    for cam in cam_keys:
        # 找出受影响的视频文件
        affected = defaultdict(list)  # {(chunk, file): [episodes_to_delete_in_this_file]}
        for ep in del_set:
            if cam in ep_video[ep]:
                ci, fi, _, _ = ep_video[ep][cam]
                affected[(ci, fi)].append(ep)

        for (ci, fi), del_eps_in_file in affected.items():
            # 该视频文件中所有 episode
            all_eps_in_file = [ep for ep, v in ep_video.items()
                               if cam in v and v[cam][0] == ci and v[cam][1] == fi]

            if set(all_eps_in_file).issubset(del_set):
                video_to_delete.add((cam, ci, fi))
                print(f"  {cam}: file-{fi} → 整文件删除 (ep {sorted(all_eps_in_file)})")
            else:
                kept = sorted(set(all_eps_in_file) - del_set)
                if kept and max(kept) < min(del_eps_in_file):
                    # 删除在末尾，可以裁剪
                    last_kept = max(kept)
                    _, _, _, cut_ts = ep_video[last_kept][cam]
                    video_to_trim[(cam, ci, fi)] = cut_ts
                    print(f"  {cam}: file-{fi} → 在 {cut_ts:.1f}s 裁剪 (保留 ep {kept}, 删除 ep {sorted(del_eps_in_file)})")
                else:
                    print(f"  错误: {cam} file-{fi} 删除的 ep 不在文件末尾!")
                    print(f"    请使用 lerobot-edit-dataset")
                    return 1

    # ---- 计算总帧数 ----
    total_del_frames = 0
    for mf in meta_files:
        pf = pq.ParquetFile(str(mf))
        for rg_idx in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg_idx, columns=["episode_index", "length"])
            for ep, length in zip(t.column("episode_index").to_pylist(),
                                  t.column("length").to_pylist()):
                if ep in del_set:
                    total_del_frames += length

    print(f"\n将删除 {total_del_frames} 帧")
    print(f"整文件删除: {len(video_to_delete)} 个视频")
    print(f"裁剪视频:   {len(video_to_trim)} 个视频")
    print()

    # ---- 确认 ----
    if not args.yes:
        resp = input("确认删除? [y/N] ")
        if resp.lower() != "y":
            print("已取消")
            return 0

    # ---- Step 1: 处理视频 ----
    for cam, ci, fi in video_to_delete:
        vp = video_dir / cam / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        if vp.exists():
            print(f"删除视频: {vp}")
            vp.unlink()

    for (cam, ci, fi), cut_ts in video_to_trim.items():
        src = video_dir / cam / f"chunk-{ci:03d}" / f"file-{fi:03d}.mp4"
        tmp = src.with_suffix(".tmp.mp4")
        if not src.exists():
            continue
        print(f"裁剪视频 {src.name} 到 {cut_ts:.1f}s ...")
        result = subprocess.run([
            "ffmpeg", "-y", "-i", str(src),
            "-to", str(cut_ts),
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "copy", "-movflags", "+faststart",
            str(tmp)
        ], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ffmpeg 错误:\n{result.stderr}")
            return 1
        tmp.replace(src)
        print(f"  完成: {src}")

    # ---- Step 2: 处理 meta episodes parquet（按行过滤，不是按 row group） ----
    for mf in meta_files:
        src = episodes_dir / mf.name
        table = pq.read_table(str(src))
        # 过滤掉待删除的 episode
        ep_col = table.column("episode_index").to_pylist()
        keep_mask = [ep not in del_set for ep in ep_col]
        if all(keep_mask):
            continue  # 没有要删的
        if not any(keep_mask):
            print(f"删除 meta: {src}")
            src.unlink()
            continue
        filtered = table.filter(pa.array(keep_mask))
        tmp = src.with_suffix(".tmp.parquet")
        pq.write_table(filtered, str(tmp))
        tmp.replace(src)
        print(f"更新 meta: {src} ({sum(keep_mask)}/{len(keep_mask)} 条保留)")

    # ---- Step 3: 处理 data parquet（按行过滤） ----
    for df in data_files:
        src = data_dir / df.name
        table = pq.read_table(str(src))
        ep_col = table.column("episode_index").to_pylist()
        keep_mask = [ep not in del_set for ep in ep_col]
        if all(keep_mask):
            continue
        if not any(keep_mask):
            print(f"删除 data: {src}")
            src.unlink()
            continue
        filtered = table.filter(pa.array(keep_mask))
        tmp = src.with_suffix(".tmp.parquet")
        pq.write_table(filtered, str(tmp))
        tmp.replace(src)
        print(f"更新 data: {src} ({sum(keep_mask)}/{len(keep_mask)} 帧保留)")

    # ---- Step 4: 更新 info.json ----
    info["total_episodes"] = info["total_episodes"] - len(del_set)
    info["total_frames"] = info["total_frames"] - total_del_frames
    info["splits"] = {"train": f"0:{info['total_episodes']}"}
    info_path.write_text(json.dumps(info, indent=4, ensure_ascii=False))
    print(f"\ninfo.json 已更新: total_episodes={info['total_episodes']}, total_frames={info['total_frames']}")

    # ---- 验证 ----
    print("\n=== 验证 ===")
    meta_final = set()
    for mf in sorted(episodes_dir.glob("file-*.parquet")):
        t = pq.read_table(str(mf), columns=["episode_index"])
        meta_final.update(t.column("episode_index").to_pylist())

    data_final = set()
    for df in sorted(data_dir.glob("file-*.parquet")):
        t = pq.read_table(str(df), columns=["episode_index"])
        data_final.update(t.column("episode_index").to_pylist())

    ok = len(meta_final) == len(data_final) == info["total_episodes"]
    print(f"meta: {len(meta_final)} 条, data: {len(data_final)} 条, info: {info['total_episodes']} 条")
    print(f"一致: {'✓' if ok else '✗'}")
    if not ok:
        print(f"  meta 多余: {sorted(meta_final - data_final)}")
        print(f"  data 多余: {sorted(data_final - meta_final)}")

    return 0 if ok else 1


if __name__ == "__main__":
    exit(main())
