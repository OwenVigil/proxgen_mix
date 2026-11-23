"""Build compact uint8 observation and action caches from a GenDGRL archive."""
import argparse
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np


def prepare(archive, output, max_episodes=None):
    output.mkdir(parents=True, exist_ok=False)
    levels, episode_ids = [], []
    archive_levels = set()
    total = 0
    # Optionally accept a pipe from a parallel decompressor without staging the tar.
    source_tar = (tarfile.open(fileobj=sys.stdin.buffer, mode="r|")
                  if archive == Path("-") else tarfile.open(archive, "r|xz"))
    with (
        source_tar as source,
        (output / "rgb.bin").open("wb") as images,
        (output / "action.bin").open("wb") as actions,
    ):
        for member in source:
            if not member.isfile() or not member.name.endswith(".npy"):
                continue
            _, episode_id, _, level, _ = Path(member.name).stem.rsplit("_", 4)
            level, episode_id = int(level), int(episode_id)
            with source.extractfile(member) as payload:
                # The official archive contains pickled dictionaries. Only use trusted data.
                data = np.load(io.BytesIO(payload.read()), allow_pickle=True).item()
            act = data["actions"].reshape(-1)
            rgb = data["observations"]
            if not len(act):
                continue
            if rgb.shape != (len(act) + 1, 3, 64, 64) or not 0 <= rgb.min() <= rgb.max() <= 255:
                raise ValueError((member.name, rgb.shape, rgb.dtype, len(act)))
            if act.min() < 0 or act.max() >= 15 or not np.issubdtype(act.dtype, np.integer):
                raise ValueError((member.name, act.dtype, act.min(), act.max()))
            if not 0 <= level < 200:
                raise ValueError(f"Unexpected training level {level}")
            rgb[:-1].transpose(0, 2, 3, 1).astype(np.uint8).tofile(images)
            act.astype(np.uint8).tofile(actions)
            levels.append((level, len(act)))
            episode_ids.append(episode_id)
            archive_levels.add(level)
            total += len(act)
            if max_episodes is not None and len(levels) == max_episodes:
                break
    if archive == Path("-") and max_episodes is None:
        while sys.stdin.buffer.read(2**20):
            pass
    if not total or len(set(episode_ids)) != len(episode_ids):
        raise ValueError("Missing transitions or duplicate episode IDs")
    np.save(output / "episodes.npy", np.asarray(levels, dtype=np.int32))
    meta = {"steps": total, "episodes": len(levels), "levels": len(archive_levels),
            "max_episodes": max_episodes}
    (output / "meta.json").write_text(json.dumps(meta) + "\n")
    print(meta, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path, help="XZ archive or '-' for an uncompressed stdin stream")
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-episodes", type=int, help="Archive prefix for smoke tests only")
    args = parser.parse_args()
    if args.max_episodes is not None and args.max_episodes < 1:
        parser.error("--max-episodes must be positive")
    prepare(args.archive, args.output, args.max_episodes)
