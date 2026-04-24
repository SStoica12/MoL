import argparse
import concurrent.futures
import json
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

try:
    import av
except ImportError:  # pragma: no cover
    av = None

try:
    from decord import VideoReader, cpu
except ImportError:  # pragma: no cover
    VideoReader = None
    cpu = None


NUM_SEGMENTS = 1


def parse_args():
    parser = argparse.ArgumentParser(description="Extract middle frames for SEED-Bench video questions.")
    parser.add_argument("--annotation-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--something-something-dir", type=str, required=True)
    parser.add_argument("--epic-kitchens-dir", type=str, required=True)
    parser.add_argument("--breakfast-dir", type=str, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    return parser.parse_args()


def get_index(num_frames, num_segments):
    if num_frames <= 0:
        return np.array([0], dtype=np.int64)
    if num_segments > num_frames:
        return np.array([idx for idx in range(num_frames)], dtype=np.int64)
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    return np.array([start + int(np.round(seg_size * idx)) for idx in range(num_segments)], dtype=np.int64)


def resolve_video_path(args, question):
    qtype = int(question["question_type_id"])
    data_id = str(question["data_id"])
    if qtype == 10:
        return Path(args.something_something_dir) / data_id
    if qtype == 11:
        return Path(args.epic_kitchens_dir) / Path(data_id).name
    if qtype == 12:
        return Path(args.breakfast_dir) / data_id
    raise ValueError(f"Unsupported SEED video question_type_id={qtype}")


def decode_with_decord(video_path, segment):
    if VideoReader is None:
        raise RuntimeError("decord is required for SEED-Bench question types 10 and 11")
    vr = VideoReader(str(video_path), num_threads=1, ctx=cpu(0))
    video_len = len(vr)
    if video_len <= 0:
        raise RuntimeError(f"Video has zero frames: {video_path}")

    if segment is not None:
        fps = vr.get_avg_fps()
        start_frame = int(min(max(segment[0] * fps, 0), video_len - 1))
        end_frame = int(min(max(segment[1] * fps, 0), video_len - 1))
        if end_frame < start_frame:
            end_frame = start_frame
        total_frames = max(end_frame - start_frame + 1, 1)
        frame_indices = get_index(total_frames, NUM_SEGMENTS) + start_frame
    else:
        frame_indices = get_index(video_len, NUM_SEGMENTS)

    vr.seek(0)
    frame = vr.get_batch(frame_indices)[0].asnumpy()
    return Image.fromarray(frame)


def decode_with_pyav(video_path, segment):
    if av is None:
        raise RuntimeError("pyav is required for SEED-Bench question type 12")
    reader = av.open(str(video_path))
    frames = [frame.to_rgb().to_ndarray() for frame in reader.decode(video=0)]
    if not frames:
        raise RuntimeError(f"Video has zero decoded frames: {video_path}")

    start_frame = int(segment[0]) if segment is not None else 0
    end_frame = int(segment[1]) if segment is not None else len(frames) - 1
    end_frame = min(max(end_frame, start_frame), len(frames) - 1)
    total_frames = max(end_frame - start_frame + 1, 1)
    frame_indices = get_index(total_frames, NUM_SEGMENTS) + start_frame
    return Image.fromarray(frames[int(frame_indices[0])])


def extract_frame(args, question):
    video_path = resolve_video_path(args, question)
    segment = question.get("segment")
    qtype = int(question["question_type_id"])

    if qtype == 12:
        image = decode_with_pyav(video_path, segment)
    else:
        image = decode_with_decord(video_path, segment)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{question['question_type_id']}_{question['question_id']}.png"
    image.save(output_path)
    return str(output_path)


def main():
    args = parse_args()
    annotation = json.load(open(args.annotation_file, "r"))
    video_questions = [question for question in annotation.get("questions", []) if question.get("data_type") == "video"]

    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.num_workers, 1)) as executor:
        futures = {executor.submit(extract_frame, args, question): question for question in video_questions}
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            question = futures[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover
                failures.append((question.get("question_id"), str(exc)))

    print(f"Processed {len(video_questions)} SEED-Bench video questions")
    if failures:
        print(f"Failed to extract {len(failures)} video frames")
        for question_id, message in failures[:20]:
            print(f"  {question_id}: {message}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
