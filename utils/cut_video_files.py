import argparse
import sys
import math
import csv
from tqdm import tqdm
from pathlib import Path
from moviepy import VideoFileClip
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("out.log"),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)

input_annotation_path = "./annotations-finished.csv"
videos_dir = "/data1/datasets/PM-400/data/data"
out_videos_dir = "./data/videos"
out_annotation_path = "./data/annotations.csv"
label_csv = "./merged_pmv400.csv"
final_annotation_path = "./data/final_annotations.csv"

# prepare annotation df
label_df = pd.read_csv(label_csv)

df = pd.read_csv(input_annotation_path)
df = df[df['file_invalid'] == 0]
df = df.drop_duplicates(subset=['video_id'], keep='last') # only keep the last annotation for each video
df['event_duration'] = df['end_time'] - df['start_time']
df = df[df['event_duration'] >= 1] # only keep events with duration >= 1s

def split_video_into_segments(duration, start_time, end_time):
    if duration < 10:
        return [(0, duration)]

    def generate_segments(start, max_possible_clip_num):
        segments = []
        for i in range(max_possible_clip_num):
            segment_start = start + i * 10
            segment_end = segment_start + 10
            if segment_end <= duration:
                segments.append((segment_start, segment_end))
        return segments

    # 从前往后切分
    max_possible_clip_num = math.ceil((end_time - start_time) / 10)
    event_midpoint = (start_time + end_time) / 2
    start = max(0, event_midpoint - max_possible_clip_num * 10 / 2)
    if start + max_possible_clip_num * 10 > duration:
        max_possible_clip_num -= 1
    segments_forward = generate_segments(start, max_possible_clip_num)

    # 从后往前切分
    max_possible_clip_num = math.ceil((end_time - start_time) / 10)
    end = min(duration, event_midpoint + max_possible_clip_num * 10 / 2)
    if end - max_possible_clip_num * 10 < 0:
        max_possible_clip_num -= 1
    start = end - max_possible_clip_num * 10
    segments_backward = generate_segments(start, max_possible_clip_num)

    # 选择片段数最多的方案
    if len(segments_forward) >= len(segments_backward):
        return segments_forward
    else:
        return segments_backward

def generate_clip_annotations():
    sample_count = 0
    processed_videos = set()

    # 恢复进度
    if Path(out_annotation_path).exists():
        processed_videos_df = pd.read_csv(out_annotation_path)
        processed_videos = set(processed_videos_df['video_id'].unique())
        sample_count = processed_videos_df['sample_id'].max() + 1
        logging.info(f"Restarting process. {len(processed_videos)} videos already processed.")

    with open(out_annotation_path, "a") as f:
        writer = csv.writer(f)
        if sample_count == 0:
            writer.writerow(["sample_id", "video_id", "start", "end",
                             "event_start", "event_end", "duration",
                             "category", "onset", "offset", "type",
                             "haveBGM", "audio_irrelevant"])

        valid_df = df[df["file_invalid"] == 0]

        for i, row in tqdm(valid_df.iterrows(), total=len(valid_df)):
            if row["video_id"] in processed_videos:
                continue

            start_time = float(row["start_time"])
            end_time = float(row["end_time"])

            video_path = Path(videos_dir) / f"{row['video_id']}.mp4"
            if not video_path.exists():
                logger.warning(f"Video not found: {video_path}")
                continue

            category = label_df[label_df["video_id"] == row["video_id"]]["label"].values[0]

            try:
                sys.stdout = open('moviepy.log', 'a')
                clip = VideoFileClip(str(video_path))
                sys.stdout = sys.__stdout__

                duration = clip.duration

                segments = split_video_into_segments(duration, start_time, end_time)

                logger.debug(f"Video {row['video_id']} duration: {duration}, event duration: {end_time - start_time}, segments: {segments}")

                for j, (start, end) in enumerate(segments):
                    if start < 0 or end < 0 or start > duration or end > duration:
                        raise ValueError(f"Invalid segment: {start} - {end}")

                    # 计算事件在片段中的onset和offset
                    if start_time >= start and end_time <= end:
                        onset = round(start_time - start, 2)
                        offset = round(end_time - start, 2)
                        event_type = 1
                    elif start_time < start and end_time <= end:
                        onset = 0
                        offset = round(end_time - start, 2)
                        event_type = 2
                    elif start_time < start and end_time > end:
                        onset = 0
                        offset = round(end - start, 2)
                        event_type = 3
                    elif start_time >= start and end_time > end:
                        onset = round(start_time - start, 2)
                        offset = round(end - start, 2)
                        event_type = 4

                    writer.writerow([sample_count, row["video_id"], start, end,
                                     start_time, end_time, duration, category,
                                     onset, offset, event_type, row["haveBGM"],
                                     row["audio_irrelevant"]])
                    sample_count += 1

                clip.close()
                logger.info(f"Finished processing video {i} / {len(valid_df)}: {video_path}")
            except Exception as e:
                logger.error(f"Error processing video {i} / {len(valid_df)}: {video_path}. Error: {e}")

def cut_clips():
    processed_clips = set()

    # 恢复进度
    if Path(out_videos_dir).exists():
        processed_clips = {clip.stem for clip in Path(out_videos_dir).glob("*.mp4")}
        logging.info(f"Restarting process. {len(processed_clips)} clips already processed.")

    df = pd.read_csv(final_annotation_path)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if str(row["sample_id"]) in processed_clips:
            continue

        video_path = Path(videos_dir) / f"{row['video_id']}.mp4"
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            continue

        try:
            sys.stdout = open('moviepy.log', 'a')
            clip = VideoFileClip(str(video_path))
            sys.stdout = sys.__stdout__

            clip_name = f"{row['sample_id']}.mp4"
            clip_path = Path(out_videos_dir) / clip_name
            clip.subclipped(row["start"], row["end"]).write_videofile(
                str(clip_path),
                codec="libx264",
                audio=True,
                audio_fps=44100,
                audio_codec="aac",
                threads=8,
            )

            clip.close()
            logger.info(f"Finished processing clip {i}/{len(df)} in {row['video_id']}")
        except Exception as e:
            logger.error(f"Error processing clip {i}/{len(df)}: {clip_path}. Error: {e}")

# add arguments
parser = argparse.ArgumentParser()
parser.add_argument("--generate_clip_annotations", action="store_true", help="Generate clip level annotations from videos")
parser.add_argument("--cut_clips", action="store_true", help="Cut clips from videos")

args = parser.parse_args()

if __name__ == "__main__":
    if args.generate_clip_annotations:
        generate_clip_annotations()
    if args.cut_clips:
        cut_clips()