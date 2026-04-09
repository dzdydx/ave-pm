import argparse
from tqdm import tqdm
from pathlib import Path
from moviepy import VideoFileClip
import pandas as pd
import logging
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler("out.log"),
    logging.StreamHandler()
])
logger = logging.getLogger(__name__)


def cut_clips(args):
    annotation_csv = args.annotation_csv
    videos_dir = args.videos_dir
    out_videos_dir = args.out_videos_dir

    os.makedirs(out_videos_dir, exist_ok=True)

    processed_clips = set()
    if Path(out_videos_dir).exists():
        processed_clips = {clip.stem for clip in Path(out_videos_dir).glob("*.mp4")}
        logging.info(f"Restarting process. {len(processed_clips)} clips already processed.")

    df = pd.read_csv(annotation_csv)
    for i, row in tqdm(df.iterrows(), total=len(df)):
        if str(row["sample_id"]) in processed_clips:
            continue

        video_path = Path(videos_dir) / f"{row['video_id']}.mp4"
        if not video_path.exists():
            logger.warning(f"Video not found: {video_path}")
            continue

        try:
            clip = VideoFileClip(str(video_path))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cut video clips from PM-400 dataset for AVE-PM")
    parser.add_argument("--cut_clips", action="store_true",
                        help="Cut 10-second clips from original videos")

    parser.add_argument("--videos_dir", type=str, default="dataset/PM-400/videos",
                        help="Directory containing source videos")
    parser.add_argument("--annotation_csv", type=str, default="dataset/PM-400/PM400-to-AVEPM.csv",
                        help="Path to annotation CSV")
    parser.add_argument("--out_videos_dir", type=str, default="dataset/AVE-PM/videos",
                        help="Directory to save cut clips")

    args = parser.parse_args()

    if args.cut_clips:
        cut_clips(args)
