import requests
import os
import csv
import argparse
from tqdm import tqdm


def download(url, out):
    r = requests.get(url, headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36"
        }, allow_redirects=True, timeout=30, stream=True)
    if r.headers.get("Content-Type", "") != "video/mp4":
        raise ValueError(f"Content-Type is {r.headers.get('Content-Type')}, not video/mp4")
    if not os.path.exists(os.path.dirname(out)):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "wb") as f:
        f.write(r.content)


def main():
    parser = argparse.ArgumentParser(description="Download AVE-PM videos from Douyin links")
    parser.add_argument("--video_links", type=str, default="dataset/PM-400/video_links.csv",
                        help="Path to video_links.csv")
    parser.add_argument("--output_dir", type=str, default="dataset/PM-400/videos/",
                        help="Directory to save downloaded videos")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    fail_path = os.path.join(os.path.dirname(args.video_links), "fail_cases.csv")

    with open(args.video_links, "r") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    fail_cases = []
    for row in tqdm(rows, desc="Downloading videos"):
        video_id = row["video_id"]
        url = row["link"]
        out_path = os.path.join(args.output_dir, f"{video_id}.mp4")

        if os.path.exists(out_path):
            continue

        try:
            download(url, out_path)
        except Exception as e:
            print(f"Failed to download {video_id}: {e}")
            fail_cases.append({"video_id": video_id, "link": url, "error": str(e)})

    if fail_cases:
        with open(fail_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["video_id", "link", "error"])
            writer.writeheader()
            writer.writerows(fail_cases)
        print(f"{len(fail_cases)} downloads failed. See {fail_path}")
    else:
        print("All videos downloaded successfully.")


if __name__ == "__main__":
    main()
