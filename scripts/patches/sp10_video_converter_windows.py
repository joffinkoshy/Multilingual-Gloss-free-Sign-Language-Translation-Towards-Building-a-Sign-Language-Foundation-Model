"""Windows-compatible patch of MLSLT/SP-10's video_converter.py.

The upstream script (https://github.com/MLSLT/SP-10/blob/main/video_converter.py)
hardcodes a Unix temp path (`/tmp/sign.mp4`, doesn't exist on Windows) and
wraps ffmpeg's path arguments in single quotes (Windows' cmd.exe doesn't
treat single quotes as quoting characters the way bash does, so the quote
marks get passed to ffmpeg literally and it fails to find the file).

Usage: copy this file over the broken one in your local SP-10-video clone,
then run it exactly the same way as upstream:

    Copy-Item scripts\patches\sp10_video_converter_windows.py references\SP-10-video\video_converter.py
    cd references\SP-10-video
    python -X utf8 video_converter.py --save_path ../../data/raw/sp10

Everything below is identical to upstream except the two lines noted.
"""
import logging
import argparse
from pathlib import Path
import os
import cv2
import shutil
import tempfile

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main(args):
    root_path = Path(args.save_path)
    for video_path in root_path.rglob("*.mp4"):
        cap = cv2.VideoCapture(str(video_path))
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        if frame_rate == 25:
            continue
        logging.info(f"Start converting {video_path}, its frame rate is {frame_rate}")
        video_path = video_path.resolve()
        # PATCHED for Windows: cross-platform temp path instead of /tmp/,
        # double quotes instead of single quotes for the shell command.
        tmp_path = os.path.join(tempfile.gettempdir(), "sign.mp4")
        os.system(f'ffmpeg -i "{video_path}" -r 25 "{tmp_path}" -y')
        shutil.move(tmp_path, str(video_path))


if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "--save_path", help="Path to save the video", default="./videos", type=str
    )
    args = parse.parse_args()
    logging.info("Start converting SP-10 dataset")
    main(args)
    logging.info("Converting SP-10 dataset finished")
