import os
import shutil
from typing import Optional
import numpy as np
import uuid
import cv2
from tqdm import tqdm
import ffmpeg

class TempDirectory:

    def __init__(self, root_path: str = "/tmp", dir_suffix: str = "vaetc_") -> None:
        
        self.root_path = root_path
        self.dir_name = None
        self.dir_suffix = str(dir_suffix)

    def path(self):

        return os.path.join(self.root_path, self.dir_suffix + self.dir_name)

    def __enter__(self):

        while self.dir_name is None or os.path.exists(self.path()):
            self.dir_name = str(uuid.uuid4())

        os.makedirs(self.path())

        return self

    def __exit__(self, exception_type, exception_value, traceback):
        
        shutil.rmtree(self.path(), ignore_errors=True)

def write_video(
    output_path: str,
    pixels: np.ndarray,
    size: Optional[tuple[int, int]] = None,
    framerate: int = 60,
    verbose: bool = False,
    overwrite: bool = True):
    """ Writes `pixels` as a h.264 MP4 video into `output_path`

    Args:
        output_path (str): the path to output the video at
        pixels (str): an ndarray with shape (frame, channels, height, width), channel RGB-ordered, float-valued in the range [0,1]
        size (Optional, :class:`tuple[int, int]`): resize to [width, height] if not None
        framerate (int): the framerate of a video
        verbose (bool): If True, progress is shown and ffmpeg is not quiet
        overwrite (bool): Overwrite if `output_path` exists.

    Notes:
        The shape of `pixels` are HW-ordered, but `size` is WH-ordered.
    """

    if not overwrite and os.path.exists(output_path):
        raise RuntimeError(f"{output_path} already exists")

    with TempDirectory() as td:

        for index, frame in enumerate(tqdm(pixels) if verbose else pixels):

            frame_path = os.path.join(td.path(), f"{index:d}.png")

            # convert frame from a torch-formatted image into a cv2-formatted
            frame = (frame * 255).astype(np.uint8) # [0,1] float -> [0,255] uint8
            frame = frame[::-1] # RGB -> BGR
            frame = frame.transpose(1, 2, 0) # CHW -> HWC

            cv2.imwrite(frame_path, frame)
        
        stream = ffmpeg.input(os.path.join(td.path(), r"%d.png"), r=framerate)
        if size is not None:
            w, h = size
            stream = stream.filter("scale", f"{w}x{h}")
        stream = ffmpeg.output(stream, output_path,
            vcodec="libx264", pix_fmt="yuv420p", strict=2, acodec="aac") # twitter-ready
        ffmpeg.run(stream, overwrite_output=True, quiet=not verbose)
