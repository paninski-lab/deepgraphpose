from pathlib import Path
import cv2
from skimage import io
from skimage.util import img_as_ubyte
import numpy as np
from moviepy.editor import VideoFileClip
import os

from deeplabcut.utils import auxiliaryfunctions


def local_extract_frames(
    config,
    frames2pick,
    crop=False,
    opencv=True,
    full_path=False,
):
    # crop ignores True
    # reads only first video
    # edited from deeplabcut.generate_training_dataset/frame_extraction.py

    config_file = Path(config).resolve()
    cfg = auxiliaryfunctions.read_config(config_file)
    print("Config file read successfully.")
    # forced to read only 1st video
    videos = cfg["video_sets"].keys()
    video = list(videos)[0]
    print(video)
    output_path = Path(config).parents[0] / "labeled-data" / Path(video).stem

    coords = cfg['video_sets'][video]['crop'].split(',')

    if not full_path:
        video = os.path.join(cfg['project_path'], video)

    print(video)
    if opencv:
        cap = cv2.VideoCapture(video)
        fps = cap.get(
            5
        )  # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
        nframes = int(cap.get(7))
        duration = nframes * 1.0 / fps
    else:
        #Moviepy:
        clip = VideoFileClip(video)
        fps = clip.fps
        duration = clip.duration
        nframes = int(np.ceil(clip.duration * 1. / fps))

    indexlength = int(np.ceil(np.log10(nframes)))

    print(fps, nframes, duration, indexlength)

    if opencv:
        for index in frames2pick:
            cap.set(1, index)  # extract a particular frame
            ret, frame = cap.read()
            if ret:
                image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_name = str(output_path / "img" + str(index).zfill(indexlength) + ".png")

                if crop:
                    io.imsave(
                        img_name,
                        image[int(coords[2]):int(coords[3]),
                              int(coords[0]):int(coords[1]), :, ],
                    )  # y1 = int(coords[2]),y2 = int(coords[3]),x1 = int(coords[0]), x2 = int(coords[1]
                else:
                    io.imsave(img_name, image)
            else:
                print("Frame", index, " not found!")
        cap.release()
    else:
        for index in frames2pick:
            try:
                image = img_as_ubyte(clip.get_frame(index * 1.0 / clip.fps))
                img_name = str(output_path / "img" + str(index).zfill(indexlength) + ".png")
                io.imsave(img_name, image)
                if np.var(image) == 0:  # constant image
                    print(
                        "Seems like black/constant images are extracted from your video. Perhaps consider using opencv under the hood, by setting: opencv=True"
                    )

            except FileNotFoundError:
                print("Frame # ", index, " does not exist.")

        # close video.
        clip.close()
    return


def local_extract_frames_md(
    config_path,
    frames2pick,
    video,
    crop=False,
    opencv=True,
    full_path=False,
):
    # crop ignores True
    # reads only first video
    # edited from deeplabcut.generate_training_dataset/frame_extraction.py

    cfg = auxiliaryfunctions.read_config(config_path)
    rel_video_path = str(Path.resolve(Path(video)))
    videoname = Path(rel_video_path).name

    model_video = str(Path(config_path).parent / 'videos' / videoname)
    output_path = Path(config_path).parents[0] / 'labeled-data' / Path(
        video).stem
    coords = cfg['video_sets'][model_video]['crop'].split(',')

    cap = cv2.VideoCapture(rel_video_path)
    fps = cap.get(
        5
    )  # https://docs.opencv.org/2.4/modules/highgui/doc/reading_and_writing_images_and_video.html#videocapture-get
    nframes = int(cap.get(7))
    duration = nframes * 1.0 / fps

    indexlength = int(np.ceil(np.log10(nframes)))

    print(fps, nframes, duration, indexlength)

    if opencv:
        for index in frames2pick:
            cap.set(1, index)  # extract a particular frame
            ret, frame = cap.read()
            if ret:
                image = img_as_ubyte(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                img_name = str(output_path / "img" + str(index).zfill(indexlength) + ".png")
                if crop:
                    io.imsave(
                        img_name,
                        image[int(coords[2]):int(coords[3]),
                              int(coords[0]):int(coords[1]), :, ],
                    )  # y1 = int(coords[2]),y2 = int(coords[3]),x1 = int(coords[0]), x2 = int(coords[1]
                else:
                    io.imsave(img_name, image)
            else:
                print("Frame", index, " not found!")
        cap.release()
    else:
        for index in frames2pick:
            try:
                image = img_as_ubyte(clip.get_frame(index * 1.0 / clip.fps))
                img_name = str(output_path / "img" + str(index).zfill(indexlength) + ".png")
                io.imsave(img_name, image)
                if np.var(image) == 0:  # constant image
                    print(
                        "Seems like black/constant images are extracted from your video. Perhaps consider using opencv under the hood, by setting: opencv=True"
                    )

            except FileNotFoundError:
                print("Frame # ", index, " does not exist.")

        # close video.
        clip.close()

    return
