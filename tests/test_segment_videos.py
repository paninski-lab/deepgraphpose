import pytest 
from deepgraphpose.contrib.segment_videos import split_video
from moviepy.editor import VideoFileClip
import math
import os 

here = os.path.abspath(os.path.dirname(__file__))
testmodel_path = os.path.join(here,"testmodel")

def test_split_video(tmp_path): 
    output = tmp_path/"subclips"
    output.mkdir()
    frame_duration = 30
    vidpath = os.path.join(testmodel_path,"videos","reachingvideo1.avi")
    video_locs = split_video(vidpath,frame_duration,suffix = "test",outputloc = str(output))
    
    origclip = VideoFileClip(vidpath)
    duration = origclip.duration*origclip.fps
    assert len(video_locs) == math.ceil(duration/frame_duration)
    vid_inds = []
    for vi in video_locs:
        prefix = os.path.splitext(os.path.basename(vidpath))[0]+"test"
        assert os.path.splitext(os.path.basename(vi))[0].startswith(prefix)
        vid_inds.append(int(vi.split(prefix)[-1].split(".mp4")[0]))
        sub = VideoFileClip(vi)
        assert sub.duration*sub.fps - frame_duration < 1e-1
    assert set(vid_inds) == set(range(len(video_locs)))    
    










