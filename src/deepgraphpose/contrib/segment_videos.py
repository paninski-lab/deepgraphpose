import os
from moviepy.editor import VideoFileClip
# Given a DLC project, check all videos and clip those that are longer than some tolerance into shorter disjoint clips. 

# Take the project, and look within for all videos that will be trained on (these come from the config file) and analyzed (these come from the folder videos_dgp). 
# Those videos that have labels need to be split on the training labels as well. 

def split_video_and_trainframes(config_path,tol=5000,suffix=None):    
    """Splits videos and trainframes in a model config that are larger than some tolerance in frames.  

    :param config_path: parameter to config file. 
    :param tol: tolerance in number of frames. 
    :param suffix: video suffix. 
    """
    trainvids = check_videos(config_path,tol)
    analyzevids = check_analysis_videos(folder,tol)
    splitlength = tol
    vids = trainvids + analyzevids
    for v in vids:
        split_video(v,splitlength,suffix)
        if v in trainvids:
            format_frames(v,splitlength,suffix)

def check_videos(config_path,tol):
    """Checks all videos given in the model cfg file and checks if any are longer than the given length. 

    :param config_path: parameter to config file. 
    :param tol: tolerance in number of frames. 
    """

def check_analysis_videos(folder_path,tol):
    """Checks all videos given in the videos_dgp directory and checks if any are longer than the given length. 

    :param config_path: parameter to config file. 
    :param tol: tolerance in number of frames. 
    """

def split_video(vidpath,splitlength,suffix = "",outputloc = None):
    """splits a given video into subclips of type mp4. Note: will work best (even frames per subclip) if you pass a splitlength that is divisible by your frame rate.  

    :param vidpath: path to video
    :param splitlength: length to chunk into in frames
    :param suffix: custom suffix to add to subclips
    :param outputloc: directory to write outputs to. Default is same directory. 
    :returns: list of paths to new video files. 
    """
    try:
        clip = VideoFileClip(vidpath)
    except FileNotFoundError:
        print("file not found.")

    duration = clip.duration     
    splitlength_secs = splitlength/clip.fps     
    viddir,vidname = os.path.dirname(vidpath),os.path.basename(vidpath)
    base,ext = os.path.splitext(vidname)
    subname = base+suffix+"{n}"+".mp4"
    if outputloc is None:
        subpath = os.path.join(viddir,subname)
    else:    
        subpath = os.path.join(outputloc,subname)
       
    clipnames = []   
    clipstart = 0
    clipind = 0
    while clipstart < duration:
        subname = subpath.format(n=clipind)
        subclip = clip.subclip(clipstart,min(duration,clipstart+splitlength_secs))
        subclip.write_videofile(subname,codec = "mpeg4")
        clipnames.append(subname)
        clipstart += splitlength_secs
        clipind+=1
    return clipnames    


    

def format_frames(vidpath,splitlength,suffix = None):    
    """reformats training frames into format that matches sublclips

    :param vidpath: path to video
    :param splitlength: length to chunk into 
    :param suffix: custom suffix to add to subclips
    """
    

