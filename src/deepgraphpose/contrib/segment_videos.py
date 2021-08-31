# Given a DLC project, check all videos and clip those that are longer than some tolerance into shorter disjoint clips. 

def split_video_and_trainframes(config_path,tol=1000,suffix):    
    """Splits videos and trainframes in a model config that are larger than some tolerance in frames.  

    :param config_path: parameter to config file. 
    :param tol: tolerance in number of frames. 
    """
    vids = check_videos(config_path,tol)
    splitlength = tol
    for v in vids:
        split_video(v,splitlength,suffix)
        format_frames(v,splitlength,suffix)

def check_videos(config_path,tol):
    """Checks all videos given in the model directory and checks if any are longer than the given length. 

    :param config_path: parameter to config file. 
    :param tol: tolerance in number of frames. 
    """

def split_video(vidpath,splitlength,suffix = None):
    """splits a given video into subclips. 

    :param vidpath: path to video
    :param splitlength: length to chunk into 
    :param suffix: custom suffix to add to subclips
    """

def format_frames(vidpath,splitlength,suffix = None):    
    """reformats training frames into format that matches sublclips

    :param vidpath: path to video
    :param splitlength: length to chunk into 
    :param suffix: custom suffix to add to subclips
    """
    

