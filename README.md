# Deepgraphpose

This is the code for [Deep Graph Pose: a semi-supervised deep graphical model for improved animal pose tracking](https://www.biorxiv.org/content/10.1101/2020.08.20.259705v2). <br>

DGP is a semi supervised model which can run on top of other tracking algorithms, such as [DLC](https://www.nature.com/articles/s41593-018-0209-y).<br>
Since DLC developers have put in a lot of work into their GUI (and made it open source!), our algorithm can be run using the same filestructure as DLC.

If you have used DLC before, you can use the DGP pipeline to run DGP on top of these results! <br>
If you have not used DLC before, you can use the DGP pipeline to run DLC and DGP. <br>

Please see the installation instructions to install and run DGP on your videos. <br> 
Note: We have cloned the DLC package within the DGP repository, so we highly recommend that you install DGP in a new conda environment to avoid any conflicts with any concurrent DLC installation. <br> 


## Installation Instructions

To install DGP, navigate to desired installation directory "DGP_DIR" and clone the repository:
```
cd "{DGP_DIR}"
git clone https://github.com/paninski-lab/deepgraphpose.git
```
Follow the installation instructions [here](https://github.com/paninski-lab/deepgraphpose/tree/main/src/DeepLabCut/conda-environments) using the dgp*.yaml files, instead of the dlc*.yaml files. <br>
 
For example, if you are using Ubuntu OS with available GPUs, you can run the following:
```
cd deepgraphpose/src/DeepLabCut/conda-environments/
conda env create -f dgp-ubuntu-GPU.yaml
```
Activate the dgp conda environment and navigate to the parent directory to install the DLC clone inside DGP:
```
source activate dgp
cd ../
pip install -e .
```
Next, install DGP in dev mode:
```
cd ../..
pip install -e .
```

Install wx:
```
conda install -c anaconda wxpython
```

Check that both packages were installed:
```python
ipython
import deeplabcut 
print(deeplabcut.__path__)
['{DGP_DIR}/deepgraphpose/src/DeepLabCut/deeplabcut']
import deepgraphpose
print(deepgraphpose.__path__)
['{DGP_DIR}/deepgraphpose/src/deepgraphpose']
```
Finally, download the resnet weights to train the network:
```
curl http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz | tar xvz -C src/DeepLabCut/deeplabcut/pose_estimation_tensorflow/models/pretrained/
```
You may need to first install `curl` from the command line; if running ubuntu, the command is:
```
sudo apt install curl
```

### Remote server comments:
On a remote server, install open-cv without head by running:
```python
pip install opencv-python-headless
```

On a remote server, install dlc/dgp in light mode by running:
```python
export DLClight=True
```
before you begin the installation process of dgp.

## To Run DGP on your videos:
1. Use DLC's GUI to collect labels as described [here](https://github.com/paninski-lab/deepgraphpose/blob/main/src/DeepLabCut/docs/UseOverviewGuide.md). This step should create a project folder: "{PROJ_DIR}/task-scorer-date" and a folder with your labeled data  "{PROJ_DIR}/task-scorer-date/labeled-data". If you label frames from multiple videos, the GUI should automatically create folders corresponding to each video which contain *.csv, *.h5 files and *.png files with information from the manual labels.
2. Inside '{PROJ_DIR}/task-scorer-date', create a folder 'videos_dgp'.
```
cd '{PROJ_DIR}/task-scorer-date'
mkdir videos_dgp
```
3. Add to this folder other videos on which you want to run DGP during test time. Since DGP is a semi supervised model, it can exploit information from frames of videos with and without manually labeled markers during training time. You don't have to include all you videos but should include at least the most representative ones. The training time will increase proportional to the number of videos in this folder.
Note: We currently don't support running multiple passes of DGP.

4. Check the "bodyparts" and "skeleton" entries in "{PROJ_DIR}/task-scorer-date/config.yaml". For example, if I am tracking 4 fingers in each paw of a mice, my  "bodyparts" and "skeleton" entries will have the following form:
```
bodyparts:
- pinky_finger_r
- ring_finger_r
- middle_finger_r
- pointer_finger_r
skeleton:
- - pinky_finger_r
  - ring_finger_r
- - ring_finger_r
  - middle_finger_r
- - middle_finger_r
  - pointer_finger_r
```
Each item in "bodyparts" corresponds to a marker, and each item in "skeleton" corresponds to a pair of connected markers (the order in which the parts are listed does not matter). If you don't want to consider skeleton (no interconnected parts), leave that field empty "skeleton"

```
skeleton:

```
5. Run the DGP pipeline by running the following command:
```python
python ['{DGP_DIR}/demo/run_dgp_demo.py'] --dlcpath '{PROJ_DIR}/task-scorer-date/' --shuffle 'the shuffle to run' --dlcsnapshot 'specify the DLC snapshot if you\'ve already run DLC with location refinement'
```
*** You can run the demo with the example project in the test mode to check the runnability of the code
```python
python {DGP_DIR}/demo/run_dgp_demo.py --dlcpath data/Reaching-Mackenzie-2018-08-30 --test
```

6. The output of the pipeline, including the labeled videos and the pickle files with predicted trajectories will be stored in "{PROJ_DIR}/task-scorer-date/videos_pred".

## Try DGP on the cloud:
You can try DGP with GPUs on the cloud for free through a web interface using [NeuroCAAS](http://www.neurocaas.org). The current DGP implementation in NeuroCAAS is in beta mode so we are looking for feedback from beta users. If you are interested please email [neurocaas@gmail.com](mailto:neurocaas@gmail.com) once you sign up to NeuroCAAS to get started.




