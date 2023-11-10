DGP demo script
---------------

You can run the DGP pipeline on the example dataset provided in this repo by running the following command *from the main DGP directory*:

```python
python demo/run_dgp_demo.py --dlcpath data/Reaching-Mackenzie-2018-08-30
```

The output of the pipeline, including the labeled videos and the h5/csv files with predicted trajectories will be stored in `{DGP_DIR}/data/Reaching-Mackenzie-2018-08-30/videos_pred`. You can see information about training statistics in the file `{DGP_DIR}/data/dlc-models/iteration-0/ReachingAug30-trainset95shuffle1/train/learning_stats.csv`.

You can run the DGP pipeline on your own dataset as long as it exists in a DLC file directory structure, for example

```
task-scorer-date
├── dlc-models
│	└── ...
├── labeled-data
│	└── ...
├── training-datasets
│	└── ...
├── videos
│	└── ...
└── config.yaml
```

In particular, you can use the DLC GUI to create a DLC project, label videos, and create training datasets. Then you can run the demo code:

`python {DGP_DIR}/demo/run_dgp_demo.py --dlcpath {PROJ_DIR}/task-scorer-date/ --shuffle 'the shuffle to run' --dlcsnapshot 'specify the DLC snapshot if you've already run DLC with location refinement'`

If you have not yet run DLC you can simply remove the `--dlcsnapshot` argument and DLC will automatically be fit as part of the pipeline.
