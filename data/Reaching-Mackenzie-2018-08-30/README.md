This data is from [Mackenzie Mathis](https://github.com/MMathisLab), adopted from the DeepLabCut repo: https://github.com/DeepLabCut/DeepLabCut/tree/master/examples.
It is not meant to be used to test the performance of DeepLabCut, it is supplied for demo data only, which can be loaded as follows directly from DeepLabCut.

```python
import deeplacbut
path_config_file = os.path.join(os.getcwd(),'Reaching-Mackenzie-2018-08-30/config.yaml')
deeplabcut.load_demo_data(path_config_file)
```

Note, it is from the following publication, so if you use this data please cite it:

Mathis et al. Neuron 2017: Somatosensory Cortex Plays an Essential Role in
Forelimb Motor Adaptation in Mice https://pubmed.ncbi.nlm.nih.gov/28334611/ 
