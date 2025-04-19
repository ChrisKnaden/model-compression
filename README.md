# Walkthrough
> [!TIP]
> TL;DR: [Google Colab Notebook](https://colab.research.google.com/drive/16PizKU4S3kicK-MoaDQY8-Ufou3Msc_Y?usp=sharing) or
>
> Create a virtual Python environment with version 3.11 or higher, install the dependencies using `pip install -r requirements.txt`, and run the notebook [notebooks/resnet110_combined_model_compression.ipynb](https://github.com/ChrisKnaden/model-compression/blob/main/notebooks/resnet110_combined_model_compression.ipynb) locally. Depending on your local GPU, you may need to adjust the corresponding configuration parameter.

### Execution in Google Colab
[Here](https://colab.research.google.com/drive/16PizKU4S3kicK-MoaDQY8-Ufou3Msc_Y?usp=sharing) is a Google Colab jupyter notebook to get you started. Note that the execution times, especially for calculations done on the CPU, are really slow. To at least speed up the calculations possible on the GPU, you need to change your runtime type after initialisation. [Here](https://www.youtube.com/watch?v=-9CLfrZISRw) is a tutorial on how to do that. To use this google colab notebook, copy it into your google colab folder. After that you can run the cells. Please note that the installation dependencies takes about 3-4 minutes.


### Local Execution
If you are not familiar with jupyter notebook, have not installed it or don't know how, you can try it with these [Windows](https://www.youtube.com/watch?v=2WL-XTl2QYI), [Linux](https://www.digitalocean.com/community/tutorials/how-to-set-up-jupyter-notebook-with-python-3-on-ubuntu-18-04) or [MacOS](https://medium.com/@kishanck/virtual-environments-for-jupyter-notebooks-847b7a3b4da0) tutorials. After installing Jupyter Notebook, you need to create a virtual environment. Please note that the Python version should be 3.11 or above. After that you have to run the following command in your virtual environment to install the dependencies:

    pip install -r requirements.txt

Then you can use the different types of notebooks. [This](https://github.com/ChrisKnaden/model-compression/blob/main/notebooks/resnet110_combined_model_compression.ipynb) notebook is the main one, which demonstrates pruning, quantization, knowledge distillation and their combination. 

## Project Structure

**notebooks**: different test pipelines to play around with. [This](https://github.com/ChrisKnaden/model-compression/blob/main/notebooks/resnet110_combined_model_compression.ipynb) notebook is the main one, which demonstrates pruning, quantization, knowledge distillation and their combination. 

**src**: Helper functions for the model, training, evaluation, data loading and other utilities.

**models**: Pretrained models to test the pipelines. They are "selfmade"-models, therefore, they may not achieve the best accuracy results. The naming conventions indicate the model compression technique. For example: pruned_45-30_kd_10_resnet110_mps.pth means ch_sparsity = 0.45, 30 epochs normal training, 10 epochs knowledge distillation training, the base model was ResNet110 and it was trained on and safed as the MPS GPU.

# Credits

https://github.com/akamaster/pytorch_resnet_cifar10/tree/master

https://github.com/VainF/Torch-Pruning

https://github.com/haitongli/knowledge-distillation-pytorch
