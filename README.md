# EHML: Extended Hybrid Machine Learning
This repository is dedicated to the implementation of the EHM algorithm as outlined in our manuscript currently under review. It also pertains to the fifth chapter of my PhD dissertation. EHML stands for Extended Hybrid Machine Learning, representing a novel, straightforward, multi-fidelity, and multiscale surrogate modeling technique. This technique facilitates both graph-based few-shot learning and zero-shot generalization in the context of knee cartilage biomechanics. Our investigation revealed that data augmentation plays a pivotal role in the performance of this model.

## Hardware and Software Requirements
- A standard CPU suffices for our experiments, although the code is compatible with GPU-based training as well.
- Abaqus 2021 is essential for FEA (Finite Element Analysis), alongside Visual Studio 2019 and Intel® Parallel Studio XE 2020 for running Fortran subroutines.
- Python 3 and several specific libraries are required for surrogate modeling and evaluation, as listed in the `environment.yml` file.

## Installation
Start by downloading and extracting the contents of the repository. Next, unzip `fea.zip`, which contains the Abaqus CAE file. Place this file and related FEA files, including `fea_core.py`, `2d_fea.py`, `3d_fea.py`, and `subroutines.for`, in its assumed default directory `C:\Temp\DA`. This directory is presumed to be within the default working directory of Abaqus (`C:\Temp`). The code frequently references this path, so any deviations from this directory structure necessitate corresponding updates in the code references. Python and Jupyter Notebook files, along with their dependent libraries, can be installed in any location, typically using a package manager like Conda.

## Dataset Generation
The dataset generation process involves scripts like `2d_fea.py` and `3d_fea.py`, which are directly executable within Abaqus. These scripts also generate necessary runtimes and metadata for subsequent data preparation. Finally, execute `tfrecorde_conversion.py` to convert these data into TFRecord files, which will be subsequently utilized by TensorFlow for modeling and analysis.

## Experiment Workflow and Expected Results
The experimental process begins with data collection and preprocessing, which is handled by the `transformation.ipynb` notebook. This notebook not only performs the necessary transformations but also analyzes the impact of these transformations on the data. Once preprocessing is complete, initiate the experiments using `run_experiment.py`. This script orchestrates the training process and stores the resultant models. Post-training, these models are loaded back into `transformation.ipynb` for evaluation. This evaluation includes analyzing results and generating data for Abaqus integration. The `visualize_fea.py` script is then used within Abaqus to visualize pointwise errors on the numerical models. This visualization aids in comparing different experimental conditions and understanding the overall performance gain.

## Experiment Customization
For those intending to customize the experiments, there are multiple options. Users can modify the source code, including even core files such as `fea_core.py` and `ml_core.py`, or simply alter the settings in the `experiment.json` file to define different experimental conditions. However, such customizations require a deep understanding of surrogate modeling principles in both Abaqus and TensorFlow. For individuals without this specialized knowledge, it is recommended to begin with our another <a href="https://github.com/shayansss/pmse" target="_blank">simpler project</a>. This project provides foundational knowledge that can be built upon before delving into more advanced modifications in this work.
