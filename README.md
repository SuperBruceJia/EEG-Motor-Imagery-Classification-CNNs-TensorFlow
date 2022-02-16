## EEG Motor Imagery Signals (Tasks) Classification via Convolutional Neural Networks (CNN)

**Author**: Shuyue Jia, Northeast Electric Power University, China.

**Date**: December of 2018

### Download Paper
[A novel approach of decoding EEG four-class motor imagery tasks via scout ESI and CNN](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta)

NOTICE: The method in our paper is EEG source imaging (ESI) + Morlet wavelet joint time-frequency analysis (JTFA) + Convolutional Neural Networks (CNNs). The raw data has been processed using the Matlab Toolkit [Brainstorm](https://neuroimage.usc.edu/brainstorm/). My job is using CNNs to classify the EEG data after the ESI + JTFA process. The Dataset (.mat Files) preprocessed via the ESI + JTFA process can be found via the [Shared Google Drive](https://drive.google.com/drive/folders/1go7_ba_JNG4shob9MJ826gKY8sDSw3jJ?usp=sharing). The corresponding preprocessed Excel files can be downloaded from the [Shared Google Drive](https://drive.google.com/drive/folders/18bMOLP0ENA5RK1gBuO78IYGXm_PupsCW?usp=sharing).

Meanwhile, the codes in this repository are based on the raw EEG data without the ESI and JTFA process, and can also achieve a good result. The main CNNs Tensorflow framework codes in the "MI_Proposed_CNNs_Architecture.py" are the same for both of the works.

---

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture1.jpg" alt="Project1">
</div>

---

### Installation and Usage

1. Python file: PhysioNet_MI_Dataset/MIND_Get_EDF.py

    --- download all the EEG Motor Movement/Imagery Dataset .edf files from [here](https://archive.physionet.org/pn4/eegmmidb/)!

    ```
    (Under Any Python Environment) $ python MIND_Get_EDF.py
    ```

2. Python file: Read_Raw_Data_Save_Into_Matlab_Files.py

    --- Read the edf Raw data of different channels and save them into matlab .m files

    --- At this stage, the Python file must be processed under a Python 2 environment (I recommend to use Python 2.7 version).

    ```
    (Under Python 2.7 Environment) $ python Read_Raw_Data_Save_Into_Matlab_Files.py
    ```

3. Matlab file: Saved_Matlab_Data/Preprocessing_Raw_Data.m

    --- Pre-process the dataset (Data Normalization mainly) and save matlab .m files into Excel .xlsx Files

4. Python file: MI_Proposed_CNNs_Architecture.py

    --- the proposed CNNs architecture 

    --- based on TensorFlow 1.12.0 with CUDA 9.0 or TensorFlow 1.13.1 with CUDA 10.0

    --- The trained results are saved in the Tensorboard

    --- Open the Tensorboard and save the results into Excel .csv files

    --- Draw the graphs using Matlab or Origin

    ```
    (Under Python 3.6 Environment) $ python MI_Proposed_CNNs_Architecture.py
    ```

### Structure of the code
At the root of the project, you will see:

```text
├── PhysioNet_MI_Dataset
|  └── MIND_Get_EDF.py
├── Read_Raw_Data_Save_Into_Matlab_Files.py
├── Saved_Matlab_Data
|  └── Preprocessing_Raw_Data.m
├── MI_Proposed_CNNs_Architecture.py
├── electrode_positions.txt
```

### Citation
If you find our work useful in your research, please consider citing in your publications. We provide a BibTeX entry below.

```
@article{hou2020novel,
	title={A novel approach of decoding EEG four-class motor imagery tasks via scout ESI and CNN},
	author={Hou, Yimin and Zhou, Lu and Jia, Shuyue and Lun, Xiangmin},
	journal={Journal of neural engineering},
	volume={17},
	number={1},
	pages={016048},
	year={Feb. 2020},
	publisher={IOP Publishing}
}
```

### Acknowledgment

We are very grateful to Prof. Yimin Hou due to his friendly guidance, and the research paper would not have happened without him.<br>
<a href="http://www.neepu.edu.cn/"> <img width="150" height="150" src="https://github.com/SuperBruceJia/EEG-DL/raw/master/NEEPU.png"></a>

