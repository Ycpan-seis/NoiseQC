# NoiseQC
A python program developed for quality control of horizontal component ambient noise cross-correlation functions based on the Rayleigh wave phase-matching method.

**Yichen Pan**

University of Science and Technology of China

January, 26, 2025

This is a **_python_** program developed for quality control of horizontal component ambient noise cross-correlation functions (CCFs) based on the Rayleigh wave phase-matching method. Regarding the Rayleigh wave phase-matching method, you can refer to Baker & Stevens (2004) for the principles of this method applied to seismic events, and to Zha et al. (2013) for its principles when applied to noise data. About the details of our practice, you can refer to Pan & Yao (in prep.).

In this program, we provide the code to compute the correlation coefficient between and (or ) and perform quality control on the dispersion data, including the code to exhibit the figure. To help users better understand how to use the program, we also provide an example for the CCFs of a single pair of stations, followed by a statistical example for a set of data from a dense array. If there are any questions, comments, and bug reports, please feel free to email [pyc2020@mail.ustc.edu.cn](mailto:pyc2020@mail.ustc.edu.cn) or [ycpan03@gmail.com](mailto:ycpan03@gmail.com).

**References**

Baker, G. E., & Stevens, J. L. (2004). Backazimuth estimation reliability using surface wave polarization. _Geophysical Research Letters_, _31_(9), 2004GL019510. <https://doi.org/10.1029/2004GL019510>

Zha, Y., Webb, S. C., & Menke, W. (2013). Determining the orientations of ocean bottom seismometers using ambient noise correlation. _Geophysical Research Letters_, _40_(14), 3585–3590. <https://doi.org/10.1002/grl.50698>

Pan, Y., & Yao, H. (in prep). Quality Control Method for Horizontal-component Ambient Noise Cross-correlations and its Application to Radial Anisotropy Tomography of the Central Tanlu Fault Zone.

**Links**

1. NoiseQC_V1.0
2. [EGFAnalysisTimeFreq](https://yaolab.ustc.edu.cn/_upload/tpl/10/f0/4336/template4336/pdf/EGFAnalysisTimeFreq_version_2015.zip) (by Huajian Yao)
3. [NoiseCorr_SAC](https://yaolab.ustc.edu.cn/_upload/tpl/10/f0/4336/template4336/pdf/NoiseCorr_2016Jul_v4_2.zip) (by Huajian Yao)
4. [FastXC](https://github.com/wangkingh/FastXC) (by Jingxi Wang)

**1 Structure**

\- **_NoiseQC_**

1\. **_main.py_**, defines the basic classes and functions we used in this program

2\. **_run_single.py_**, (**Single Mode**) the example for calculating the QC parameters (including the correlation coefficient ( and ), optimal correction angle, SNR and lag-time () between and ) for the CCFs between a single pair of stations.

3\. **_run_statistic.py_**, (**Statistic Mode**) the statistical example showing how to conduct the quality control process and set the threshold of QC parameters.

**2 QuickStart**

This guide provides a brief introduction to program installation, data file preparation, and essential parameter configuration, ensuring an efficient start for users. **Chapter 3** and **4** will provide detailed explanations and examples of this program.

**2.1 Installation**

This program can be downloaded from [NoiseQC.](https://github.com/Ycpan-seis/NoiseQC)

**2.2 File Location**

In **Single Mode**, we utilize the **_tkinter_** package to provide an interactive interface for selecting the Z-Z component CCF of the station pair to be processed. The storage structure of the CCFs files follows the output format of [**_NoiseCorr_SAC_**](https://yaolab.ustc.edu.cn/_upload/tpl/10/f0/4336/template4336/pdf/NoiseCorr_2016Jul_v4_2.zip) (Yao et al., 2006, 2011), as shown below,

\- CFZ-E-N

  ├── E-E

  ├── E-N

  … … (nine components)

  └── Z-Z

      ├── ascii

      └── mat
users only need to select the CCF data in the Z-Z, while the other components can be read automatically.
In **Statistic Mode**, users should modify the following variables to select the path:

- **_Flag.data_path_**, the directory path of the folder containing the Z-Z component CCFs (the other will be read automatically)
- **_Flag.disp_path_**, the directory path of the folder containing the dispersion data for quality control.
- **_Flag.run_path_**, the directory path for running an example, contains **INPUT** and **OUTPUT** folders.
- **_stafile_**, file containing the information of the stations, with the default being ‘**./INPUT/sta_all.txt**’. (Format: station name; array name; latitude; longitude)

**2.3 Basic Parameters for a Quick Start**

The parameters are defined in the Para and Flag classes, and the meaning of each parameter can be found in the accompanying comments. This section will focus on the basic parameter settings required to run the code. For more advanced usage and customization to fit your own data, please refer to the detailed explanation in the following chapters.

**Para**

- **Sample_rate**, sample rate of CCFs data.
- **Vlow/Vhigh**, the lower/upper limit of group wave velocity
- **Ts/Te**, the period range; **dT**, the interval of period

**Flag**

- **Corr_Method**, 1: non-normalized correlation coefficient; 2: normalized
- **Auto**, 0: **Single Mode**; 1: **Statistic Mode**

**2.4 Input and Output**

Users need to prepare the following input files:

- Nine-component CCFs
- Dispersion data that requires quality control. (Only in **Statistic Mode**)
- Reference dispersion data to choose the period. (Only in **Single Mode** and **Period_setting** \= 1)

After preparing the input data and parameters, run **run_single.py** or **run_statistic.py**. In **Single Mode**, figures are output to **_run_path_**_/single_, and results are displayed in the terminal. and results are displayed in the terminal. In **Statistic Mode**, figures are output to **_run_path_**_/OUTPUT_.

  
