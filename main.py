<<<<<<< HEAD
import numpy as np
import os
import math
import glob
import re
import sys
import shutil
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
#signal processing
from scipy.signal import hilbert,windows,butter,filtfilt,firwin,fftconvolve
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d

# --- Definition of some basic classes ---
class Para: #definition of parameters
    def __init__(self, Sample_rate=10, nsta=1, dv=0.002, Vlow=1.0, Vhigh=6, 
                Ts=0.6, dT=0.2, Te=10, Low_bound=-180, High_bound=180, 
                T_show=[], T_num=4, minSNR=6, minCorr=0.8, maxLag=1, maxAngle=30):
        self.dt = 1/Sample_rate                 
        self.Sample_rate = Sample_rate # Sample rate of the seismic data(Hz)
        self.nsta = nsta              # Number of stations
        self.dv = dv                  # Velocity interval of group velocity spectrum(no need to modify)
        self.Vlow = Vlow              # Signal window,the lower limit of group wave velocity
        self.Vhigh = Vhigh            # the higher limit of group wave velocity
        self.Ts = Ts                  # the start period of orientation correct
        self.dT = dT                  # the interval of period(usually has little effect on the results) 
        self.Te = Te                  # the end period of orientation correct
        self.Low_bound = Low_bound    # the angle search range,lower boundary
        self.High_bound = High_bound  # the higher boundary,the default interval is 1
        self.T_show = T_show          # the period for which you want to show its angle-correlation coefficient diagram
        self.T_num = T_num            # the window length of time variable filtering,refer to 'EGFAnalysisTimeFreq'
        self.minSNR = minSNR          # QC thresholds
        self.minCorr = minCorr
        self.maxLag = maxLag
        self.maxAngle=maxAngle
        self.T = self.generate_T()    # generate T array
    
    def generate_T(self):
        return np.arange(self.Ts, self.Te + self.dT, self.dT)

class Flag: #definition of methods
    def __init__(self,  Corr_Method=2, Half=0, 
                Save_png=1, Period_setting=0, Auto=0, Rotate=1):
        self.Corr_Method = Corr_Method          # 1: ZR*ZZ / (|ZR| * |ZZ|); 2: ZR*ZZ / |ZZ|^2
                                                # Method 2 is better: better noise resistance, more stable
        self.Half = Half                        # 1: Positive half-axis data; -1: Use negative half-axis data, virtual source and receiver flipped; 
                                                # 0: Average of positive and negative half-axis 
        self.Save_png = Save_png                # 1: Save image; 0: Do not save image
        self.Period_setting = Period_setting    # Period setting (Only works when Auto=0)
                                                # 1：Choose period based on dispersion file; 0: use the default Para.T
        self.Auto = Auto                        # 1: Auto mode,count the azimuth correction of all stations; 0: Normal mode,for single station
        self.Rotate = Rotate                    # Only works when Auto=1; 1: Rotate and save the data that meets the conditions; 0: Don't rotate
        self.run_path = './EXAMPLE'             # The path for running the code, contains input and output
        self.data_path = os.path.join(self.run_path, 'INPUT/CFs/Z-Z/ascii') # Seis data
        self.disp_path = os.path.join(self.run_path, 'INPUT/Disper')   # disper data that requires quality control
        self.ref_disp = os.path.join(self.run_path, 'INPUT/Disper_Ref')    # You need edit it only if Period_setting=1,the path of ZZ component dispersion file

class station: 
    def __init__(self, Lat=0, Lon=0, Name=''):
        self.Lat = Lat      # Latitude
        self.Lon = Lon      # Longitude
        self.Name = Name    # Name
        
class Seis_Data:# Cross-correlation data of the positive half
    def __init__(self):
        self.t_axis = []   # Horizontal axis
        self.ZZdata = []   
        self.ZNdata = []
        self.ZEdata = []
        self.NZdata = []
        self.NNdata = []
        self.NEdata = []
        self.EZdata = []
        self.ENdata = []
        self.EEdata = []

# --- Definition of some functions ---        
def DataRead(Flag):
    if Flag.Auto==0:
        #Open a dialog box and choose a Z-Z component file
        root = tk.Tk()
        #root.withdraw()          
        ZZ_path = filedialog.askopenfilename(
            title="Please selects a ZZ file.",
            filetypes=[("ZZ files", "*.dat"), ("All files", "*.*")]
        )
        if ZZ_path:
            print("User selects the ZZ file.:", ZZ_path)
        else:
            print("User cancels the operation to select a file.")
        root.destroy()
        FileName = os.path.basename(ZZ_path)
        
    elif Flag.Auto==1:
        #Alternatively,input the path of the directory
        Folder_path = Flag.data_path
        FileName = Flag.File
        ZZ_path = os.path.join(Folder_path, FileName)
    
    Disper_file = f'CDisp.T.{FileName}'
    
    #substitute the string to obtain the file path of other component
    ZN_path = ZZ_path.replace('Z-Z', 'Z-N').replace('ZZ', 'ZN')
    ZE_path = ZZ_path.replace('Z-Z', 'Z-E').replace('ZZ', 'ZE')
    NZ_path = ZZ_path.replace('Z-Z', 'N-Z').replace('ZZ', 'NZ')
    EZ_path = ZZ_path.replace('Z-Z', 'E-Z').replace('ZZ', 'EZ')
    EE_path = ZZ_path.replace('Z-Z', 'E-E').replace('ZZ', 'EE')
    EN_path = ZZ_path.replace('Z-Z', 'E-N').replace('ZZ', 'EN')
    NE_path = ZZ_path.replace('Z-Z', 'N-E').replace('ZZ', 'NE')
    NN_path = ZZ_path.replace('Z-Z', 'N-N').replace('ZZ', 'NN')
    
    #load data
    ZZ_data = np.loadtxt(ZZ_path)
    ZN_data = np.loadtxt(ZN_path)
    ZE_data = np.loadtxt(ZE_path)
    NZ_data = np.loadtxt(NZ_path)
    NN_data = np.loadtxt(NN_path)
    NE_data = np.loadtxt(NE_path)
    EZ_data = np.loadtxt(EZ_path)
    EN_data = np.loadtxt(EN_path)
    EE_data = np.loadtxt(EE_path)
    t_axis = ZZ_data[2:, 0] 
    seisData = Seis_Data()
    seisData.t_axis = t_axis
    
    #load station
    staSplit = FileName.split('_')
    staPair = staSplit[1]
    staCell = staPair.split('-')
    sta1 = staCell[0]
    sta2 = staCell[1]
    Source_sta=station()
    Receive_sta=station()
    # use positive axis data
    if Flag.Half == 1:
        Source_sta.Lat=ZZ_data[0, 1];Source_sta.Lon=ZZ_data[0, 0];Source_sta.Name=sta1
        Receive_sta.Lat=ZZ_data[1, 1];Receive_sta.Lon=ZZ_data[1, 0];Receive_sta.Name=sta2
        ZZ_posi = ZZ_data[2:, 1]
        ZN_posi = ZN_data[2:, 1]
        ZE_posi = ZE_data[2:, 1]
        NZ_posi = NZ_data[2:, 1]
        EZ_posi = EZ_data[2:, 1]
    
        EE_posi = EE_data[2:, 1]
        EN_posi = EN_data[2:, 1]
        NE_posi = NE_data[2:, 1]
        NN_posi = NN_data[2:, 1]
    # use half axis data
    elif Flag.Half == -1:
        Source_sta.Lat=ZZ_data[1, 1];Receive_sta.Lon=ZZ_data[1, 0];Receive_sta.Name=sta2
        Receive_sta.Lat=ZZ_data[0, 1];Source_sta.Lon=ZZ_data[0, 0];Source_sta.Name=sta1
        ZZ_posi = ZZ_data[2:, 2]
        ZN_posi = ZN_data[2:, 2]
        ZE_posi = ZE_data[2:, 2]
        NZ_posi = NZ_data[2:, 2]
        EZ_posi = EZ_data[2:, 2]
    
        EE_posi = EE_data[2:, 2]
        EN_posi = EN_data[2:, 2]
        NE_posi = NE_data[2:, 2]
        NN_posi = NN_data[2:, 2]
    # average of positive and negative half-axis 
    elif Flag.Half == 0:
        Source_sta.Lat=ZZ_data[0, 1];Source_sta.Lon=ZZ_data[0, 0];Source_sta.Name=sta1
        Receive_sta.Lat=ZZ_data[1, 1];Receive_sta.Lon=ZZ_data[1, 0];Receive_sta.Name=sta2
        ZZ_posi = (ZZ_data[2:, 1] + ZZ_data[2:, 2]) / 2
        ZN_posi = (ZN_data[2:, 1] + ZN_data[2:, 2]) / 2
        ZE_posi = (ZE_data[2:, 1] + ZE_data[2:, 2]) / 2
        NZ_posi = (NZ_data[2:, 1] + NZ_data[2:, 2]) / 2
        EZ_posi = (EZ_data[2:, 1] + EZ_data[2:, 2]) / 2
        EE_posi = (EE_data[2:, 1] + EE_data[2:, 2]) / 2
        EN_posi = (EN_data[2:, 1] + EN_data[2:, 2]) / 2
        NE_posi = (NE_data[2:, 1] + NE_data[2:, 2]) / 2
        NN_posi = (NN_data[2:, 1] + NN_data[2:, 2]) / 2
    seisData.ZZdata = ZZ_posi
    seisData.ZNdata = ZN_posi
    seisData.ZEdata = ZE_posi
    seisData.NZdata = NZ_posi
    seisData.EZdata = EZ_posi
    seisData.EEdata = EE_posi
    seisData.ENdata = EN_posi
    seisData.NEdata = NE_posi
    seisData.NNdata = NN_posi
    return seisData,Source_sta,Receive_sta,Disper_file

# Determine the period based on the ZZ-dispersion 
def DisperRead(Flag,Para,Disper_file):
    disper_root = Flag.ref_disp
    disper_path = os.path.join(disper_root, Disper_file)

    # Check if the file exists
    if not os.path.exists(disper_path) or Flag.Period_setting == 0:
        print('Use the period data in settings.')
    else:
        # Read dispersion file
        try:
            with open(disper_path, 'r') as fileID:
                # Skip the first two lines
                for _ in range(2):
                    next(fileID)
                # Read the remaining data
                data = np.loadtxt(fileID)
            # Convert to numpy array
            disperdata = np.array(data)
            
            # Find the initial and final indices
            ini_index = np.where(disperdata[:, 3] == 1)[0][0]
            fin_index = np.where(disperdata[:, 3] == 1)[0][-1]
            
            # Set parameters
            Para.Ts = disperdata[ini_index, 0]
            Para.Te = disperdata[fin_index, 0]
            Para.T = np.arange(Para.Ts, Para.Te + Para.dT, Para.dT)  # Assuming Para.dT is defined

        except IOError:
            raise Exception(f'Unable to open file: {disper_path}')

#Traverse all periods to find the best azimuth angle
def Base_SearchT(seisData, Para, Flag, souSta, recSta):
    Angle_T_ZR = np.zeros((len(Para.T), 2))
    Angle_T_RZ = np.zeros((len(Para.T), 2))
    T_show = np.array(Para.T_show)
    for i in range(len(Para.T)):
        Angle_T_ZR[i, 0] = Para.T[i]
        Angle_T_RZ[i, 0] = Para.T[i]
        Angle_corr_ZR_T, Angle_corr_RZ_T, max_angle_T = AngleSearchT(seisData, Para, Flag, Para.T[i])
        if len(T_show) > 0 and np.any(np.isclose(T_show, Para.T[i], atol=1e-8)):
            PlotAngleCorr(Angle_corr_ZR_T, Angle_corr_RZ_T, Para.T[i], max_angle_T, Para, Flag, souSta, recSta)
    
        Angle_T_ZR[i, 1] = max_angle_T[0]
        Angle_T_RZ[i, 1] = max_angle_T[1]
    return Angle_T_ZR, Angle_T_RZ

# The main funtion to calculate the correlation of different azimuth angles for each period
def AngleSearchT(seisData, Para, Flag, T):
    ZZdata = seisData.ZZdata
    ZNdata = seisData.ZNdata
    ZEdata = seisData.ZEdata
    NZdata = seisData.NZdata
    EZdata = seisData.EZdata

    # Convert data from CF to EGF:
    ZZdata = np.imag(hilbert(ZZdata))
    ZNdata = np.imag(hilbert(ZNdata))
    ZEdata = np.imag(hilbert(ZEdata))
    NZdata = np.imag(hilbert(NZdata))
    EZdata = np.imag(hilbert(EZdata))

    Angle = np.arange(Para.Low_bound, Para.High_bound + 1)  # All angles to iterate through
    Angle_len = len(Angle)
    Data_len = len(ZZdata)  # Length of the data

    
    # Apparent velocity window
    Vel_Window, Vlow_index, Vhigh_index = VelocityWindow(Data_len, Para, T)

    # Apply apparent velocity filtering to the ZZ component data:
    ZZdata_win = ZZdata * Vel_Window
    # Calculate group velocity spectrum:
    ZZdata_filter_first = Groupfilter(ZZdata_win, T, Para.Dist, Para.Sample_rate)  # Filtering independent of dT, amplitude spectrum multiplied by Gaussian window in frequency domain
    # Calculate group velocity envelope for the ZZ component:
    ZZ_envelope = np.abs(hilbert(ZZdata_filter_first))
    # Find the maximum point of the group velocity envelope within the apparent velocity window, note this point's position is relative to 'Vhigh_index'
    Max_index = np.argmax(ZZ_envelope[Vhigh_index:Vlow_index])
    Max_index += Vhigh_index 
    # Design cut window, return cut window, bandpass lower and upper limits
    Cut_Window, Cut_min, Cut_max = CutWindow(Data_len, Max_index, T, Para)
    # Apply Hilbert transform to ZE and ZN (shift right by 90 degrees):
    ZEdata_hilbert = np.imag(hilbert(ZEdata))
    ZNdata_hilbert = np.imag(hilbert(ZNdata))
    # Apply Hilbert transform to RZ (shift left by 90 degrees), applying Hilbert transform twice continuously to a signal is equivalent to negating the signal:
    EZdata_hilbert = -np.imag(hilbert(EZdata))
    NZdata_hilbert = -np.imag(hilbert(NZdata))
    # Apply apparent velocity filtering:
    ZEdata_win = ZEdata_hilbert * Vel_Window
    ZNdata_win = ZNdata_hilbert * Vel_Window
    EZdata_win = EZdata_hilbert * Vel_Window
    NZdata_win = NZdata_hilbert * Vel_Window

    ZZdata_filter = Phasefilter(ZZdata_win, T, Max_index, Para)
    ZEdata_filter = Phasefilter(ZEdata_win, T, Max_index, Para)
    ZNdata_filter = Phasefilter(ZNdata_win, T, Max_index, Para)
    EZdata_filter = Phasefilter(EZdata_win, T, Max_index, Para)
    NZdata_filter = Phasefilter(NZdata_win, T, Max_index, Para)

    # First column is correction angle, second column is corresponding correlation coefficient
    Angle_corr_ZR = np.zeros((Angle_len, 2))
    Angle_corr_RZ = np.zeros((Angle_len, 2))

    # Iterate through angles:
    for i in range(Angle_len):
        Angle_corr_ZR[i, 0] = Angle[i]
        Angle_corr_RZ[i, 0] = Angle[i]

        # Rotate data from ZN and ZE components to obtain ZR component data
        ZRdata_filter = ZEdata_filter * (-np.sin(np.deg2rad(Para.psi + Angle[i]))) + ZNdata_filter * (-np.cos(np.deg2rad(Para.psi + Angle[i])))
        # Rotate data from NZ and EZ components to obtain RZ component data:
        RZdata_filter = EZdata_filter * np.sin(np.deg2rad(Para.thet + Angle[i])) + NZdata_filter * np.cos(np.deg2rad(Para.thet + Angle[i]))

        # Perform cross-correlation on the filtered results (phase velocity):
        X = ZZdata_filter[Cut_min:Cut_max]
        Y_ZR = ZRdata_filter[Cut_min:Cut_max]
        Y_RZ = RZdata_filter[Cut_min:Cut_max]

        if Flag.Corr_Method == 1:
            Angle_corr_ZR[i, 1] = np.sum(X * Y_ZR) / (np.sqrt(np.sum(X**2)) * np.sqrt(np.sum(Y_ZR**2)))  # X*Y / (|X| * |Y|) -- p(ZR, ZZ)/sqrt(p(ZR,ZR)) * sqrt(p(ZZ,ZZ)) -- p(): zero-delay cross-correlation
            Angle_corr_RZ[i, 1] = np.sum(X * Y_RZ) / (np.sqrt(np.sum(X**2)) * np.sqrt(np.sum(Y_RZ**2)))
        else:
            Angle_corr_ZR[i, 1] = np.sum(X * Y_ZR) / np.sum(X**2)  # X*Y / |X|^2 -- p(ZR, ZZ)/p(ZZ,ZZ) -- The denominator normalization factor is a constant because the ZZ component does not change
            Angle_corr_RZ[i, 1] = np.sum(X * Y_RZ) / np.sum(X**2)
    
    #Record the azimuth with the largest cross-correlation,ZR in maxangle[0],RZ in maxangle[1]
    maxangle=np.zeros(2)
    ZR_angle=np.argmax(Angle_corr_ZR[:, 1])
    RZ_angle=np.argmax(Angle_corr_RZ[:, 1])
    maxangle[0]=Angle_corr_ZR[ZR_angle, 0] 
    maxangle[1]=Angle_corr_RZ[RZ_angle, 0]
    
    return Angle_corr_ZR, Angle_corr_RZ, maxangle

# The main funtion to calculate the correlation of different azimuth angles in the full frequency band
def AngleSearch(seisData, Para, Flag):
    ZZdata = seisData.ZZdata
    ZNdata = seisData.ZNdata
    ZEdata = seisData.ZEdata
    NZdata = seisData.NZdata
    EZdata = seisData.EZdata
    
    # Convert data from CF to EGF:
    ZZdata = np.imag(hilbert(ZZdata))
    ZNdata = np.imag(hilbert(ZNdata))
    ZEdata = np.imag(hilbert(ZEdata))
    NZdata = np.imag(hilbert(NZdata))
    EZdata = np.imag(hilbert(EZdata))
    
    Angle = np.arange(Para.Low_bound, Para.High_bound + 1)  # All angles to be searched
    Angle_len = len(Angle)
    Data_len = len(ZZdata)  # Length of the data
    
    T_min = 2 / Para.Sample_rate  # Minimum period (Nyquist sampling theorem)
    filter_min = max(Para.Ts, T_min)
    filter_max = Para.Te
    
    # Velocity window, coordinates corresponding to velocity thresholds
    Vel_Window, Vlow_index, Vhigh_index = VelocityWindow(Data_len, Para, Para.Te)
    
    # Velocity filtering of ZZ component data
    ZZdata_win = ZZdata * Vel_Window
    # Calculate amplitude envelope using the entire frequency band signal
    ZZ_envelope = np.abs(hilbert(ZZdata_win))
    # Find the maximum point of the group velocity envelope within the velocity window
    Max_index = np.argmax(ZZ_envelope[Vhigh_index:Vlow_index])
    Max_index += Vhigh_index 
    
    # Design and return the cut window, bandpass upper and lower limits
    Cut_Window, Cut_min, Cut_max = CutWindow(Data_len, Max_index, Para.Te, Para)
    
    # Hilbert transform of ZE and ZN (shifted 90 degrees to the right)
    ZEdata_hilbert = np.imag(hilbert(ZEdata))
    ZNdata_hilbert = np.imag(hilbert(ZNdata))
    # Hilbert transform of EZ (shifted 90 degrees to the left), negative due to two successive Hilbert transforms
    EZdata_hilbert = -np.imag(hilbert(EZdata))
    NZdata_hilbert = -np.imag(hilbert(NZdata))
    
    # Velocity filtering
    ZEdata_win = ZEdata_hilbert * Vel_Window
    ZNdata_win = ZNdata_hilbert * Vel_Window
    EZdata_win = EZdata_hilbert * Vel_Window
    NZdata_win = NZdata_hilbert * Vel_Window
    
    # Data window
    ZZdata_cut = ZZdata_win * Cut_Window
    ZEdata_cut = ZEdata_win * Cut_Window
    ZNdata_cut = ZNdata_win * Cut_Window
    EZdata_cut = EZdata_win * Cut_Window
    NZdata_cut = NZdata_win * Cut_Window
    
    # Bandpass filtering
    ZZdata_filter = Bandpass(ZZdata_cut, filter_min, filter_max, Para.Sample_rate)
    ZEdata_filter = Bandpass(ZEdata_cut, filter_min, filter_max, Para.Sample_rate)
    ZNdata_filter = Bandpass(ZNdata_cut, filter_min, filter_max, Para.Sample_rate)
    EZdata_filter = Bandpass(EZdata_cut, filter_min, filter_max, Para.Sample_rate)
    NZdata_filter = Bandpass(NZdata_cut, filter_min, filter_max, Para.Sample_rate)
    
    # Initialize arrays for angle correction and cross-correlation coefficients
    Angle_corr_ZR = np.zeros((Angle_len, 2))
    Angle_corr_RZ = np.zeros((Angle_len, 2))
    
    temp_corr_ZR = np.zeros(Angle_len)
    temp_corr_RZ = np.zeros(Angle_len)
    
    corr_ZR = np.zeros(Angle_len) # non-normalized
    corr_RZ = np.zeros(Angle_len)
    
    nor_corr_ZR = np.zeros(Angle_len) # normalized
    nor_corr_RZ = np.zeros(Angle_len)
    
    # Traverse all angles 
    for i in range(Angle_len):
        Angle_corr_ZR[i, 0] = Angle[i]
        Angle_corr_RZ[i, 0] = Angle[i]
        
        # Rotate ZN and ZE component data to obtain ZR component data
        ZRdata_filter = ZEdata_filter * (-np.sin(np.deg2rad(Para.psi + Angle[i]))) + ZNdata_filter * (-np.cos(np.deg2rad(Para.psi + Angle[i])))
        # Rotate NZ and EZ component data to obtain RZ component data
        RZdata_filter = EZdata_filter * np.sin(np.deg2rad(Para.thet + Angle[i])) + NZdata_filter * np.cos(np.deg2rad(Para.thet + Angle[i]))
        
        # Cross-correlation of filtered results (phase velocity)
        X = ZZdata_filter[Cut_min:Cut_max + 1]
        Y_ZR = ZRdata_filter[Cut_min:Cut_max + 1]
        Y_RZ = RZdata_filter[Cut_min:Cut_max + 1]
        
        nor_corr_ZR[i] = np.sum(X * Y_ZR) / (np.sqrt(np.sum(X ** 2)) * np.sqrt(np.sum(Y_ZR ** 2)))
        nor_corr_RZ[i] = np.sum(X * Y_RZ) / (np.sqrt(np.sum(X ** 2)) * np.sqrt(np.sum(Y_RZ ** 2)))
        corr_ZR[i] = np.sum(X * Y_ZR) / np.sum(X ** 2)
        corr_RZ[i] = np.sum(X * Y_RZ) / np.sum(X ** 2)
        
        if Flag.Corr_Method == 1:
            Angle_corr_ZR[i, 1] = nor_corr_ZR[i]
            Angle_corr_RZ[i, 1] = nor_corr_RZ[i]
        else:
            Angle_corr_ZR[i, 1] = corr_ZR[i]
            Angle_corr_RZ[i, 1] = corr_RZ[i]
    
    #Record the azimuth with the largest cross-correlation,ZR in maxangle[0],RZ in maxangle[1]
    maxangle=np.zeros(2)
    ZR_angle=np.argmax(corr_ZR)
    RZ_angle=np.argmax(corr_RZ)
    maxangle[0]=Angle_corr_ZR[ZR_angle, 0] 
    maxangle[1]=Angle_corr_RZ[RZ_angle, 0]
    
    # Calculate maximum correlation coefficient(method I)
    #indexZR = np.argmax(temp_corr_ZR)
    #indexRZ = np.argmax(temp_corr_RZ)
    #corrZR = temp_corr_ZR[indexZR]
    #corrRZ = temp_corr_RZ[indexRZ]
    #---------
    #print(corrZR,corrRZ)
    
    # Another way to define the normalized correlation coefficient,correspond to the best angle
    corrZR = nor_corr_ZR[ZR_angle]
    corrRZ = nor_corr_RZ[RZ_angle]
    #---------
    Corr_I = max(corrRZ, corrZR)
    #Record which is greater,corrZR or RZ
    if corrZR>corrRZ:
        index_angle=0
    else:
        index_angle=1
    
    return Angle_corr_ZR, Angle_corr_RZ, Corr_I, index_angle, maxangle

# Remove the angle ​​that deviate greatly from the mean angle 
def ModifyAngle(Angle):
    T = Angle[:, 0]  # Period 
    T_len = len(T)
    Angle_raw = Angle[:, 1]  # Original corrected angle values
    #parameters can be modified
    threshold1 = 45  # First threshold
    threshold2 = 30 # Second threshold
    Ratio = 0.4    # Proportion of points to be removed, if less than this proportion, return the modified data, otherwise return the original data.

    Angle_raw_mean = np.mean(Angle_raw)

    # First screening
    Angle_first = np.copy(Angle_raw)
    Angle_first[np.abs(Angle_raw - Angle_raw_mean) > threshold1] = np.nan
    Angle_first_mean = np.nanmean(Angle_first)
    
    # Second screening
    Angle_second = np.copy(Angle_first)
    Angle_second[np.abs(Angle_first - Angle_first_mean) > threshold2] = np.nan
    angle_mean = np.nanmean(Angle_second) 
    
   
    # Number of removed points and their indices
    index_modify = np.isnan(Angle_second)
    index_save = ~index_modify
    modify_num = np.sum(index_modify)
    modify_Ratio = modify_num / T_len
    
    angle_modified = Angle
    # If the proportion of removed points exceeds the threshold
    if modify_Ratio > Ratio:
        angle_modified[:,1] = Angle_raw
        angle_mean = 0
        print(f'Modify rate: {modify_Ratio}')
    else:
        # Use 'cubic' interpolation to obtain the values of the removed points
        f = interp1d(T[index_save], Angle_raw[index_save], kind='cubic', fill_value=angle_mean, bounds_error=False)
        angle_modified[:,1] = np.copy(Angle_raw)
        angle_modified[index_modify,1] = f(T[index_modify])

    return angle_modified, angle_mean,modify_Ratio

#plot ZZ/RR and ZR/RZ component
def PlotSeis(t_axis, ZZdata, ZRdata, RZdata, RRdata, Para, Flag, souSta, recSta):
    # Perform Hilbert transform on ZR (shifted by 90 degrees):
    ZRdataH = np.imag(hilbert(ZRdata))
    # Perform Hilbert transform on RZ (shifted by -90 degrees):
    RZdataH = -np.imag(hilbert(RZdata))
    # Compute velocity window based on parameters
    Vel_Window, _, _ = VelocityWindow(len(ZZdata), Para, Para.Te)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # First subplot
    ax1.plot(t_axis, ZZdata / np.max(np.abs(ZZdata)), 'k', label='ZZ')
    ax1.plot(t_axis, RRdata / np.max(np.abs(RRdata)), 'r', label='RR')
    ax1.plot(t_axis, Vel_Window, 'g', label='window')  # Velocity window
    ax1.set_xlabel('t(s)', fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax1.set_ylabel('Normalized Amplitude', fontsize=16, fontweight='bold', fontname='Times New Roman')
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_title(f'ZZ/RR components (Dist = {Para.Dist:.2f} Km)')
    ax1.set_xlim(0, 100)
    #ax1.set_ylim(-1, 1.2)
    ax1.legend(fontsize=15,loc='upper right')
    
    
    # Second subplot
    ax2.plot(t_axis, ZRdataH / np.max(np.abs(ZRdataH)), 'b', label='ZR')
    ax2.plot(t_axis, RZdataH / np.max(np.abs(RZdataH)), 'r', label='RZ')
    ax2.plot(t_axis, Vel_Window, 'g', label='window')  # Velocity window
    ax2.set_xlabel('t/s')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('ZR*exp(-90);RZ*exp(-90)')
    ax2.set_xlim(0, 100)
    #ax2.set_ylim(-1, 1.2)
    ax2.legend()

    plt.tight_layout()
    
    if Flag.Save_png == 1:
        savePath = os.path.join(Flag.run_path, 'single')
        os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, f'CFs_{souSta}_{recSta}.png'))
    plt.show(block=False)

# plot group velocity spectrum
def PlotSpectrum(Seisdata, Para, souSta, recSta, component, Flag):
    T_len = len(Para.T)
    Data_len = len(Seisdata)
    
    env = np.zeros((Data_len, T_len))
    # Calculate the group velocity spectrum
    for i in range(T_len):
        Vel_Window, Vlow_index, Vhigh_index = VelocityWindow(Data_len, Para, Para.T[i])
        dataWin = Seisdata * Vel_Window
        dataGroup = Groupfilter(dataWin, Para.T[i], Para.Dist, Para.Sample_rate)
        env[:, i] = np.abs(hilbert(dataGroup))
    timeptnum = np.arange(Vhigh_index, Vlow_index + 1)
    time = np.arange(Vhigh_index+1, Vlow_index + 2) / Para.Sample_rate
    
    VPoint = np.arange(Para.Vlow, Para.Vhigh + Para.dv, Para.dv)
    VPpoint_len = len(VPoint)
    
    TravPtV = Para.Dist / time
    GroupVImg = np.zeros((VPpoint_len, T_len))

    # Interpolate waveform data and normalize amplitude
    V_max = np.zeros(T_len)
    for i in range(T_len):
        interp_func = interp1d(TravPtV, env[timeptnum, i] / np.max(env[timeptnum, i]), kind='cubic',fill_value="extrapolate")
        GroupVImg[:, i] = interp_func(VPoint)
        Max_index = np.argmax(GroupVImg[:, i])
        V_max[i] = VPoint[Max_index]
    
    minamp = np.min(GroupVImg)

    # Plotting
    fig, ax = plt.subplots()
    cmap = plt.cm.jet
    cax = ax.imshow(GroupVImg, extent=[Para.T[0], Para.T[-1], VPoint[0], VPoint[-1]], cmap=cmap,aspect='auto', vmin=minamp, vmax=1, origin='lower')
    #np.savetxt('img2.txt',GroupVImg) #test modify
    # Load custom colormap
    # Uncomment and customize the following lines if you have a specific colormap to load
    # import matplotlib.colors as mcolors
    # cmap = mcolors.ListedColormap([...])  # Define your colormap here
    # plt.colormap(cmap)  # Uncomment this line if cmap is defined
    
    ax.plot(Para.T, V_max, 'g-*')
    ax.set_xlabel('Period(s)')
    ax.set_ylabel('Group Vel(Km/s)')
    ax.set_title(f'{component}-Group-Spectrum({souSta}-{recSta})')
    
    fig.colorbar(cax)
    
    if Flag.Save_png == 1:
        savePath = os.path.join(Flag.run_path, 'single')
        os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, f'spectrum_{component}_{souSta}_{recSta}.png'))
    plt.show(block=False)

# plot Corr versus azimuth 
def PlotAngleCorr(Angle_corr_ZR, Angle_corr_RZ, T, maxangle, Para, Flag, souSta, recSta):
    angleZR = maxangle[0]
    angleRZ = maxangle[1]
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    corMax = max(Angle_corr_ZR[:, 1])
    corMax_plot = max(1, corMax)
    
    ax[0].plot(Angle_corr_ZR[:, 0], Angle_corr_ZR[:, 1], 'b')
    ax[0].set_xlabel('Angle')
    ax[0].set_ylabel('Corr')
    if T:
        ax[0].set_title(f'ZR(T = {T:.1f}s), corrMax = {corMax:.2f}')
    else:
        ax[0].set_title(f'ZR(T = {Para.Ts}-{Para.Te}s), corrMax = {corMax:.2f}')
    
    ylim = ax[0].get_ylim()
    ax[0].plot([angleZR, angleZR], ylim, 'g--',label=f'Max corr angle={angleZR}°')
    ax[0].plot(Angle_corr_ZR[:, 0], np.zeros(len(Angle_corr_ZR[:, 0])), 'm--')
    ax[0].plot(Angle_corr_ZR[:, 0], np.ones(len(Angle_corr_ZR[:, 0])) * 0.9, 'r--',label='corr=0.9')
    ax[0].axis([-200, 200, -corMax_plot*1.1, corMax_plot*1.1])
    ax[0].legend(loc='upper right')
    
    corMax = max(Angle_corr_RZ[:, 1])
    corMax_plot = max(1, corMax)
    ax[1].plot(Angle_corr_RZ[:, 0], Angle_corr_RZ[:, 1], 'b')
    ax[1].set_xlabel('Angle')
    ax[1].set_ylabel('Corr')
    ax[1].set_title(f'RZ, corrMax = {corMax:.2f}')
    
    ylim = ax[1].get_ylim()
    ax[1].plot([angleRZ, angleRZ], ylim, 'g--',label=f'Max corr angle={angleRZ}°')
    ax[1].plot(Angle_corr_RZ[:, 0], np.zeros(len(Angle_corr_RZ[:, 0])), 'm--')
    ax[1].plot(Angle_corr_RZ[:, 0], np.ones(len(Angle_corr_RZ[:, 0])) * 0.9, 'r--',label='corr=0.9')
    ax[1].axis([-200, 200, -corMax_plot*1.1, corMax_plot*1.1])
    ax[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    if Flag.Save_png == 1:
        savePath = os.path.join(Flag.run_path, 'single')
        os.makedirs(savePath, exist_ok=True)
        if T:
            plt.savefig(os.path.join(savePath, f'angle_correlation_T{T:.1f}_{souSta}_{recSta}.png'))
        else:
            plt.savefig(os.path.join(savePath, f'angle_Corr_fullband_{souSta}_{recSta}.png'))
    plt.show(block=False)

# plot the best azimuth angle versus period
def PlotTAngle(Angle_T_ZR, Angle_T_RZ, angleZR, angleRZ, souSta, recSta, Flag):
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    ax[0].plot(Angle_T_ZR[:, 0], Angle_T_ZR[:, 1], 'b-o')
    ax[0].plot(Angle_T_ZR[:, 0], np.full_like(Angle_T_ZR[:, 0], angleZR), 'r--', label=f'Full band correct angle={angleZR}°')
    ax[0].legend()
    ax[0].set_xlabel('T/s')
    ax[0].set_ylabel('Angle')
    ax[0].set_ylim(-180, 180)
    ax[0].set_title(f'ZR-{souSta}-{recSta}')

    ax[1].plot(Angle_T_RZ[:, 0], Angle_T_RZ[:, 1], 'b-o')
    ax[1].plot(Angle_T_RZ[:, 0], np.full_like(Angle_T_RZ[:, 0], angleRZ), 'r--', label=f'Full band correct angle={angleRZ}°')
    ax[1].legend()
    ax[1].set_xlabel('T/s')
    ax[1].set_ylabel('Angle')
    ax[1].set_ylim(-180, 180)
    ax[1].set_title('RZ')

    plt.tight_layout()
    
    if Flag.Save_png == 1:
        savePath = os.path.join(Flag.run_path, 'single')
        os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, f'period_angle_{souSta}_{recSta}.png'))
    plt.show(block=False)

# plot RR/TT component before and after modify
def PlotContrast(t_axis, RRdata, RRmodify, TTdata, TTmodify, Para, Flag, souSta, recSta):
    # Normalize RRdata and RRmodify with the same factor
    RRamp = max(np.max(np.abs(RRdata)), np.max(np.abs(RRmodify)))
    RRdata = RRdata / RRamp
    RRmodify = RRmodify / RRamp

    # Normalize TTdata and TTmodify with the same factor
    TTamp = max(np.max(np.abs(TTdata)), np.max(np.abs(TTmodify)))
    TTdata = TTdata / TTamp
    TTmodify = TTmodify / TTamp

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot RR components before and after correction
    ax1.plot(t_axis, RRdata, 'b', label='RR')
    ax1.plot(t_axis, RRmodify, 'r', label='RR-modify')
    ax1.set_xlabel('t/s')
    ax1.set_ylabel('Amplitude')
    ax1.axis([0, 60, -1.2, 1.2])
    ylim = ax1.get_ylim()
    ax1.plot([Para.Dist / Para.Vlow, Para.Dist / Para.Vlow], ylim, 'k--')
    ax1.plot([Para.Dist / Para.Vhigh, Para.Dist / Para.Vhigh], ylim, 'k--')
    ax1.legend(['RR', 'RR-modify', f'Window:[{Para.Vlow:.1f} {Para.Vhigh:.1f}]km/s'],loc='upper right')
    ax1.set_title('RR')

    # Plot TT components before and after correction
    ax2.plot(t_axis, TTdata, 'b', label='TT')
    ax2.plot(t_axis, TTmodify, 'r', label='TT-modify')
    ax2.set_xlabel('t/s')
    ax2.set_ylabel('Amplitude')
    ax2.axis([0, 60, -1.2, 1.2])
    ylim = ax2.get_ylim()
    ax2.plot([Para.Dist / Para.Vlow, Para.Dist / Para.Vlow], ylim, 'k--')
    ax2.plot([Para.Dist / Para.Vhigh, Para.Dist / Para.Vhigh], ylim, 'k--')
    ax2.legend(['TT', 'TT-modify', f'Window:[{Para.Vlow:.1f} {Para.Vhigh:.1f}]km/s'],loc='upper right')
    ax2.set_title('TT')

    plt.tight_layout()
    
    if Flag.Save_png == 1:
        savePath = os.path.join(Flag.run_path, 'single')
        os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, f'CFs_contrast_{souSta}_{recSta}.png'))
    plt.show(block=False)

#MFT Gaussian filter
def Groupfilter(data_raw, T, Dist, Sample_rate):
    # data_raw: data to be filtered
    # T: center period
    # Dist: station distance

    alfa = np.array([[0, 100, 250, 500, 1000, 2000, 4000, 20000],
                     [5, 8, 12, 20, 25, 35, 50, 75]])
    guassalfa = interp1d(alfa[0], alfa[1])(Dist)  # Determine the Gaussian filter parameter based on the distance
    data_len = len(data_raw)  # Length of the waveform

    nfft = int(2 ** math.ceil(np.log2(max(data_len, 1024 * Sample_rate))))  # Ensure the FFT length is sufficiently long
    xxfft = fft(data_raw, nfft)  # Perform the FFT
    fxx = np.arange(0, nfft // 2 + 1) / nfft * Sample_rate  # Frequency domain coordinates (first half)
    IIf = np.arange(nfft // 2 + 1)  # Indices for the first half
    JJf = np.arange(nfft // 2 + 1, nfft)  # Indices for the second half

    fc = 1 / T  # Center frequency
    Hf = np.exp(-guassalfa * (fxx - fc) ** 2 / fc ** 2)
    yyfft = np.zeros(nfft)  
    yyfft[IIf] = xxfft[IIf] * Hf  # Apply the Gaussian window
    yyfft[JJf] = np.conj(yyfft[(nfft // 2)-1:0:-1])
    yy = np.real(ifft(yyfft, nfft))  # Perform the inverse FFT
    data_filter = yy[:data_len]  # Filtered result
    return data_filter

#Band pass filter around center period
def Phasefilter(data_raw, T, Max_index, Para):
    # data_raw: Data after "apparent velocity filtering"
    # Max_index: Index corresponding to the peak of the group velocity envelope

    data_len = len(data_raw)  # Length of the data
    filter_len = 2**np.ceil(np.log2(1024 * Para.Sample_rate)).astype(int)  # Length of the filter
    KaiserPara = 6  
    HalfFilterNum = round(filter_len / 2)  # Length of half the filter
    data_raw = np.pad(data_raw, (0, HalfFilterNum), 'constant')  # Append zeros to the end of the waveform
    
    # Band-pass filtering:
    F = (2 / Para.Sample_rate) / T  # Center frequency
    LowF = (2 / Para.Sample_rate) / (T + 0.5 * Para.dT)  # Lower bound of the filter frequency
    HighF = (2 / Para.Sample_rate) / (T - 0.5 * Para.dT)  # Upper bound of the filter frequency
    Filter = firwin(filter_len , [LowF, HighF], window=('kaiser', KaiserPara), pass_zero=False)
    
    # Two-pass filtering (time and time-reverse) to remove phase shift
    winpt = round(Para.T_num * T * Para.Sample_rate)  # Length of the band-pass in terms of points
    if winpt % 2 == 1:  # Ensure winpt is an even number
        winpt += 1  
    wintukey = windows.tukey(winpt, 0.2)  # Tukey (tapered cosine) window
    grouppt = winpt + Max_index  # Point number of band-pass length plus point number of group velocity peak
    tmpWave = np.concatenate([np.zeros(winpt), data_raw[:data_len], np.zeros(winpt)])
    tmpWave[grouppt - winpt // 2:grouppt + winpt // 2] *= wintukey  # Filter here - centered at group velocity peak point with window length T*Tnum
    tmpWave[:grouppt - winpt // 2] = 0
    tmpWave[grouppt + winpt // 2 - 1:] = 0

    NewWinWave = np.zeros(data_len + HalfFilterNum)
    NewWinWave[:data_len] = tmpWave[winpt:winpt + data_len]  # Data after windowing
    FilteredWave = fftconvolve(NewWinWave[:data_len + HalfFilterNum],Filter, mode='same' )  # First pass of filtering
    FilteredWave = FilteredWave[::-1]  # Reverse the result of the first filtering
    FilteredWave = fftconvolve(FilteredWave[:data_len + HalfFilterNum],Filter, mode='same' )  # Second pass of filtering
    FilteredWave = FilteredWave[::-1]

    data_bandfilter = FilteredWave[:data_len]  # Final filtered result
    
    return data_bandfilter

#the signal window,calculate the window time through velocity boundary.
def VelocityWindow(Data_len, Para, T):
    Vel_Window = np.zeros(Data_len)  # Velocity window initialization

    # Calculate indices based on parameters
    Vlow_index = min(round((Para.Dist / Para.Vlow) / Para.dt)-1, Data_len - 1)
    Vhigh_index = round((Para.Dist / Para.Vhigh) / Para.dt)-1

    # Bandpass window
    Vel_Window[int(Vhigh_index):int(Vlow_index)+1] = 1
    
    # Sine window ramps
    Slope_len_before = min(round(T / 2 / Para.dt), int(Vhigh_index))
    Vel_Window[int(Vhigh_index) - Slope_len_before:int(Vhigh_index)] = np.sin(0.5 * np.pi * np.arange(Slope_len_before) / Slope_len_before)

    Slope_len_after = min(round(T / 2 / Para.dt), int(Data_len - Vlow_index)-1)
    Vel_Window[int(Vlow_index)+1 :int(Vlow_index) + Slope_len_after + 1] = np.sin(0.5 * np.pi * (np.arange(Slope_len_after, 0, -1)) / Slope_len_after)
    
    
    return Vel_Window, Vlow_index, Vhigh_index

#design a signal window,just like temporal variable filtering in phase velocity analysis
def CutWindow(Data_len, Max_index, T, Para):
    # Indices corresponding to the velocity window
    Vlow_index = min(round((Para.Dist / Para.Vlow) / Para.dt)-1, Data_len - 1)
    Vhigh_index = round((Para.Dist / Para.Vhigh) / Para.dt)-1
    
    Cut_Window = np.zeros(Data_len)  # Cut window
    Cut_min = max(1, Max_index - round((Para.T_num / 2) * T / Para.dt))-1  # Prevent window out of bounds
    Cut_max = min(Data_len, Max_index + round((Para.T_num / 2) * T / Para.dt))-1
    
    # The time range of the 'time-varying window' should not exceed the 'velocity window':
    Cut_min = max(Cut_min, Vhigh_index)
    Cut_max = min(Cut_max, Vlow_index)
    
    # Window 2: tukeywin(Len, 0.2), with 10% cosine ramp on both sides
    Cut_Window[Cut_min : Cut_max + 1] = windows.tukey(Cut_max - Cut_min + 1, 0.2)
    
    return Cut_Window, Cut_min, Cut_max

def Bandpass(data_raw, filter_min, filter_max, Sample_rate):
    # Design Butterworth bandpass filter
    nyquist_freq = 0.5 * Sample_rate
    lowcut = (1 / filter_max) / nyquist_freq
    highcut = (1 / filter_min) / nyquist_freq
    b, a = butter(2, [lowcut, highcut], btype='band')  # 2nd order Butterworth filter
    
    # Apply zero-phase filtering using filtfilt
    data_filter = filtfilt(b, a, data_raw)
    
    return data_filter   

# Rotate to obtain the T/R component data.  
def CompRotate(Seisdata, thet, psi):
    # Define sin and cos functions that operate in degrees
    sind = lambda x: np.sin(np.deg2rad(x))
    cosd = lambda x: np.cos(np.deg2rad(x))
    
    ZR = Seisdata.ZEdata * (-sind(psi)) + Seisdata.ZNdata * (-cosd(psi))
    RZ = Seisdata.EZdata * sind(thet) + Seisdata.NZdata * cosd(thet)
    RR = (Seisdata.EEdata * (-sind(thet) * sind(psi)) + 
          Seisdata.ENdata * (-sind(thet) * cosd(psi)) +
          Seisdata.NEdata * (-cosd(thet) * sind(psi)) + 
          Seisdata.NNdata * (-cosd(thet) * cosd(psi)))
    TT = (Seisdata.EEdata * (-cosd(thet) * cosd(psi)) + 
          Seisdata.ENdata * (cosd(thet) * sind(psi)) +
          Seisdata.NEdata * (sind(thet) * cosd(psi)) + 
          Seisdata.NNdata * (-sind(thet) * sind(psi)))
    
    return ZR, RZ, RR, TT

# Calculate the distance and azimuth angle between a pair of staions   
def distaz(sta, sto, epa, epo):
    # dk is the distance between two stations in kilometers
    # dd is the distance in degrees
    # daze is the back azimuth angle in 360 degrees
    # dazs is the azimuth angle in 360 degrees

    rad = math.pi / 180.0

    sa  = math.atan(0.993270 * math.tan(sta * rad))  # Calculate station latitude angle
    ea  = math.atan(0.993270 * math.tan(epa * rad))  # Calculate event latitude angle
    ssa = math.sin(sa)
    csa = math.cos(sa)
    so  = sto * rad  # Convert station longitude to radians
    eo  = epo * rad  # Convert event longitude to radians
    sea = math.sin(ea)
    cea = math.cos(ea)
    ces = math.cos(eo - so)
    ses = math.sin(eo - so)

    # Handle special cases where the coordinates are the same
    if sa == ea and sto == epo:
        return 0.0, 0.0, 0.0, 0.0

    if sta == 90.0 and epa == 90.0:
        return 0.0, 0.0, 0.0, 0.0

    if sta == -90.0 and epa == -90.0:
        return 0.0, 0.0, 0.0, 0.0

    # Calculate distance in degrees
    dd = ssa * sea + csa * cea * ces
    if dd != 0.0:
        dd = math.atan(math.sqrt(1.0 - dd * dd) / dd)
    if dd == 0.0:
        dd = math.pi / 2.0
    if dd < 0.0:
        dd = dd + math.pi
    dd = dd / rad
    dk = dd * 111.19  # Convert distance to kilometers

    # Calculate azimuth and back azimuth angles
    dazs = math.atan2(-ses, (ssa / csa * cea - sea * ces))
    daze = math.atan2(ses, (sea * csa / cea - ssa * ces))
    dazs = dazs / rad
    daze = daze / rad
    if dazs < 0.0:
        dazs = dazs + 360.0
    if daze < 0.0:
        daze = daze + 360.0

    return dk, dd, daze, dazs

# Process a patch of stations,this part is not fine,refer to the test of Auto mode
def AutoProcess(Para,Flag):
    # parameters need modify:folderPath,dispPath,flag,stafile
    folderPath = "C:/Users/ycpan/Auser/Seis/Codes/Dispersion analysis/CF_weifang/CF/CFZ_T_R/Z-Z"
    dispPath = "D:/data/Weifang_dispersion/disp_wf_love_orient_new_0.9"

    # List files matching the patterns
    fileList = glob.glob(os.path.join(folderPath, 'ZZ*.dat'))
    dispList = glob.glob(os.path.join(dispPath, 'CD*'))  

    # Load flag data
    flag = np.loadtxt("flag_love.txt")  # Determine whether the station pair has dispersion data,you can choose to plot it separately. 

    # Read station file
    stafile = './sta_all.txt'
    with open(stafile, "r") as file:
        staname = []
        for line in file:
            fields = line.split()
            if len(fields) == 4:
                staname.append(str(fields[0]))

    # Loop through the stations you selected
    for i in range(100,101):
        print(f"Processing  station {i}")
        sta = staname[i]
        selectedFiles = []
        dispFiles = [] 
        
        #select all the ZZ component seis files contain the i-th station
        for file in fileList:
            fileName = os.path.basename(file)
            parts = re.split('[_-]', fileName)
            sta1 = parts[1]
            sta2 = parts[2]
            if sta == sta1 or sta == sta2:
                selectedFiles.append(fileName)
        #select all the dispersion files contain the i-th station
        for disp in dispList:
            dispName = os.path.basename(disp)
            parts = re.split('[_-]', dispName)
            sta1 = parts[1]
            sta2 = parts[2]
            if sta == sta1 or sta == sta2:
                dispFiles.append(dispName)
                
        n = len(selectedFiles)
        angleZR = np.zeros(n)
        angleRZ = np.zeros(n)
        ZR_angle_mean = np.zeros(n)
        RZ_angle_mean = np.zeros(n)
        Disper_true = []
        Disper_false = []
        Corr_all= []
        LagTime = []
        SNRindex = np.zeros(n)
        Corrindex = np.zeros(n)
        
        for j in range(n):
            delete_index = 0  # If the quality is below the control criteria, set index to 1

            # read data
            Flag.File = selectedFiles[j]
            SeisData,Source_sta,Receive_sta,Disper_file=DataRead(Flag)         
            # calculate distance and azimuth angle between a pair of stations
            Para.Dist, _, Para.psi, Para.thet = distaz(Receive_sta.Lat, Receive_sta.Lon, Source_sta.Lat, Source_sta.Lon)
            if Para.Dist < 1:
                continue
            # Output station info
            print(f'Station pairs: {Source_sta.Name} - {Receive_sta.Name}')
            print(f'Dist is: {Para.Dist}')
            print(f'psi is: {Para.psi}')
            print(f'thet is: {Para.thet}')
            #Rotate to obtain the T/R component data.
            SeisData.ZRdata, SeisData.RZdata, SeisData.RRdata, SeisData.TTdata= CompRotate(SeisData, Para.thet, Para.psi)
            
            #calculate SNR of the full band
            SNR = CalculateSNR(SeisData.ZRdata, Para)
            if SNR > Para.minSNR:
                SNRindex[j] = 1
            else:
                delete_index = 1
            
            Angle_corr_ZR, Angle_corr_RZ,Corr_I,angle_Fullband=AngleSearch(SeisData, Para, Flag)
            lagtime=CalculateLagTime(SeisData, Para, Flag)
            LagTime.append(lagtime)
            #Use Corr as quality control
            if Corr_I > Para.minCorr:
                Corrindex[j] = 1
            else:
                delete_index = 1
            Corr_all.append(Corr_I)
            
            ''' 
            #Statistic station pair separately based on whether it has dispersion data
            sta1_index = staname.index(Source_sta.Name)
            sta2_index = staname.index(Receive_sta.Name)
            if flag[sta1_index, sta2_index] == 1:
                Disper_true.append(angle_Fullband[0])
                Corr.append(Corr_I)
                print(Corr_I)
            if flag[sta1_index, sta2_index] == 0:
                Disper_false.append(angle_Fullband[0])
                delete_index = 0
            '''
            Disper_true.append(angle_Fullband[0])

            
            '''  
            #delete the file with bad quality
            if delete_index == 1:
                parts = re.split('[_-]', Flag.File)
                sta1 = parts[1]
                sta2 = parts[2]
                deletename = f'{sta1}-{sta2}'
                for disp in dispList:
                    if deletename in disp:
                        os.remove(disp)
                        print(f'delete file:{disp}')
            '''
            
            #Disper_true.append(angle_Fullband[0])
            print(f'Full band correction angle for ZR: {angle_Fullband[0]}°')
            print(f'Full band correction angle for RZ: {angle_Fullband[1]}°')
            
        #plot statistic histogram
        edges = np.linspace(-180, 180, 80)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.hist(Disper_true, bins=edges, alpha=0.5, label='Counts disp')
        #ax1.hist(Disper_false, bins=edges, alpha=0.5, label='Counts no disp')

        ax1.set_xlabel('Correction angle(°)')
        ax1.set_title(f'Station-{sta}')

        # compute mean and standard deviation
        data = Disper_true  # you can choose to use Disper_true or Disper_false
        mean_value = np.mean(data)
        std_value = np.std(data)
        lower_bound = mean_value - std_value
        upper_bound = mean_value + std_value


        edges1 = np.linspace(0, 5, 20)
        ax2.hist(LagTime, bins=edges1, alpha=0.5, label='Counts LagTime')
        ax2.set_xlabel('LagTime(s)')
        #ax2.set_title(f'Station-{sta}')
        ax2.legend()
        
        fig, ax = plt.subplots(figsize=(5, 5))
        edges2 = np.linspace(0, 1, 10)
        ax.hist(Corr_all, bins=edges2, alpha=0.5, label='Counts Corr')
        ax.set_xlabel('Normalized correlation')
        ax.legend()
        
        # 
        #max_count_true = np.max(np.histogram(Disper_true, bins=edges)[0])
        #print(max_count_true)

        '''  
        # add text
        ax.text(mean_value, 1.1 * max_count_true,
                f'Mean: {mean_value:.2f} Std: {std_value:.2f}',
                horizontalalignment='center', verticalalignment='top', fontsize=10)
        '''
        
        # Plot vertical lines to indicate standard deviation range and mean value
        ax1.axvline(x=lower_bound, color='g', linestyle='--', linewidth=1)
        ax1.axvline(x=upper_bound, color='g', linestyle='--', linewidth=1,label=f'std={std_value:.2f}°')
        ax1.axvline(x=mean_value, color='r', linestyle='--', linewidth=1,label=f'mean={mean_value:.2f}°')
        
        ax1.legend()

        plt.show()

# Calculate the QC criterion - SNR
def CalculateSNR(data, Para):
    # Filter to the signal band
    T_min = 2 / Para.Sample_rate  # Lower limit of the period (Nyquist sampling theorem)
    filter_min = max(Para.Ts, T_min)
    filter_max = Para.Te
    data_filter = Bandpass(data, filter_min, filter_max, Para.Sample_rate)
    
    # Define the signal window and noise window
    # Signal window
    Data_len = len(data_filter)
    Vlow_index = min(round((Para.Dist / Para.Vlow) / Para.dt), Data_len)-1  # Index corresponding to the minimum speed
    Vhigh_index = round((Para.Dist / Para.Vhigh) / Para.dt)-1  # Index corresponding to the maximum speed
    
    # Noise window
    if Vlow_index == Data_len:
        Noiselow_index = 0
        Noisehigh_index = Data_len-1
    else:
        Noiselow_index = Vlow_index + 1
        Noisehigh_index = Data_len-1
    
    AmpData = abs(max(data_filter[Vhigh_index:Vlow_index+1]))
    AmpNoise = np.mean(abs(data_filter[Noiselow_index:Noisehigh_index+1]))
    SNR = AmpData / AmpNoise
    
    return SNR

# Calculate the QC criterion - lag-time
def CalculateLagTime(seisData, Para, Flag, souSta, recSta):
    ZZdata = seisData.ZZdata
    EEdata = seisData.EEdata
    ENdata = seisData.ENdata
    NEdata = seisData.NEdata
    NNdata = seisData.NNdata
    
    # Convert data from CF to EGF:
    ZZdata = np.imag(hilbert(ZZdata))
    EEdata = np.imag(hilbert(EEdata))
    ENdata = np.imag(hilbert(ENdata))
    NEdata = np.imag(hilbert(NEdata))
    NNdata = np.imag(hilbert(NNdata))
    
    Data_len = len(ZZdata)  # Length of the data
    
    T_min = 2 / Para.Sample_rate  # Minimum period (Nyquist sampling theorem)
    filter_min = max(Para.Ts, T_min)
    filter_max = Para.Te
    
    # Velocity window, coordinates corresponding to velocity thresholds
    Vel_Window, Vlow_index, Vhigh_index = VelocityWindow(Data_len, Para, Para.Te)
    
    # Velocity filtering of ZZ component data
    ZZdata_win = ZZdata * Vel_Window
    # Calculate amplitude envelope using the entire frequency band signal
    ZZ_envelope = np.abs(hilbert(ZZdata_win))
    # Find the maximum point of the group velocity envelope within the velocity window
    Max_index = np.argmax(ZZ_envelope[Vhigh_index:Vlow_index])
    Max_index += Vhigh_index 
    
    # Design and return the cut window, bandpass upper and lower limits
    Cut_Window, Cut_min, Cut_max = CutWindow(Data_len, Max_index, Para.Te, Para)
    
    
    # Velocity filtering
    EEdata_win = EEdata * Vel_Window
    ENdata_win = ENdata * Vel_Window
    NEdata_win = NEdata * Vel_Window
    NNdata_win = NNdata * Vel_Window
    
    # Data window
    ZZdata_cut = ZZdata_win * Cut_Window
    EEdata_cut = EEdata_win * Cut_Window
    ENdata_cut = ENdata_win * Cut_Window
    NEdata_cut = NEdata_win * Cut_Window
    NNdata_cut = NNdata_win * Cut_Window
    
    # Bandpass filtering
    ZZdata_filter = Bandpass(ZZdata_cut, filter_min, filter_max, Para.Sample_rate)
    EEdata_filter = Bandpass(EEdata_cut, filter_min, filter_max, Para.Sample_rate)
    ENdata_filter = Bandpass(ENdata_cut, filter_min, filter_max, Para.Sample_rate)
    NEdata_filter = Bandpass(NEdata_cut, filter_min, filter_max, Para.Sample_rate)
    NNdata_filter = Bandpass(NNdata_cut, filter_min, filter_max, Para.Sample_rate)
    
    # Signal delay analysis, calculate cross-correlation of RR and ZZ components when correction angle is 0
    RRdata_filter = (EEdata_filter * (-np.sin(np.deg2rad(Para.thet)) * np.sin(np.deg2rad(Para.psi))) +
                     ENdata_filter * (-np.sin(np.deg2rad(Para.thet)) * np.cos(np.deg2rad(Para.psi))) +
                     NEdata_filter * (-np.cos(np.deg2rad(Para.thet)) * np.sin(np.deg2rad(Para.psi))) +
                     NNdata_filter * (-np.cos(np.deg2rad(Para.thet)) * np.cos(np.deg2rad(Para.psi))))
    Cross=np.correlate(RRdata_filter, ZZdata_filter, mode='full')
    center = len(Cross) // 2
    Corss=Cross[center:]
    
    max_corr_index = np.argmax(Corss)
    fs = Para.Sample_rate
    lag_time = max_corr_index / fs
    
    # Plot the cross-correlation for a single station pair
    if Flag.Auto==0:
        fig, ax = plt.subplots()

        ax.plot(Corss)
        # 添加标记最大相关位置的虚线
        ylim = ax.get_ylim()
        ax.plot([max_corr_index, max_corr_index], ylim, 'r--', label=f'Tlag = {lag_time:.1f}s')
        
        ax.set_xlabel(f'Lag Time (1/{fs}) (s)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title(f'Cross-correlation between RR and ZZ')
        ax.legend()
        
        print(f'ZZ-RR delay time: {lag_time:.4f} seconds')
        if Flag.Save_png == 1:
            savePath = os.path.join(Flag.run_path, 'single')
            os.makedirs(savePath, exist_ok=True)
            plt.savefig(os.path.join(savePath, f'lagtime_{souSta}_{recSta}.png'))
        plt.show()
    # return 
    else:
        print(f'ZZ-RR delay time: {lag_time:.4f} seconds')
        return(lag_time)

# Rotate data after angle correct
def RotateTT(Flag,thet,psi):
    #Read the posi and neg data,then rotate
    Folder_path = Flag.data_path
    FileName = Flag.File
    ZZ_path = os.path.join(Folder_path, FileName)
    EE_path = ZZ_path.replace('Z-Z', 'E-E').replace('ZZ', 'EE')
    EN_path = ZZ_path.replace('Z-Z', 'E-N').replace('ZZ', 'EN')
    NE_path = ZZ_path.replace('Z-Z', 'N-E').replace('ZZ', 'NE')
    NN_path = ZZ_path.replace('Z-Z', 'N-N').replace('ZZ', 'NN')
    
    ZZ_data = np.loadtxt(ZZ_path)
    NN_data = np.loadtxt(NN_path)
    NE_data = np.loadtxt(NE_path)
    EN_data = np.loadtxt(EN_path)
    EE_data = np.loadtxt(EE_path)
    
    head= ZZ_data[0:2,:]
    t_axis = ZZ_data[2:, 0]
    
    EE_posi = EE_data[2:, 1]
    EN_posi = EN_data[2:, 1]
    NE_posi = NE_data[2:, 1]
    NN_posi = NN_data[2:, 1]
    
    EE_neg = EE_data[2:, 2]
    EN_neg = EN_data[2:, 2]
    NE_neg = NE_data[2:, 2]
    NN_neg = NN_data[2:, 2]
    
    sind = lambda x: np.sin(np.deg2rad(x))
    cosd = lambda x: np.cos(np.deg2rad(x))
    
    TT_neg = (EE_neg * (-cosd(thet) * cosd(psi)) + 
          EN_neg * (cosd(thet) * sind(psi)) +
          NE_neg * (sind(thet) * cosd(psi)) + 
          NN_neg * (-sind(thet) * sind(psi)))
    
    TT_posi = (EE_posi * (-cosd(thet) * cosd(psi)) + 
          EN_posi * (cosd(thet) * sind(psi)) +
          NE_posi * (sind(thet) * cosd(psi)) + 
          NN_posi * (-sind(thet) * sind(psi)))
    
    TT_data = np.zeros((len(ZZ_data),3))
    TT_data[:2,:]=head
    TT_data[2:,0]=t_axis
    TT_data[2:,1]=TT_posi
    TT_data[2:,2]=TT_neg
    
    return TT_data
=======
import numpy as np
import os
import math
import glob
import re
import sys
import shutil
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
#signal processing
from scipy.signal import hilbert,windows,butter,filtfilt,firwin,fftconvolve
from scipy.fft import fft, ifft
from scipy.interpolate import interp1d

# --- Definition of some basic classes ---
class Para: #definition of parameters
    def __init__(self, Sample_rate=10, nsta=1, dv=0.002, Vlow=1.0, Vhigh=6, 
                Ts=0.6, dT=0.2, Te=10, Low_bound=-180, High_bound=180, 
                T_show=[], T_num=4, minSNR=6, minCorr=0.8, maxLag=1, maxAngle=30):
        self.dt = 1/Sample_rate                 
        self.Sample_rate = Sample_rate # Sample rate of the seismic data(Hz)
        self.nsta = nsta              # Number of stations
        self.dv = dv                  # Velocity interval of group velocity spectrum(no need to modify)
        self.Vlow = Vlow              # Signal window,the lower limit of group wave velocity
        self.Vhigh = Vhigh            # the higher limit of group wave velocity
        self.Ts = Ts                  # the start period of orientation correct
        self.dT = dT                  # the interval of period(usually has little effect on the results) 
        self.Te = Te                  # the end period of orientation correct
        self.Low_bound = Low_bound    # the angle search range,lower boundary
        self.High_bound = High_bound  # the higher boundary,the default interval is 1
        self.T_show = T_show          # the period for which you want to show its angle-correlation coefficient diagram
        self.T_num = T_num            # the window length of time variable filtering,refer to 'EGFAnalysisTimeFreq'
        self.minSNR = minSNR          # QC thresholds
        self.minCorr = minCorr
        self.maxLag = maxLag
        self.maxAngle=maxAngle
        self.T = self.generate_T()    # generate T array
    
    def generate_T(self):
        return np.arange(self.Ts, self.Te + self.dT, self.dT)

class Flag: #definition of methods
    def __init__(self,  Corr_Method=2, Half=0, 
                Save_png=1, Period_setting=0, Auto=0, Rotate=1):
        self.Corr_Method = Corr_Method          # 1: ZR*ZZ / (|ZR| * |ZZ|); 2: ZR*ZZ / |ZZ|^2
                                                # Method 2 is better: better noise resistance, more stable
        self.Half = Half                        # 1: Positive half-axis data; -1: Use negative half-axis data, virtual source and receiver flipped; 
                                                # 0: Average of positive and negative half-axis 
        self.Save_png = Save_png                # 1: Save image; 0: Do not save image
        self.Period_setting = Period_setting    # Period setting (Only works when Auto=0)
                                                # 1：Choose period based on dispersion file; 0: use the default Para.T
        self.Auto = Auto                        # 1: Auto mode,count the azimuth correction of all stations; 0: Normal mode,for single station
        self.Rotate = Rotate                    # Only works when Auto=1; 1: Rotate and save the data that meets the conditions; 0: Don't rotate
        self.run_path = './EXAMPLE'             # The path for running the code, contains input and output
        self.data_path = os.path.join(self.run_path, 'INPUT/CFs/Z-Z/ascii') # Seis data
        self.disp_path = os.path.join(self.run_path, 'INPUT/Disper')   # disper data that requires quality control
        self.ref_disp = os.path.join(self.run_path, 'INPUT/Disper_Ref')    # You need edit it only if Period_setting=1,the path of ZZ component dispersion file

class station: 
    def __init__(self, Lat=0, Lon=0, Name=''):
        self.Lat = Lat      # Latitude
        self.Lon = Lon      # Longitude
        self.Name = Name    # Name
        
class Seis_Data:# Cross-correlation data of the positive half
    def __init__(self):
        self.t_axis = []   # Horizontal axis
        self.ZZdata = []   
        self.ZNdata = []
        self.ZEdata = []
        self.NZdata = []
        self.NNdata = []
        self.NEdata = []
        self.EZdata = []
        self.ENdata = []
        self.EEdata = []

# --- Definition of some functions ---        
def DataRead(Flag):
    if Flag.Auto==0:
        #Open a dialog box and choose a Z-Z component file
        root = tk.Tk()
        #root.withdraw()          
        ZZ_path = filedialog.askopenfilename(
            title="Please selects a ZZ file.",
            filetypes=[("ZZ files", "*.dat"), ("All files", "*.*")]
        )
        if ZZ_path:
            print("User selects the ZZ file.:", ZZ_path)
        else:
            print("User cancels the operation to select a file.")
        root.destroy()
        FileName = os.path.basename(ZZ_path)
        
    elif Flag.Auto==1:
        #Alternatively,input the path of the directory
        Folder_path = Flag.data_path
        FileName = Flag.File
        ZZ_path = os.path.join(Folder_path, FileName)
    
    Disper_file = f'CDisp.T.{FileName}'
    
    #substitute the string to obtain the file path of other component
    ZN_path = ZZ_path.replace('Z-Z', 'Z-N').replace('ZZ', 'ZN')
    ZE_path = ZZ_path.replace('Z-Z', 'Z-E').replace('ZZ', 'ZE')
    NZ_path = ZZ_path.replace('Z-Z', 'N-Z').replace('ZZ', 'NZ')
    EZ_path = ZZ_path.replace('Z-Z', 'E-Z').replace('ZZ', 'EZ')
    EE_path = ZZ_path.replace('Z-Z', 'E-E').replace('ZZ', 'EE')
    EN_path = ZZ_path.replace('Z-Z', 'E-N').replace('ZZ', 'EN')
    NE_path = ZZ_path.replace('Z-Z', 'N-E').replace('ZZ', 'NE')
    NN_path = ZZ_path.replace('Z-Z', 'N-N').replace('ZZ', 'NN')
    
    #load data
    ZZ_data = np.loadtxt(ZZ_path)
    ZN_data = np.loadtxt(ZN_path)
    ZE_data = np.loadtxt(ZE_path)
    NZ_data = np.loadtxt(NZ_path)
    NN_data = np.loadtxt(NN_path)
    NE_data = np.loadtxt(NE_path)
    EZ_data = np.loadtxt(EZ_path)
    EN_data = np.loadtxt(EN_path)
    EE_data = np.loadtxt(EE_path)
    t_axis = ZZ_data[2:, 0] 
    seisData = Seis_Data()
    seisData.t_axis = t_axis
    
    #load station
    staSplit = FileName.split('_')
    staPair = staSplit[1]
    staCell = staPair.split('-')
    sta1 = staCell[0]
    sta2 = staCell[1]
    Source_sta=station()
    Receive_sta=station()
    # use positive axis data
    if Flag.Half == 1:
        Source_sta.Lat=ZZ_data[0, 1];Source_sta.Lon=ZZ_data[0, 0];Source_sta.Name=sta1
        Receive_sta.Lat=ZZ_data[1, 1];Receive_sta.Lon=ZZ_data[1, 0];Receive_sta.Name=sta2
        ZZ_posi = ZZ_data[2:, 1]
        ZN_posi = ZN_data[2:, 1]
        ZE_posi = ZE_data[2:, 1]
        NZ_posi = NZ_data[2:, 1]
        EZ_posi = EZ_data[2:, 1]
    
        EE_posi = EE_data[2:, 1]
        EN_posi = EN_data[2:, 1]
        NE_posi = NE_data[2:, 1]
        NN_posi = NN_data[2:, 1]
    # use half axis data
    elif Flag.Half == -1:
        Source_sta.Lat=ZZ_data[1, 1];Receive_sta.Lon=ZZ_data[1, 0];Receive_sta.Name=sta2
        Receive_sta.Lat=ZZ_data[0, 1];Source_sta.Lon=ZZ_data[0, 0];Source_sta.Name=sta1
        ZZ_posi = ZZ_data[2:, 2]
        ZN_posi = ZN_data[2:, 2]
        ZE_posi = ZE_data[2:, 2]
        NZ_posi = NZ_data[2:, 2]
        EZ_posi = EZ_data[2:, 2]
    
        EE_posi = EE_data[2:, 2]
        EN_posi = EN_data[2:, 2]
        NE_posi = NE_data[2:, 2]
        NN_posi = NN_data[2:, 2]
    # average of positive and negative half-axis 
    elif Flag.Half == 0:
        Source_sta.Lat=ZZ_data[0, 1];Source_sta.Lon=ZZ_data[0, 0];Source_sta.Name=sta1
        Receive_sta.Lat=ZZ_data[1, 1];Receive_sta.Lon=ZZ_data[1, 0];Receive_sta.Name=sta2
        ZZ_posi = (ZZ_data[2:, 1] + ZZ_data[2:, 2]) / 2
        ZN_posi = (ZN_data[2:, 1] + ZN_data[2:, 2]) / 2
        ZE_posi = (ZE_data[2:, 1] + ZE_data[2:, 2]) / 2
        NZ_posi = (NZ_data[2:, 1] + NZ_data[2:, 2]) / 2
        EZ_posi = (EZ_data[2:, 1] + EZ_data[2:, 2]) / 2
        EE_posi = (EE_data[2:, 1] + EE_data[2:, 2]) / 2
        EN_posi = (EN_data[2:, 1] + EN_data[2:, 2]) / 2
        NE_posi = (NE_data[2:, 1] + NE_data[2:, 2]) / 2
        NN_posi = (NN_data[2:, 1] + NN_data[2:, 2]) / 2
    seisData.ZZdata = ZZ_posi
    seisData.ZNdata = ZN_posi
    seisData.ZEdata = ZE_posi
    seisData.NZdata = NZ_posi
    seisData.EZdata = EZ_posi
    seisData.EEdata = EE_posi
    seisData.ENdata = EN_posi
    seisData.NEdata = NE_posi
    seisData.NNdata = NN_posi
    return seisData,Source_sta,Receive_sta,Disper_file

# Determine the period based on the ZZ-dispersion 
def DisperRead(Flag,Para,Disper_file):
    disper_root = Flag.ref_disp
    disper_path = os.path.join(disper_root, Disper_file)

    # Check if the file exists
    if not os.path.exists(disper_path) or Flag.Period_setting == 0:
        print('Use the period data in settings.')
    else:
        # Read dispersion file
        try:
            with open(disper_path, 'r') as fileID:
                # Skip the first two lines
                for _ in range(2):
                    next(fileID)
                # Read the remaining data
                data = np.loadtxt(fileID)
            # Convert to numpy array
            disperdata = np.array(data)
            
            # Find the initial and final indices
            ini_index = np.where(disperdata[:, 3] == 1)[0][0]
            fin_index = np.where(disperdata[:, 3] == 1)[0][-1]
            
            # Set parameters
            Para.Ts = disperdata[ini_index, 0]
            Para.Te = disperdata[fin_index, 0]
            Para.T = np.arange(Para.Ts, Para.Te + Para.dT, Para.dT)  # Assuming Para.dT is defined

        except IOError:
            raise Exception(f'Unable to open file: {disper_path}')

#Traverse all periods to find the best azimuth angle
def Base_SearchT(seisData, Para, Flag, souSta, recSta):
    Angle_T_ZR = np.zeros((len(Para.T), 2))
    Angle_T_RZ = np.zeros((len(Para.T), 2))
    T_show = np.array(Para.T_show)
    for i in range(len(Para.T)):
        Angle_T_ZR[i, 0] = Para.T[i]
        Angle_T_RZ[i, 0] = Para.T[i]
        Angle_corr_ZR_T, Angle_corr_RZ_T, max_angle_T = AngleSearchT(seisData, Para, Flag, Para.T[i])
        if len(T_show) > 0 and np.any(np.isclose(T_show, Para.T[i], atol=1e-8)):
            PlotAngleCorr(Angle_corr_ZR_T, Angle_corr_RZ_T, Para.T[i], max_angle_T, Para, Flag, souSta, recSta)
    
        Angle_T_ZR[i, 1] = max_angle_T[0]
        Angle_T_RZ[i, 1] = max_angle_T[1]
    return Angle_T_ZR, Angle_T_RZ

# The main funtion to calculate the correlation of different azimuth angles for each period
def AngleSearchT(seisData, Para, Flag, T):
    ZZdata = seisData.ZZdata
    ZNdata = seisData.ZNdata
    ZEdata = seisData.ZEdata
    NZdata = seisData.NZdata
    EZdata = seisData.EZdata

    # Convert data from CF to EGF:
    ZZdata = np.imag(hilbert(ZZdata))
    ZNdata = np.imag(hilbert(ZNdata))
    ZEdata = np.imag(hilbert(ZEdata))
    NZdata = np.imag(hilbert(NZdata))
    EZdata = np.imag(hilbert(EZdata))

    Angle = np.arange(Para.Low_bound, Para.High_bound + 1)  # All angles to iterate through
    Angle_len = len(Angle)
    Data_len = len(ZZdata)  # Length of the data

    
    # Apparent velocity window
    Vel_Window, Vlow_index, Vhigh_index = VelocityWindow(Data_len, Para, T)

    # Apply apparent velocity filtering to the ZZ component data:
    ZZdata_win = ZZdata * Vel_Window
    # Calculate group velocity spectrum:
    ZZdata_filter_first = Groupfilter(ZZdata_win, T, Para.Dist, Para.Sample_rate)  # Filtering independent of dT, amplitude spectrum multiplied by Gaussian window in frequency domain
    # Calculate group velocity envelope for the ZZ component:
    ZZ_envelope = np.abs(hilbert(ZZdata_filter_first))
    # Find the maximum point of the group velocity envelope within the apparent velocity window, note this point's position is relative to 'Vhigh_index'
    Max_index = np.argmax(ZZ_envelope[Vhigh_index:Vlow_index])
    Max_index += Vhigh_index 
    # Design cut window, return cut window, bandpass lower and upper limits
    Cut_Window, Cut_min, Cut_max = CutWindow(Data_len, Max_index, T, Para)
    # Apply Hilbert transform to ZE and ZN (shift right by 90 degrees):
    ZEdata_hilbert = np.imag(hilbert(ZEdata))
    ZNdata_hilbert = np.imag(hilbert(ZNdata))
    # Apply Hilbert transform to RZ (shift left by 90 degrees), applying Hilbert transform twice continuously to a signal is equivalent to negating the signal:
    EZdata_hilbert = -np.imag(hilbert(EZdata))
    NZdata_hilbert = -np.imag(hilbert(NZdata))
    # Apply apparent velocity filtering:
    ZEdata_win = ZEdata_hilbert * Vel_Window
    ZNdata_win = ZNdata_hilbert * Vel_Window
    EZdata_win = EZdata_hilbert * Vel_Window
    NZdata_win = NZdata_hilbert * Vel_Window

    ZZdata_filter = Phasefilter(ZZdata_win, T, Max_index, Para)
    ZEdata_filter = Phasefilter(ZEdata_win, T, Max_index, Para)
    ZNdata_filter = Phasefilter(ZNdata_win, T, Max_index, Para)
    EZdata_filter = Phasefilter(EZdata_win, T, Max_index, Para)
    NZdata_filter = Phasefilter(NZdata_win, T, Max_index, Para)

    # First column is correction angle, second column is corresponding correlation coefficient
    Angle_corr_ZR = np.zeros((Angle_len, 2))
    Angle_corr_RZ = np.zeros((Angle_len, 2))

    # Iterate through angles:
    for i in range(Angle_len):
        Angle_corr_ZR[i, 0] = Angle[i]
        Angle_corr_RZ[i, 0] = Angle[i]

        # Rotate data from ZN and ZE components to obtain ZR component data
        ZRdata_filter = ZEdata_filter * (-np.sin(np.deg2rad(Para.psi + Angle[i]))) + ZNdata_filter * (-np.cos(np.deg2rad(Para.psi + Angle[i])))
        # Rotate data from NZ and EZ components to obtain RZ component data:
        RZdata_filter = EZdata_filter * np.sin(np.deg2rad(Para.thet + Angle[i])) + NZdata_filter * np.cos(np.deg2rad(Para.thet + Angle[i]))

        # Perform cross-correlation on the filtered results (phase velocity):
        X = ZZdata_filter[Cut_min:Cut_max]
        Y_ZR = ZRdata_filter[Cut_min:Cut_max]
        Y_RZ = RZdata_filter[Cut_min:Cut_max]

        if Flag.Corr_Method == 1:
            Angle_corr_ZR[i, 1] = np.sum(X * Y_ZR) / (np.sqrt(np.sum(X**2)) * np.sqrt(np.sum(Y_ZR**2)))  # X*Y / (|X| * |Y|) -- p(ZR, ZZ)/sqrt(p(ZR,ZR)) * sqrt(p(ZZ,ZZ)) -- p(): zero-delay cross-correlation
            Angle_corr_RZ[i, 1] = np.sum(X * Y_RZ) / (np.sqrt(np.sum(X**2)) * np.sqrt(np.sum(Y_RZ**2)))
        else:
            Angle_corr_ZR[i, 1] = np.sum(X * Y_ZR) / np.sum(X**2)  # X*Y / |X|^2 -- p(ZR, ZZ)/p(ZZ,ZZ) -- The denominator normalization factor is a constant because the ZZ component does not change
            Angle_corr_RZ[i, 1] = np.sum(X * Y_RZ) / np.sum(X**2)
    
    #Record the azimuth with the largest cross-correlation,ZR in maxangle[0],RZ in maxangle[1]
    maxangle=np.zeros(2)
    ZR_angle=np.argmax(Angle_corr_ZR[:, 1])
    RZ_angle=np.argmax(Angle_corr_RZ[:, 1])
    maxangle[0]=Angle_corr_ZR[ZR_angle, 0] 
    maxangle[1]=Angle_corr_RZ[RZ_angle, 0]
    
    return Angle_corr_ZR, Angle_corr_RZ, maxangle

# The main funtion to calculate the correlation of different azimuth angles in the full frequency band
def AngleSearch(seisData, Para, Flag):
    ZZdata = seisData.ZZdata
    ZNdata = seisData.ZNdata
    ZEdata = seisData.ZEdata
    NZdata = seisData.NZdata
    EZdata = seisData.EZdata
    
    # Convert data from CF to EGF:
    ZZdata = np.imag(hilbert(ZZdata))
    ZNdata = np.imag(hilbert(ZNdata))
    ZEdata = np.imag(hilbert(ZEdata))
    NZdata = np.imag(hilbert(NZdata))
    EZdata = np.imag(hilbert(EZdata))
    
    Angle = np.arange(Para.Low_bound, Para.High_bound + 1)  # All angles to be searched
    Angle_len = len(Angle)
    Data_len = len(ZZdata)  # Length of the data
    
    T_min = 2 / Para.Sample_rate  # Minimum period (Nyquist sampling theorem)
    filter_min = max(Para.Ts, T_min)
    filter_max = Para.Te
    
    # Velocity window, coordinates corresponding to velocity thresholds
    Vel_Window, Vlow_index, Vhigh_index = VelocityWindow(Data_len, Para, Para.Te)
    
    # Velocity filtering of ZZ component data
    ZZdata_win = ZZdata * Vel_Window
    # Calculate amplitude envelope using the entire frequency band signal
    ZZ_envelope = np.abs(hilbert(ZZdata_win))
    # Find the maximum point of the group velocity envelope within the velocity window
    Max_index = np.argmax(ZZ_envelope[Vhigh_index:Vlow_index])
    Max_index += Vhigh_index 
    
    # Design and return the cut window, bandpass upper and lower limits
    Cut_Window, Cut_min, Cut_max = CutWindow(Data_len, Max_index, Para.Te, Para)
    
    # Hilbert transform of ZE and ZN (shifted 90 degrees to the right)
    ZEdata_hilbert = np.imag(hilbert(ZEdata))
    ZNdata_hilbert = np.imag(hilbert(ZNdata))
    # Hilbert transform of EZ (shifted 90 degrees to the left), negative due to two successive Hilbert transforms
    EZdata_hilbert = -np.imag(hilbert(EZdata))
    NZdata_hilbert = -np.imag(hilbert(NZdata))
    
    # Velocity filtering
    ZEdata_win = ZEdata_hilbert * Vel_Window
    ZNdata_win = ZNdata_hilbert * Vel_Window
    EZdata_win = EZdata_hilbert * Vel_Window
    NZdata_win = NZdata_hilbert * Vel_Window
    
    # Data window
    ZZdata_cut = ZZdata_win * Cut_Window
    ZEdata_cut = ZEdata_win * Cut_Window
    ZNdata_cut = ZNdata_win * Cut_Window
    EZdata_cut = EZdata_win * Cut_Window
    NZdata_cut = NZdata_win * Cut_Window
    
    # Bandpass filtering
    ZZdata_filter = Bandpass(ZZdata_cut, filter_min, filter_max, Para.Sample_rate)
    ZEdata_filter = Bandpass(ZEdata_cut, filter_min, filter_max, Para.Sample_rate)
    ZNdata_filter = Bandpass(ZNdata_cut, filter_min, filter_max, Para.Sample_rate)
    EZdata_filter = Bandpass(EZdata_cut, filter_min, filter_max, Para.Sample_rate)
    NZdata_filter = Bandpass(NZdata_cut, filter_min, filter_max, Para.Sample_rate)
    
    # Initialize arrays for angle correction and cross-correlation coefficients
    Angle_corr_ZR = np.zeros((Angle_len, 2))
    Angle_corr_RZ = np.zeros((Angle_len, 2))
    
    temp_corr_ZR = np.zeros(Angle_len)
    temp_corr_RZ = np.zeros(Angle_len)
    
    corr_ZR = np.zeros(Angle_len) # non-normalized
    corr_RZ = np.zeros(Angle_len)
    
    nor_corr_ZR = np.zeros(Angle_len) # normalized
    nor_corr_RZ = np.zeros(Angle_len)
    
    # Traverse all angles 
    for i in range(Angle_len):
        Angle_corr_ZR[i, 0] = Angle[i]
        Angle_corr_RZ[i, 0] = Angle[i]
        
        # Rotate ZN and ZE component data to obtain ZR component data
        ZRdata_filter = ZEdata_filter * (-np.sin(np.deg2rad(Para.psi + Angle[i]))) + ZNdata_filter * (-np.cos(np.deg2rad(Para.psi + Angle[i])))
        # Rotate NZ and EZ component data to obtain RZ component data
        RZdata_filter = EZdata_filter * np.sin(np.deg2rad(Para.thet + Angle[i])) + NZdata_filter * np.cos(np.deg2rad(Para.thet + Angle[i]))
        
        # Cross-correlation of filtered results (phase velocity)
        X = ZZdata_filter[Cut_min:Cut_max + 1]
        Y_ZR = ZRdata_filter[Cut_min:Cut_max + 1]
        Y_RZ = RZdata_filter[Cut_min:Cut_max + 1]
        
        nor_corr_ZR[i] = np.sum(X * Y_ZR) / (np.sqrt(np.sum(X ** 2)) * np.sqrt(np.sum(Y_ZR ** 2)))
        nor_corr_RZ[i] = np.sum(X * Y_RZ) / (np.sqrt(np.sum(X ** 2)) * np.sqrt(np.sum(Y_RZ ** 2)))
        corr_ZR[i] = np.sum(X * Y_ZR) / np.sum(X ** 2)
        corr_RZ[i] = np.sum(X * Y_RZ) / np.sum(X ** 2)
        
        if Flag.Corr_Method == 1:
            Angle_corr_ZR[i, 1] = nor_corr_ZR[i]
            Angle_corr_RZ[i, 1] = nor_corr_RZ[i]
        else:
            Angle_corr_ZR[i, 1] = corr_ZR[i]
            Angle_corr_RZ[i, 1] = corr_RZ[i]
    
    #Record the azimuth with the largest cross-correlation,ZR in maxangle[0],RZ in maxangle[1]
    maxangle=np.zeros(2)
    ZR_angle=np.argmax(corr_ZR)
    RZ_angle=np.argmax(corr_RZ)
    maxangle[0]=Angle_corr_ZR[ZR_angle, 0] 
    maxangle[1]=Angle_corr_RZ[RZ_angle, 0]
    
    # Calculate maximum correlation coefficient(method I)
    #indexZR = np.argmax(temp_corr_ZR)
    #indexRZ = np.argmax(temp_corr_RZ)
    #corrZR = temp_corr_ZR[indexZR]
    #corrRZ = temp_corr_RZ[indexRZ]
    #---------
    #print(corrZR,corrRZ)
    
    # Another way to define the normalized correlation coefficient,correspond to the best angle
    corrZR = nor_corr_ZR[ZR_angle]
    corrRZ = nor_corr_RZ[RZ_angle]
    #---------
    Corr_I = max(corrRZ, corrZR)
    #Record which is greater,corrZR or RZ
    if corrZR>corrRZ:
        index_angle=0
    else:
        index_angle=1
    
    return Angle_corr_ZR, Angle_corr_RZ, Corr_I, index_angle, maxangle

# Remove the angle ​​that deviate greatly from the mean angle 
def ModifyAngle(Angle):
    T = Angle[:, 0]  # Period 
    T_len = len(T)
    Angle_raw = Angle[:, 1]  # Original corrected angle values
    #parameters can be modified
    threshold1 = 45  # First threshold
    threshold2 = 30 # Second threshold
    Ratio = 0.4    # Proportion of points to be removed, if less than this proportion, return the modified data, otherwise return the original data.

    Angle_raw_mean = np.mean(Angle_raw)

    # First screening
    Angle_first = np.copy(Angle_raw)
    Angle_first[np.abs(Angle_raw - Angle_raw_mean) > threshold1] = np.nan
    Angle_first_mean = np.nanmean(Angle_first)
    
    # Second screening
    Angle_second = np.copy(Angle_first)
    Angle_second[np.abs(Angle_first - Angle_first_mean) > threshold2] = np.nan
    angle_mean = np.nanmean(Angle_second) 
    
   
    # Number of removed points and their indices
    index_modify = np.isnan(Angle_second)
    index_save = ~index_modify
    modify_num = np.sum(index_modify)
    modify_Ratio = modify_num / T_len
    
    angle_modified = Angle
    # If the proportion of removed points exceeds the threshold
    if modify_Ratio > Ratio:
        angle_modified[:,1] = Angle_raw
        angle_mean = 0
        print(f'Modify rate: {modify_Ratio}')
    else:
        # Use 'cubic' interpolation to obtain the values of the removed points
        f = interp1d(T[index_save], Angle_raw[index_save], kind='cubic', fill_value=angle_mean, bounds_error=False)
        angle_modified[:,1] = np.copy(Angle_raw)
        angle_modified[index_modify,1] = f(T[index_modify])

    return angle_modified, angle_mean,modify_Ratio

#plot ZZ/RR and ZR/RZ component
def PlotSeis(t_axis, ZZdata, ZRdata, RZdata, RRdata, Para, Flag, souSta, recSta):
    # Perform Hilbert transform on ZR (shifted by 90 degrees):
    ZRdataH = np.imag(hilbert(ZRdata))
    # Perform Hilbert transform on RZ (shifted by -90 degrees):
    RZdataH = -np.imag(hilbert(RZdata))
    # Compute velocity window based on parameters
    Vel_Window, _, _ = VelocityWindow(len(ZZdata), Para, Para.Te)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # First subplot
    ax1.plot(t_axis, ZZdata / np.max(np.abs(ZZdata)), 'k', label='ZZ')
    ax1.plot(t_axis, RRdata / np.max(np.abs(RRdata)), 'r', label='RR')
    ax1.plot(t_axis, Vel_Window, 'g', label='window')  # Velocity window
    ax1.set_xlabel('t(s)', fontsize=20, fontweight='bold', fontname='Times New Roman')
    ax1.set_ylabel('Normalized Amplitude', fontsize=16, fontweight='bold', fontname='Times New Roman')
    ax1.tick_params(axis='both', which='major', labelsize=16)
    ax1.set_title(f'ZZ/RR components (Dist = {Para.Dist:.2f} Km)')
    ax1.set_xlim(0, 100)
    #ax1.set_ylim(-1, 1.2)
    ax1.legend(fontsize=15,loc='upper right')
    
    
    # Second subplot
    ax2.plot(t_axis, ZRdataH / np.max(np.abs(ZRdataH)), 'b', label='ZR')
    ax2.plot(t_axis, RZdataH / np.max(np.abs(RZdataH)), 'r', label='RZ')
    ax2.plot(t_axis, Vel_Window, 'g', label='window')  # Velocity window
    ax2.set_xlabel('t/s')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('ZR*exp(-90);RZ*exp(-90)')
    ax2.set_xlim(0, 100)
    #ax2.set_ylim(-1, 1.2)
    ax2.legend()

    plt.tight_layout()
    
    if Flag.Save_png == 1:
        savePath = os.path.join(Flag.run_path, 'single')
        os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, f'CFs_{souSta}_{recSta}.png'))
    plt.show(block=False)

# plot group velocity spectrum
def PlotSpectrum(Seisdata, Para, souSta, recSta, component, Flag):
    T_len = len(Para.T)
    Data_len = len(Seisdata)
    
    env = np.zeros((Data_len, T_len))
    # Calculate the group velocity spectrum
    for i in range(T_len):
        Vel_Window, Vlow_index, Vhigh_index = VelocityWindow(Data_len, Para, Para.T[i])
        dataWin = Seisdata * Vel_Window
        dataGroup = Groupfilter(dataWin, Para.T[i], Para.Dist, Para.Sample_rate)
        env[:, i] = np.abs(hilbert(dataGroup))
    timeptnum = np.arange(Vhigh_index, Vlow_index + 1)
    time = np.arange(Vhigh_index+1, Vlow_index + 2) / Para.Sample_rate
    
    VPoint = np.arange(Para.Vlow, Para.Vhigh + Para.dv, Para.dv)
    VPpoint_len = len(VPoint)
    
    TravPtV = Para.Dist / time
    GroupVImg = np.zeros((VPpoint_len, T_len))

    # Interpolate waveform data and normalize amplitude
    V_max = np.zeros(T_len)
    for i in range(T_len):
        interp_func = interp1d(TravPtV, env[timeptnum, i] / np.max(env[timeptnum, i]), kind='cubic',fill_value="extrapolate")
        GroupVImg[:, i] = interp_func(VPoint)
        Max_index = np.argmax(GroupVImg[:, i])
        V_max[i] = VPoint[Max_index]
    
    minamp = np.min(GroupVImg)

    # Plotting
    fig, ax = plt.subplots()
    cmap = plt.cm.jet
    cax = ax.imshow(GroupVImg, extent=[Para.T[0], Para.T[-1], VPoint[0], VPoint[-1]], cmap=cmap,aspect='auto', vmin=minamp, vmax=1, origin='lower')
    #np.savetxt('img2.txt',GroupVImg) #test modify
    # Load custom colormap
    # Uncomment and customize the following lines if you have a specific colormap to load
    # import matplotlib.colors as mcolors
    # cmap = mcolors.ListedColormap([...])  # Define your colormap here
    # plt.colormap(cmap)  # Uncomment this line if cmap is defined
    
    ax.plot(Para.T, V_max, 'g-*')
    ax.set_xlabel('Period(s)')
    ax.set_ylabel('Group Vel(Km/s)')
    ax.set_title(f'{component}-Group-Spectrum({souSta}-{recSta})')
    
    fig.colorbar(cax)
    
    if Flag.Save_png == 1:
        savePath = os.path.join(Flag.run_path, 'single')
        os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, f'spectrum_{component}_{souSta}_{recSta}.png'))
    plt.show(block=False)

# plot Corr versus azimuth 
def PlotAngleCorr(Angle_corr_ZR, Angle_corr_RZ, T, maxangle, Para, Flag, souSta, recSta):
    angleZR = maxangle[0]
    angleRZ = maxangle[1]
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 8))
    
    corMax = max(Angle_corr_ZR[:, 1])
    corMax_plot = max(1, corMax)
    
    ax[0].plot(Angle_corr_ZR[:, 0], Angle_corr_ZR[:, 1], 'b')
    ax[0].set_xlabel('Angle')
    ax[0].set_ylabel('Corr')
    if T:
        ax[0].set_title(f'ZR(T = {T:.1f}s), corrMax = {corMax:.2f}')
    else:
        ax[0].set_title(f'ZR(T = {Para.Ts}-{Para.Te}s), corrMax = {corMax:.2f}')
    
    ylim = ax[0].get_ylim()
    ax[0].plot([angleZR, angleZR], ylim, 'g--',label=f'Max corr angle={angleZR}°')
    ax[0].plot(Angle_corr_ZR[:, 0], np.zeros(len(Angle_corr_ZR[:, 0])), 'm--')
    ax[0].plot(Angle_corr_ZR[:, 0], np.ones(len(Angle_corr_ZR[:, 0])) * 0.9, 'r--',label='corr=0.9')
    ax[0].axis([-200, 200, -corMax_plot*1.1, corMax_plot*1.1])
    ax[0].legend(loc='upper right')
    
    corMax = max(Angle_corr_RZ[:, 1])
    corMax_plot = max(1, corMax)
    ax[1].plot(Angle_corr_RZ[:, 0], Angle_corr_RZ[:, 1], 'b')
    ax[1].set_xlabel('Angle')
    ax[1].set_ylabel('Corr')
    ax[1].set_title(f'RZ, corrMax = {corMax:.2f}')
    
    ylim = ax[1].get_ylim()
    ax[1].plot([angleRZ, angleRZ], ylim, 'g--',label=f'Max corr angle={angleRZ}°')
    ax[1].plot(Angle_corr_RZ[:, 0], np.zeros(len(Angle_corr_RZ[:, 0])), 'm--')
    ax[1].plot(Angle_corr_RZ[:, 0], np.ones(len(Angle_corr_RZ[:, 0])) * 0.9, 'r--',label='corr=0.9')
    ax[1].axis([-200, 200, -corMax_plot*1.1, corMax_plot*1.1])
    ax[1].legend(loc='upper right')
    
    plt.tight_layout()
    
    if Flag.Save_png == 1:
        savePath = os.path.join(Flag.run_path, 'single')
        os.makedirs(savePath, exist_ok=True)
        if T:
            plt.savefig(os.path.join(savePath, f'angle_correlation_T{T:.1f}_{souSta}_{recSta}.png'))
        else:
            plt.savefig(os.path.join(savePath, f'angle_Corr_fullband_{souSta}_{recSta}.png'))
    plt.show(block=False)

# plot the best azimuth angle versus period
def PlotTAngle(Angle_T_ZR, Angle_T_RZ, angleZR, angleRZ, souSta, recSta, Flag):
    
    fig, ax = plt.subplots(2, 1, figsize=(8, 6))

    ax[0].plot(Angle_T_ZR[:, 0], Angle_T_ZR[:, 1], 'b-o')
    ax[0].plot(Angle_T_ZR[:, 0], np.full_like(Angle_T_ZR[:, 0], angleZR), 'r--', label=f'Full band correct angle={angleZR}°')
    ax[0].legend()
    ax[0].set_xlabel('T/s')
    ax[0].set_ylabel('Angle')
    ax[0].set_ylim(-180, 180)
    ax[0].set_title(f'ZR-{souSta}-{recSta}')

    ax[1].plot(Angle_T_RZ[:, 0], Angle_T_RZ[:, 1], 'b-o')
    ax[1].plot(Angle_T_RZ[:, 0], np.full_like(Angle_T_RZ[:, 0], angleRZ), 'r--', label=f'Full band correct angle={angleRZ}°')
    ax[1].legend()
    ax[1].set_xlabel('T/s')
    ax[1].set_ylabel('Angle')
    ax[1].set_ylim(-180, 180)
    ax[1].set_title('RZ')

    plt.tight_layout()
    
    if Flag.Save_png == 1:
        savePath = os.path.join(Flag.run_path, 'single')
        os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, f'period_angle_{souSta}_{recSta}.png'))
    plt.show(block=False)

# plot RR/TT component before and after modify
def PlotContrast(t_axis, RRdata, RRmodify, TTdata, TTmodify, Para, Flag, souSta, recSta):
    # Normalize RRdata and RRmodify with the same factor
    RRamp = max(np.max(np.abs(RRdata)), np.max(np.abs(RRmodify)))
    RRdata = RRdata / RRamp
    RRmodify = RRmodify / RRamp

    # Normalize TTdata and TTmodify with the same factor
    TTamp = max(np.max(np.abs(TTdata)), np.max(np.abs(TTmodify)))
    TTdata = TTdata / TTamp
    TTmodify = TTmodify / TTamp

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Plot RR components before and after correction
    ax1.plot(t_axis, RRdata, 'b', label='RR')
    ax1.plot(t_axis, RRmodify, 'r', label='RR-modify')
    ax1.set_xlabel('t/s')
    ax1.set_ylabel('Amplitude')
    ax1.axis([0, 60, -1.2, 1.2])
    ylim = ax1.get_ylim()
    ax1.plot([Para.Dist / Para.Vlow, Para.Dist / Para.Vlow], ylim, 'k--')
    ax1.plot([Para.Dist / Para.Vhigh, Para.Dist / Para.Vhigh], ylim, 'k--')
    ax1.legend(['RR', 'RR-modify', f'Window:[{Para.Vlow:.1f} {Para.Vhigh:.1f}]km/s'],loc='upper right')
    ax1.set_title('RR')

    # Plot TT components before and after correction
    ax2.plot(t_axis, TTdata, 'b', label='TT')
    ax2.plot(t_axis, TTmodify, 'r', label='TT-modify')
    ax2.set_xlabel('t/s')
    ax2.set_ylabel('Amplitude')
    ax2.axis([0, 60, -1.2, 1.2])
    ylim = ax2.get_ylim()
    ax2.plot([Para.Dist / Para.Vlow, Para.Dist / Para.Vlow], ylim, 'k--')
    ax2.plot([Para.Dist / Para.Vhigh, Para.Dist / Para.Vhigh], ylim, 'k--')
    ax2.legend(['TT', 'TT-modify', f'Window:[{Para.Vlow:.1f} {Para.Vhigh:.1f}]km/s'],loc='upper right')
    ax2.set_title('TT')

    plt.tight_layout()
    
    if Flag.Save_png == 1:
        savePath = os.path.join(Flag.run_path, 'single')
        os.makedirs(savePath, exist_ok=True)
        plt.savefig(os.path.join(savePath, f'CFs_contrast_{souSta}_{recSta}.png'))
    plt.show(block=False)

#MFT Gaussian filter
def Groupfilter(data_raw, T, Dist, Sample_rate):
    # data_raw: data to be filtered
    # T: center period
    # Dist: station distance

    alfa = np.array([[0, 100, 250, 500, 1000, 2000, 4000, 20000],
                     [5, 8, 12, 20, 25, 35, 50, 75]])
    guassalfa = interp1d(alfa[0], alfa[1])(Dist)  # Determine the Gaussian filter parameter based on the distance
    data_len = len(data_raw)  # Length of the waveform

    nfft = int(2 ** math.ceil(np.log2(max(data_len, 1024 * Sample_rate))))  # Ensure the FFT length is sufficiently long
    xxfft = fft(data_raw, nfft)  # Perform the FFT
    fxx = np.arange(0, nfft // 2 + 1) / nfft * Sample_rate  # Frequency domain coordinates (first half)
    IIf = np.arange(nfft // 2 + 1)  # Indices for the first half
    JJf = np.arange(nfft // 2 + 1, nfft)  # Indices for the second half

    fc = 1 / T  # Center frequency
    Hf = np.exp(-guassalfa * (fxx - fc) ** 2 / fc ** 2)
    yyfft = np.zeros(nfft)  
    yyfft[IIf] = xxfft[IIf] * Hf  # Apply the Gaussian window
    yyfft[JJf] = np.conj(yyfft[(nfft // 2)-1:0:-1])
    yy = np.real(ifft(yyfft, nfft))  # Perform the inverse FFT
    data_filter = yy[:data_len]  # Filtered result
    return data_filter

#Band pass filter around center period
def Phasefilter(data_raw, T, Max_index, Para):
    # data_raw: Data after "apparent velocity filtering"
    # Max_index: Index corresponding to the peak of the group velocity envelope

    data_len = len(data_raw)  # Length of the data
    filter_len = 2**np.ceil(np.log2(1024 * Para.Sample_rate)).astype(int)  # Length of the filter
    KaiserPara = 6  
    HalfFilterNum = round(filter_len / 2)  # Length of half the filter
    data_raw = np.pad(data_raw, (0, HalfFilterNum), 'constant')  # Append zeros to the end of the waveform
    
    # Band-pass filtering:
    F = (2 / Para.Sample_rate) / T  # Center frequency
    LowF = (2 / Para.Sample_rate) / (T + 0.5 * Para.dT)  # Lower bound of the filter frequency
    HighF = (2 / Para.Sample_rate) / (T - 0.5 * Para.dT)  # Upper bound of the filter frequency
    Filter = firwin(filter_len , [LowF, HighF], window=('kaiser', KaiserPara), pass_zero=False)
    
    # Two-pass filtering (time and time-reverse) to remove phase shift
    winpt = round(Para.T_num * T * Para.Sample_rate)  # Length of the band-pass in terms of points
    if winpt % 2 == 1:  # Ensure winpt is an even number
        winpt += 1  
    wintukey = windows.tukey(winpt, 0.2)  # Tukey (tapered cosine) window
    grouppt = winpt + Max_index  # Point number of band-pass length plus point number of group velocity peak
    tmpWave = np.concatenate([np.zeros(winpt), data_raw[:data_len], np.zeros(winpt)])
    tmpWave[grouppt - winpt // 2:grouppt + winpt // 2] *= wintukey  # Filter here - centered at group velocity peak point with window length T*Tnum
    tmpWave[:grouppt - winpt // 2] = 0
    tmpWave[grouppt + winpt // 2 - 1:] = 0

    NewWinWave = np.zeros(data_len + HalfFilterNum)
    NewWinWave[:data_len] = tmpWave[winpt:winpt + data_len]  # Data after windowing
    FilteredWave = fftconvolve(NewWinWave[:data_len + HalfFilterNum],Filter, mode='same' )  # First pass of filtering
    FilteredWave = FilteredWave[::-1]  # Reverse the result of the first filtering
    FilteredWave = fftconvolve(FilteredWave[:data_len + HalfFilterNum],Filter, mode='same' )  # Second pass of filtering
    FilteredWave = FilteredWave[::-1]

    data_bandfilter = FilteredWave[:data_len]  # Final filtered result
    
    return data_bandfilter

#the signal window,calculate the window time through velocity boundary.
def VelocityWindow(Data_len, Para, T):
    Vel_Window = np.zeros(Data_len)  # Velocity window initialization

    # Calculate indices based on parameters
    Vlow_index = min(round((Para.Dist / Para.Vlow) / Para.dt)-1, Data_len - 1)
    Vhigh_index = round((Para.Dist / Para.Vhigh) / Para.dt)-1

    # Bandpass window
    Vel_Window[int(Vhigh_index):int(Vlow_index)+1] = 1
    
    # Sine window ramps
    Slope_len_before = min(round(T / 2 / Para.dt), int(Vhigh_index))
    Vel_Window[int(Vhigh_index) - Slope_len_before:int(Vhigh_index)] = np.sin(0.5 * np.pi * np.arange(Slope_len_before) / Slope_len_before)

    Slope_len_after = min(round(T / 2 / Para.dt), int(Data_len - Vlow_index)-1)
    Vel_Window[int(Vlow_index)+1 :int(Vlow_index) + Slope_len_after + 1] = np.sin(0.5 * np.pi * (np.arange(Slope_len_after, 0, -1)) / Slope_len_after)
    
    
    return Vel_Window, Vlow_index, Vhigh_index

#design a signal window,just like temporal variable filtering in phase velocity analysis
def CutWindow(Data_len, Max_index, T, Para):
    # Indices corresponding to the velocity window
    Vlow_index = min(round((Para.Dist / Para.Vlow) / Para.dt)-1, Data_len - 1)
    Vhigh_index = round((Para.Dist / Para.Vhigh) / Para.dt)-1
    
    Cut_Window = np.zeros(Data_len)  # Cut window
    Cut_min = max(1, Max_index - round((Para.T_num / 2) * T / Para.dt))-1  # Prevent window out of bounds
    Cut_max = min(Data_len, Max_index + round((Para.T_num / 2) * T / Para.dt))-1
    
    # The time range of the 'time-varying window' should not exceed the 'velocity window':
    Cut_min = max(Cut_min, Vhigh_index)
    Cut_max = min(Cut_max, Vlow_index)
    
    # Window 2: tukeywin(Len, 0.2), with 10% cosine ramp on both sides
    Cut_Window[Cut_min : Cut_max + 1] = windows.tukey(Cut_max - Cut_min + 1, 0.2)
    
    return Cut_Window, Cut_min, Cut_max

def Bandpass(data_raw, filter_min, filter_max, Sample_rate):
    # Design Butterworth bandpass filter
    nyquist_freq = 0.5 * Sample_rate
    lowcut = (1 / filter_max) / nyquist_freq
    highcut = (1 / filter_min) / nyquist_freq
    b, a = butter(2, [lowcut, highcut], btype='band')  # 2nd order Butterworth filter
    
    # Apply zero-phase filtering using filtfilt
    data_filter = filtfilt(b, a, data_raw)
    
    return data_filter   

# Rotate to obtain the T/R component data.  
def CompRotate(Seisdata, thet, psi):
    # Define sin and cos functions that operate in degrees
    sind = lambda x: np.sin(np.deg2rad(x))
    cosd = lambda x: np.cos(np.deg2rad(x))
    
    ZR = Seisdata.ZEdata * (-sind(psi)) + Seisdata.ZNdata * (-cosd(psi))
    RZ = Seisdata.EZdata * sind(thet) + Seisdata.NZdata * cosd(thet)
    RR = (Seisdata.EEdata * (-sind(thet) * sind(psi)) + 
          Seisdata.ENdata * (-sind(thet) * cosd(psi)) +
          Seisdata.NEdata * (-cosd(thet) * sind(psi)) + 
          Seisdata.NNdata * (-cosd(thet) * cosd(psi)))
    TT = (Seisdata.EEdata * (-cosd(thet) * cosd(psi)) + 
          Seisdata.ENdata * (cosd(thet) * sind(psi)) +
          Seisdata.NEdata * (sind(thet) * cosd(psi)) + 
          Seisdata.NNdata * (-sind(thet) * sind(psi)))
    
    return ZR, RZ, RR, TT

# Calculate the distance and azimuth angle between a pair of staions   
def distaz(sta, sto, epa, epo):
    # dk is the distance between two stations in kilometers
    # dd is the distance in degrees
    # daze is the back azimuth angle in 360 degrees
    # dazs is the azimuth angle in 360 degrees

    rad = math.pi / 180.0

    sa  = math.atan(0.993270 * math.tan(sta * rad))  # Calculate station latitude angle
    ea  = math.atan(0.993270 * math.tan(epa * rad))  # Calculate event latitude angle
    ssa = math.sin(sa)
    csa = math.cos(sa)
    so  = sto * rad  # Convert station longitude to radians
    eo  = epo * rad  # Convert event longitude to radians
    sea = math.sin(ea)
    cea = math.cos(ea)
    ces = math.cos(eo - so)
    ses = math.sin(eo - so)

    # Handle special cases where the coordinates are the same
    if sa == ea and sto == epo:
        return 0.0, 0.0, 0.0, 0.0

    if sta == 90.0 and epa == 90.0:
        return 0.0, 0.0, 0.0, 0.0

    if sta == -90.0 and epa == -90.0:
        return 0.0, 0.0, 0.0, 0.0

    # Calculate distance in degrees
    dd = ssa * sea + csa * cea * ces
    if dd != 0.0:
        dd = math.atan(math.sqrt(1.0 - dd * dd) / dd)
    if dd == 0.0:
        dd = math.pi / 2.0
    if dd < 0.0:
        dd = dd + math.pi
    dd = dd / rad
    dk = dd * 111.19  # Convert distance to kilometers

    # Calculate azimuth and back azimuth angles
    dazs = math.atan2(-ses, (ssa / csa * cea - sea * ces))
    daze = math.atan2(ses, (sea * csa / cea - ssa * ces))
    dazs = dazs / rad
    daze = daze / rad
    if dazs < 0.0:
        dazs = dazs + 360.0
    if daze < 0.0:
        daze = daze + 360.0

    return dk, dd, daze, dazs

# Process a patch of stations,this part is not fine,refer to the test of Auto mode
def AutoProcess(Para,Flag):
    # parameters need modify:folderPath,dispPath,flag,stafile
    folderPath = "C:/Users/ycpan/Auser/Seis/Codes/Dispersion analysis/CF_weifang/CF/CFZ_T_R/Z-Z"
    dispPath = "D:/data/Weifang_dispersion/disp_wf_love_orient_new_0.9"

    # List files matching the patterns
    fileList = glob.glob(os.path.join(folderPath, 'ZZ*.dat'))
    dispList = glob.glob(os.path.join(dispPath, 'CD*'))  

    # Load flag data
    flag = np.loadtxt("flag_love.txt")  # Determine whether the station pair has dispersion data,you can choose to plot it separately. 

    # Read station file
    stafile = './sta_all.txt'
    with open(stafile, "r") as file:
        staname = []
        for line in file:
            fields = line.split()
            if len(fields) == 4:
                staname.append(str(fields[0]))

    # Loop through the stations you selected
    for i in range(100,101):
        print(f"Processing  station {i}")
        sta = staname[i]
        selectedFiles = []
        dispFiles = [] 
        
        #select all the ZZ component seis files contain the i-th station
        for file in fileList:
            fileName = os.path.basename(file)
            parts = re.split('[_-]', fileName)
            sta1 = parts[1]
            sta2 = parts[2]
            if sta == sta1 or sta == sta2:
                selectedFiles.append(fileName)
        #select all the dispersion files contain the i-th station
        for disp in dispList:
            dispName = os.path.basename(disp)
            parts = re.split('[_-]', dispName)
            sta1 = parts[1]
            sta2 = parts[2]
            if sta == sta1 or sta == sta2:
                dispFiles.append(dispName)
                
        n = len(selectedFiles)
        angleZR = np.zeros(n)
        angleRZ = np.zeros(n)
        ZR_angle_mean = np.zeros(n)
        RZ_angle_mean = np.zeros(n)
        Disper_true = []
        Disper_false = []
        Corr_all= []
        LagTime = []
        SNRindex = np.zeros(n)
        Corrindex = np.zeros(n)
        
        for j in range(n):
            delete_index = 0  # If the quality is below the control criteria, set index to 1

            # read data
            Flag.File = selectedFiles[j]
            SeisData,Source_sta,Receive_sta,Disper_file=DataRead(Flag)         
            # calculate distance and azimuth angle between a pair of stations
            Para.Dist, _, Para.psi, Para.thet = distaz(Receive_sta.Lat, Receive_sta.Lon, Source_sta.Lat, Source_sta.Lon)
            if Para.Dist < 1:
                continue
            # Output station info
            print(f'Station pairs: {Source_sta.Name} - {Receive_sta.Name}')
            print(f'Dist is: {Para.Dist}')
            print(f'psi is: {Para.psi}')
            print(f'thet is: {Para.thet}')
            #Rotate to obtain the T/R component data.
            SeisData.ZRdata, SeisData.RZdata, SeisData.RRdata, SeisData.TTdata= CompRotate(SeisData, Para.thet, Para.psi)
            
            #calculate SNR of the full band
            SNR = CalculateSNR(SeisData.ZRdata, Para)
            if SNR > Para.minSNR:
                SNRindex[j] = 1
            else:
                delete_index = 1
            
            Angle_corr_ZR, Angle_corr_RZ,Corr_I,angle_Fullband=AngleSearch(SeisData, Para, Flag)
            lagtime=CalculateLagTime(SeisData, Para, Flag)
            LagTime.append(lagtime)
            #Use Corr as quality control
            if Corr_I > Para.minCorr:
                Corrindex[j] = 1
            else:
                delete_index = 1
            Corr_all.append(Corr_I)
            
            ''' 
            #Statistic station pair separately based on whether it has dispersion data
            sta1_index = staname.index(Source_sta.Name)
            sta2_index = staname.index(Receive_sta.Name)
            if flag[sta1_index, sta2_index] == 1:
                Disper_true.append(angle_Fullband[0])
                Corr.append(Corr_I)
                print(Corr_I)
            if flag[sta1_index, sta2_index] == 0:
                Disper_false.append(angle_Fullband[0])
                delete_index = 0
            '''
            Disper_true.append(angle_Fullband[0])

            
            '''  
            #delete the file with bad quality
            if delete_index == 1:
                parts = re.split('[_-]', Flag.File)
                sta1 = parts[1]
                sta2 = parts[2]
                deletename = f'{sta1}-{sta2}'
                for disp in dispList:
                    if deletename in disp:
                        os.remove(disp)
                        print(f'delete file:{disp}')
            '''
            
            #Disper_true.append(angle_Fullband[0])
            print(f'Full band correction angle for ZR: {angle_Fullband[0]}°')
            print(f'Full band correction angle for RZ: {angle_Fullband[1]}°')
            
        #plot statistic histogram
        edges = np.linspace(-180, 180, 80)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        ax1.hist(Disper_true, bins=edges, alpha=0.5, label='Counts disp')
        #ax1.hist(Disper_false, bins=edges, alpha=0.5, label='Counts no disp')

        ax1.set_xlabel('Correction angle(°)')
        ax1.set_title(f'Station-{sta}')

        # compute mean and standard deviation
        data = Disper_true  # you can choose to use Disper_true or Disper_false
        mean_value = np.mean(data)
        std_value = np.std(data)
        lower_bound = mean_value - std_value
        upper_bound = mean_value + std_value


        edges1 = np.linspace(0, 5, 20)
        ax2.hist(LagTime, bins=edges1, alpha=0.5, label='Counts LagTime')
        ax2.set_xlabel('LagTime(s)')
        #ax2.set_title(f'Station-{sta}')
        ax2.legend()
        
        fig, ax = plt.subplots(figsize=(5, 5))
        edges2 = np.linspace(0, 1, 10)
        ax.hist(Corr_all, bins=edges2, alpha=0.5, label='Counts Corr')
        ax.set_xlabel('Normalized correlation')
        ax.legend()
        
        # 
        #max_count_true = np.max(np.histogram(Disper_true, bins=edges)[0])
        #print(max_count_true)

        '''  
        # add text
        ax.text(mean_value, 1.1 * max_count_true,
                f'Mean: {mean_value:.2f} Std: {std_value:.2f}',
                horizontalalignment='center', verticalalignment='top', fontsize=10)
        '''
        
        # Plot vertical lines to indicate standard deviation range and mean value
        ax1.axvline(x=lower_bound, color='g', linestyle='--', linewidth=1)
        ax1.axvline(x=upper_bound, color='g', linestyle='--', linewidth=1,label=f'std={std_value:.2f}°')
        ax1.axvline(x=mean_value, color='r', linestyle='--', linewidth=1,label=f'mean={mean_value:.2f}°')
        
        ax1.legend()

        plt.show()

# Calculate the QC criterion - SNR
def CalculateSNR(data, Para):
    # Filter to the signal band
    T_min = 2 / Para.Sample_rate  # Lower limit of the period (Nyquist sampling theorem)
    filter_min = max(Para.Ts, T_min)
    filter_max = Para.Te
    data_filter = Bandpass(data, filter_min, filter_max, Para.Sample_rate)
    
    # Define the signal window and noise window
    # Signal window
    Data_len = len(data_filter)
    Vlow_index = min(round((Para.Dist / Para.Vlow) / Para.dt), Data_len)-1  # Index corresponding to the minimum speed
    Vhigh_index = round((Para.Dist / Para.Vhigh) / Para.dt)-1  # Index corresponding to the maximum speed
    
    # Noise window
    if Vlow_index == Data_len:
        Noiselow_index = 0
        Noisehigh_index = Data_len-1
    else:
        Noiselow_index = Vlow_index + 1
        Noisehigh_index = Data_len-1
    
    AmpData = abs(max(data_filter[Vhigh_index:Vlow_index+1]))
    AmpNoise = np.mean(abs(data_filter[Noiselow_index:Noisehigh_index+1]))
    SNR = AmpData / AmpNoise
    
    return SNR

# Calculate the QC criterion - lag-time
def CalculateLagTime(seisData, Para, Flag, souSta, recSta):
    ZZdata = seisData.ZZdata
    EEdata = seisData.EEdata
    ENdata = seisData.ENdata
    NEdata = seisData.NEdata
    NNdata = seisData.NNdata
    
    # Convert data from CF to EGF:
    ZZdata = np.imag(hilbert(ZZdata))
    EEdata = np.imag(hilbert(EEdata))
    ENdata = np.imag(hilbert(ENdata))
    NEdata = np.imag(hilbert(NEdata))
    NNdata = np.imag(hilbert(NNdata))
    
    Data_len = len(ZZdata)  # Length of the data
    
    T_min = 2 / Para.Sample_rate  # Minimum period (Nyquist sampling theorem)
    filter_min = max(Para.Ts, T_min)
    filter_max = Para.Te
    
    # Velocity window, coordinates corresponding to velocity thresholds
    Vel_Window, Vlow_index, Vhigh_index = VelocityWindow(Data_len, Para, Para.Te)
    
    # Velocity filtering of ZZ component data
    ZZdata_win = ZZdata * Vel_Window
    # Calculate amplitude envelope using the entire frequency band signal
    ZZ_envelope = np.abs(hilbert(ZZdata_win))
    # Find the maximum point of the group velocity envelope within the velocity window
    Max_index = np.argmax(ZZ_envelope[Vhigh_index:Vlow_index])
    Max_index += Vhigh_index 
    
    # Design and return the cut window, bandpass upper and lower limits
    Cut_Window, Cut_min, Cut_max = CutWindow(Data_len, Max_index, Para.Te, Para)
    
    
    # Velocity filtering
    EEdata_win = EEdata * Vel_Window
    ENdata_win = ENdata * Vel_Window
    NEdata_win = NEdata * Vel_Window
    NNdata_win = NNdata * Vel_Window
    
    # Data window
    ZZdata_cut = ZZdata_win * Cut_Window
    EEdata_cut = EEdata_win * Cut_Window
    ENdata_cut = ENdata_win * Cut_Window
    NEdata_cut = NEdata_win * Cut_Window
    NNdata_cut = NNdata_win * Cut_Window
    
    # Bandpass filtering
    ZZdata_filter = Bandpass(ZZdata_cut, filter_min, filter_max, Para.Sample_rate)
    EEdata_filter = Bandpass(EEdata_cut, filter_min, filter_max, Para.Sample_rate)
    ENdata_filter = Bandpass(ENdata_cut, filter_min, filter_max, Para.Sample_rate)
    NEdata_filter = Bandpass(NEdata_cut, filter_min, filter_max, Para.Sample_rate)
    NNdata_filter = Bandpass(NNdata_cut, filter_min, filter_max, Para.Sample_rate)
    
    # Signal delay analysis, calculate cross-correlation of RR and ZZ components when correction angle is 0
    RRdata_filter = (EEdata_filter * (-np.sin(np.deg2rad(Para.thet)) * np.sin(np.deg2rad(Para.psi))) +
                     ENdata_filter * (-np.sin(np.deg2rad(Para.thet)) * np.cos(np.deg2rad(Para.psi))) +
                     NEdata_filter * (-np.cos(np.deg2rad(Para.thet)) * np.sin(np.deg2rad(Para.psi))) +
                     NNdata_filter * (-np.cos(np.deg2rad(Para.thet)) * np.cos(np.deg2rad(Para.psi))))
    Cross=np.correlate(RRdata_filter, ZZdata_filter, mode='full')
    center = len(Cross) // 2
    Corss=Cross[center:]
    
    max_corr_index = np.argmax(Corss)
    fs = Para.Sample_rate
    lag_time = max_corr_index / fs
    
    # Plot the cross-correlation for a single station pair
    if Flag.Auto==0:
        fig, ax = plt.subplots()

        ax.plot(Corss)
        # 添加标记最大相关位置的虚线
        ylim = ax.get_ylim()
        ax.plot([max_corr_index, max_corr_index], ylim, 'r--', label=f'Tlag = {lag_time:.1f}s')
        
        ax.set_xlabel(f'Lag Time (1/{fs}) (s)')
        ax.set_ylabel('Cross-correlation')
        ax.set_title(f'Cross-correlation between RR and ZZ')
        ax.legend()
        
        print(f'ZZ-RR delay time: {lag_time:.4f} seconds')
        if Flag.Save_png == 1:
            savePath = os.path.join(Flag.run_path, 'single')
            os.makedirs(savePath, exist_ok=True)
            plt.savefig(os.path.join(savePath, f'lagtime_{souSta}_{recSta}.png'))
        plt.show()
    # return 
    else:
        print(f'ZZ-RR delay time: {lag_time:.4f} seconds')
        return(lag_time)

# Rotate data after angle correct
def RotateTT(Flag,thet,psi):
    #Read the posi and neg data,then rotate
    Folder_path = Flag.data_path
    FileName = Flag.File
    ZZ_path = os.path.join(Folder_path, FileName)
    EE_path = ZZ_path.replace('Z-Z', 'E-E').replace('ZZ', 'EE')
    EN_path = ZZ_path.replace('Z-Z', 'E-N').replace('ZZ', 'EN')
    NE_path = ZZ_path.replace('Z-Z', 'N-E').replace('ZZ', 'NE')
    NN_path = ZZ_path.replace('Z-Z', 'N-N').replace('ZZ', 'NN')
    
    ZZ_data = np.loadtxt(ZZ_path)
    NN_data = np.loadtxt(NN_path)
    NE_data = np.loadtxt(NE_path)
    EN_data = np.loadtxt(EN_path)
    EE_data = np.loadtxt(EE_path)
    
    head= ZZ_data[0:2,:]
    t_axis = ZZ_data[2:, 0]
    
    EE_posi = EE_data[2:, 1]
    EN_posi = EN_data[2:, 1]
    NE_posi = NE_data[2:, 1]
    NN_posi = NN_data[2:, 1]
    
    EE_neg = EE_data[2:, 2]
    EN_neg = EN_data[2:, 2]
    NE_neg = NE_data[2:, 2]
    NN_neg = NN_data[2:, 2]
    
    sind = lambda x: np.sin(np.deg2rad(x))
    cosd = lambda x: np.cos(np.deg2rad(x))
    
    TT_neg = (EE_neg * (-cosd(thet) * cosd(psi)) + 
          EN_neg * (cosd(thet) * sind(psi)) +
          NE_neg * (sind(thet) * cosd(psi)) + 
          NN_neg * (-sind(thet) * sind(psi)))
    
    TT_posi = (EE_posi * (-cosd(thet) * cosd(psi)) + 
          EN_posi * (cosd(thet) * sind(psi)) +
          NE_posi * (sind(thet) * cosd(psi)) + 
          NN_posi * (-sind(thet) * sind(psi)))
    
    TT_data = np.zeros((len(ZZ_data),3))
    TT_data[:2,:]=head
    TT_data[2:,0]=t_axis
    TT_data[2:,1]=TT_posi
    TT_data[2:,2]=TT_neg
    
    return TT_data
>>>>>>> d99add667d16fa586fa3449c3c0684451c2b1264
