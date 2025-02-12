<<<<<<< HEAD
from main import *

# Example for statistic all stations
Para=Para(Ts=1,Te=5,nsta=1)
Flag=Flag(Half=0,Period_setting=1,Auto=1,Rotate=0,Corr_Method = 1)

Flag.run_path = './EXAMPLE'
runPath = Flag.run_path
Flag.data_path = os.path.join(runPath, 'INPUT/CFs/Z-Z/ascii')
Flag.disp_path = os.path.join(runPath, 'INPUT/Disper')
folderPath = Flag.data_path
dispPath = Flag.disp_path
n=Para.nsta

# List files matching the patterns
fileList = glob.glob(os.path.join(folderPath, 'ZZ*.dat'))
dispList = glob.glob(os.path.join(dispPath, 'CD*'))  

# Load flag data
#flag = np.loadtxt("EXAMPLE/flag_love.txt")  # Determine whether the station pair has dispersion data,you can choose to plot it separately. 

# Read station file
stafile = os.path.join(runPath, 'INPUT/sta_all.txt')
with open(stafile, "r") as file:
    staname = []
    for line in file:

        fields = line.split()
        if len(fields) == 4:
            staname.append(str(fields[0]))
Corr = [] #record Corr_I for all stations
angle_good = [] #record the correct angle satisfy criterion
LagTime_all = [] #record lagtime for all stations
Angle_all = [] #record correct angle for all stations
SNR_all = []

# Loop through the stations you selected
for i in range(0,n):
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
    Lagindex = np.zeros(n)
    
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
        SNR_all.append(SNR)
        if SNR > Para.minSNR:
            SNRindex[j] = 1
        else:
            delete_index = 1
        
        Angle_corr_ZR, Angle_corr_RZ,Corr_I,angle_index,angle_Fullband=AngleSearch(SeisData, Para, Flag)
        lagtime=CalculateLagTime(SeisData, Para, Flag, Source_sta.Name, Receive_sta.Name)
        LagTime.append(lagtime)
        LagTime_all.append(lagtime)
        
        #Use Corr as quality control
        if Corr_I > Para.minCorr:
            Corrindex[j] = 1
        else:
            delete_index = 1
        Corr_all.append(Corr_I)
        Corr.append(Corr_I) #!statistic Corr_I for all stations
        Angle_all.append(angle_Fullband[angle_index]) #!statistic correct angle for all stations

        #Use Lag-time as quality control
        if lagtime <= Para.maxLag:
            Corrindex[j] = 1
        else:
            delete_index = 1
        
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
        Disper_true.append(angle_Fullband[angle_index])
        
        if Flag.Rotate == 1:
            #rotate based on the search angle if delete_index=0
            if delete_index==0 and abs(angle_Fullband[angle_index])<=Para.maxAngle:
                #Use corrected angle to rotate R/T component
                thet_m = Para.thet + angle_Fullband[angle_index]
                psi_m = Para.psi + angle_Fullband[angle_index]
                TT_rotate = RotateTT(Flag,thet_m,psi_m)
                #angle_good.append(angle_Fullband[angle_index])
                os.makedirs(os.path.join(runPath, 'OUTPUT/rotateData'), exist_ok=True)
                #save TT_rotate
                np.savetxt(os.path.join(runPath, f'OUTPUT/rotateData/TT_{Source_sta.Name}-{Receive_sta.Name}_rot.dat'), TT_rotate)
        else:
            if delete_index==0 and abs(angle_Fullband[angle_index])<=Para.maxAngle:
                os.makedirs(os.path.join(runPath, 'OUTPUT/dispData'), exist_ok=True)
                #copy the file with good quality
                parts = re.split('[_-]', Flag.File)
                sta1 = parts[1]
                sta2 = parts[2]
                copyname = f'{sta1}-{sta2}'
                for disp in dispList:
                    if copyname in disp:
                        src_file = disp
                        dst_folder = os.path.join(runPath, "OUTPUT/dispData")
                        dst_file = os.path.join(dst_folder, os.path.basename(src_file))
                        shutil.copy(src_file, dst_file)
                        print(f'delete file:{disp}')
        #Disper_true.append(angle_Fullband[0])
        print(f'Full band correction angle for ZR: {angle_Fullband[0]}°')
        print(f'Full band correction angle for RZ: {angle_Fullband[1]}°')
        print(f'Correlation coefficient: {Corr_I}')
       
    #plot statistic histogram
    edges = np.linspace(-180, 180, 80)
    #plot statistic histogram
    # path to save figures
    figure_path = os.path.join(runPath, 'OUTPUT', 'Figure', sta)
    if Flag.Save_png == 1:
        os.makedirs(figure_path, exist_ok=True)
    
    # Figure 1: Angle correction histogram
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    edges = np.linspace(-180, 180, 80)
    
    ax1.hist(Disper_true, bins=edges, alpha=0.5, label='Counts disp')
    ax1.set_xlabel('Correction angle(°)')
    ax1.set_title(f'Station-{sta}')
    
    # Calculate mean and standard deviation
    data = Disper_true
    mean_value = np.mean(data)
    std_value = np.std(data)
    lower_bound = mean_value - std_value
    upper_bound = mean_value + std_value
    
    # Add vertical lines for mean and standard deviation
    ax1.axvline(x=lower_bound, color='g', linestyle='--', linewidth=1)
    ax1.axvline(x=upper_bound, color='g', linestyle='--', linewidth=1, label=f'std={std_value:.2f}°')
    ax1.axvline(x=mean_value, color='r', linestyle='--', linewidth=1, label=f'mean={mean_value:.2f}°')
    ax1.legend()
    
    # Save angle correction figure
    if Flag.Save_png == 1:
        fig1.savefig(os.path.join(figure_path, f'angle_correction.png'))
    plt.close(fig1)
    
    # Figure 2: LagTime histogram
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    edges1 = np.linspace(0, 5, 20)
    
    ax2.hist(LagTime, bins=edges1, alpha=0.5, label='Counts LagTime')
    ax2.set_xlabel('LagTime(s)')
    ax2.set_title(f'Station-{sta}')
    ax2.legend()
    
    # Save LagTime figure
    if Flag.Save_png == 1:
        fig2.savefig(os.path.join(figure_path, f'lag_time.png'))
    plt.close(fig2)
    
    # Figure 3: Correlation coefficient histogram
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111)
    edges2 = np.linspace(0, 1, 10)
    
    ax3.hist(Corr_all, bins=edges2, alpha=0.5, label='Counts Corr')
    ax3.set_xlabel('Normalized correlation')
    ax3.set_title(f'Station-{sta}')
    ax3.legend()
    
    # Save correlation figure
    if Flag.Save_png == 1:
        fig3.savefig(os.path.join(figure_path, f'Rnorm.png'))
    plt.close(fig3)
    
    # Figure 4: SNR histogram
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111)
    edges3 = np.linspace(0, 20, 40)  # Adjust the range based on your SNR values
    
    ax4.hist(SNR_all, bins=edges3, alpha=0.5, label='Counts SNR')
    ax4.set_xlabel('SNR')
    ax4.set_title(f'Station-{sta}')
    ax4.legend()
    
    # Save SNR figure
    if Flag.Save_png == 1:
        fig4.savefig(os.path.join(figure_path, f'SNR.png'))
=======
from main import *

# Example for statistic all stations
Para=Para(Ts=1,Te=5,nsta=1)
Flag=Flag(Half=0,Period_setting=1,Auto=1,Rotate=0,Corr_Method = 1)

Flag.run_path = './EXAMPLE'
runPath = Flag.run_path
Flag.data_path = os.path.join(runPath, 'INPUT/CFs/Z-Z/ascii')
Flag.disp_path = os.path.join(runPath, 'INPUT/Disper')
folderPath = Flag.data_path
dispPath = Flag.disp_path
n=Para.nsta

# List files matching the patterns
fileList = glob.glob(os.path.join(folderPath, 'ZZ*.dat'))
dispList = glob.glob(os.path.join(dispPath, 'CD*'))  

# Load flag data
#flag = np.loadtxt("EXAMPLE/flag_love.txt")  # Determine whether the station pair has dispersion data,you can choose to plot it separately. 

# Read station file
stafile = os.path.join(runPath, 'INPUT/sta_all.txt')
with open(stafile, "r") as file:
    staname = []
    for line in file:

        fields = line.split()
        if len(fields) == 4:
            staname.append(str(fields[0]))
Corr = [] #record Corr_I for all stations
angle_good = [] #record the correct angle satisfy criterion
LagTime_all = [] #record lagtime for all stations
Angle_all = [] #record correct angle for all stations
SNR_all = []

# Loop through the stations you selected
for i in range(0,n):
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
    Lagindex = np.zeros(n)
    
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
        SNR_all.append(SNR)
        if SNR > Para.minSNR:
            SNRindex[j] = 1
        else:
            delete_index = 1
        
        Angle_corr_ZR, Angle_corr_RZ,Corr_I,angle_index,angle_Fullband=AngleSearch(SeisData, Para, Flag)
        lagtime=CalculateLagTime(SeisData, Para, Flag, Source_sta.Name, Receive_sta.Name)
        LagTime.append(lagtime)
        LagTime_all.append(lagtime)
        
        #Use Corr as quality control
        if Corr_I > Para.minCorr:
            Corrindex[j] = 1
        else:
            delete_index = 1
        Corr_all.append(Corr_I)
        Corr.append(Corr_I) #!statistic Corr_I for all stations
        Angle_all.append(angle_Fullband[angle_index]) #!statistic correct angle for all stations

        #Use Lag-time as quality control
        if lagtime <= Para.maxLag:
            Corrindex[j] = 1
        else:
            delete_index = 1
        
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
        Disper_true.append(angle_Fullband[angle_index])
        
        if Flag.Rotate == 1:
            #rotate based on the search angle if delete_index=0
            if delete_index==0 and abs(angle_Fullband[angle_index])<=Para.maxAngle:
                #Use corrected angle to rotate R/T component
                thet_m = Para.thet + angle_Fullband[angle_index]
                psi_m = Para.psi + angle_Fullband[angle_index]
                TT_rotate = RotateTT(Flag,thet_m,psi_m)
                #angle_good.append(angle_Fullband[angle_index])
                os.makedirs(os.path.join(runPath, 'OUTPUT/rotateData'), exist_ok=True)
                #save TT_rotate
                np.savetxt(os.path.join(runPath, f'OUTPUT/rotateData/TT_{Source_sta.Name}-{Receive_sta.Name}_rot.dat'), TT_rotate)
        else:
            if delete_index==0 and abs(angle_Fullband[angle_index])<=Para.maxAngle:
                os.makedirs(os.path.join(runPath, 'OUTPUT/dispData'), exist_ok=True)
                #copy the file with good quality
                parts = re.split('[_-]', Flag.File)
                sta1 = parts[1]
                sta2 = parts[2]
                copyname = f'{sta1}-{sta2}'
                for disp in dispList:
                    if copyname in disp:
                        src_file = disp
                        dst_folder = os.path.join(runPath, "OUTPUT/dispData")
                        dst_file = os.path.join(dst_folder, os.path.basename(src_file))
                        shutil.copy(src_file, dst_file)
                        print(f'delete file:{disp}')
        #Disper_true.append(angle_Fullband[0])
        print(f'Full band correction angle for ZR: {angle_Fullband[0]}°')
        print(f'Full band correction angle for RZ: {angle_Fullband[1]}°')
        print(f'Correlation coefficient: {Corr_I}')
       
    #plot statistic histogram
    edges = np.linspace(-180, 180, 80)
    #plot statistic histogram
    # path to save figures
    figure_path = os.path.join(runPath, 'OUTPUT', 'Figure', sta)
    if Flag.Save_png == 1:
        os.makedirs(figure_path, exist_ok=True)
    
    # Figure 1: Angle correction histogram
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111)
    edges = np.linspace(-180, 180, 80)
    
    ax1.hist(Disper_true, bins=edges, alpha=0.5, label='Counts disp')
    ax1.set_xlabel('Correction angle(°)')
    ax1.set_title(f'Station-{sta}')
    
    # Calculate mean and standard deviation
    data = Disper_true
    mean_value = np.mean(data)
    std_value = np.std(data)
    lower_bound = mean_value - std_value
    upper_bound = mean_value + std_value
    
    # Add vertical lines for mean and standard deviation
    ax1.axvline(x=lower_bound, color='g', linestyle='--', linewidth=1)
    ax1.axvline(x=upper_bound, color='g', linestyle='--', linewidth=1, label=f'std={std_value:.2f}°')
    ax1.axvline(x=mean_value, color='r', linestyle='--', linewidth=1, label=f'mean={mean_value:.2f}°')
    ax1.legend()
    
    # Save angle correction figure
    if Flag.Save_png == 1:
        fig1.savefig(os.path.join(figure_path, f'angle_correction.png'))
    plt.close(fig1)
    
    # Figure 2: LagTime histogram
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    edges1 = np.linspace(0, 5, 20)
    
    ax2.hist(LagTime, bins=edges1, alpha=0.5, label='Counts LagTime')
    ax2.set_xlabel('LagTime(s)')
    ax2.set_title(f'Station-{sta}')
    ax2.legend()
    
    # Save LagTime figure
    if Flag.Save_png == 1:
        fig2.savefig(os.path.join(figure_path, f'lag_time.png'))
    plt.close(fig2)
    
    # Figure 3: Correlation coefficient histogram
    fig3 = plt.figure(figsize=(10, 8))
    ax3 = fig3.add_subplot(111)
    edges2 = np.linspace(0, 1, 10)
    
    ax3.hist(Corr_all, bins=edges2, alpha=0.5, label='Counts Corr')
    ax3.set_xlabel('Normalized correlation')
    ax3.set_title(f'Station-{sta}')
    ax3.legend()
    
    # Save correlation figure
    if Flag.Save_png == 1:
        fig3.savefig(os.path.join(figure_path, f'Rnorm.png'))
    plt.close(fig3)
    
    # Figure 4: SNR histogram
    fig4 = plt.figure(figsize=(10, 8))
    ax4 = fig4.add_subplot(111)
    edges3 = np.linspace(0, 20, 40)  # Adjust the range based on your SNR values
    
    ax4.hist(SNR_all, bins=edges3, alpha=0.5, label='Counts SNR')
    ax4.set_xlabel('SNR')
    ax4.set_title(f'Station-{sta}')
    ax4.legend()
    
    # Save SNR figure
    if Flag.Save_png == 1:
        fig4.savefig(os.path.join(figure_path, f'SNR.png'))
>>>>>>> d99add667d16fa586fa3449c3c0684451c2b1264
    plt.close(fig4)