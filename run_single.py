from main import *

# Example usage, for a single pair data
if __name__ == "__main__":
    Para=Para(Ts=1,Te=5,T_show=[1,2])
    Flag=Flag(Half=0,Period_setting=0,Auto=0,Corr_Method = 2,Save_png=1)
    Flag.run_path = './EXAMPLE'
    runPath = Flag.run_path
    Flag.ref_disp = os.path.join(runPath, 'INPUT/Disper_Ref')
    if Flag.Auto == 1:
        AutoProcess(Para,Flag)
        sys.exit()
    SeisData,Source_sta,Receive_sta,Disper_file=DataRead(Flag)
    DisperRead(Flag,Para,Disper_file)
    #calculate and print the information of station
    Para.Dist, dd, Para.psi, Para.thet= distaz(Receive_sta.Lat, Receive_sta.Lon, Source_sta.Lat, Source_sta.Lon)


    print(f"Station pairs: {Source_sta.Name} - {Receive_sta.Name}")
    print(f"Dist is: {Para.Dist}")
    print(f"back_azi is: {Para.psi}")
    print(f"azi is: {Para.thet}")
    #Rotate to obtain the T/R component data.
    SeisData.ZRdata, SeisData.RZdata, SeisData.RRdata, SeisData.TTdata = CompRotate(SeisData, Para.thet, Para.psi)
    
    PlotSeis(SeisData.t_axis, SeisData.ZZdata, SeisData.ZRdata, SeisData.RZdata, SeisData.RRdata, Para, Flag, Source_sta.Name, Receive_sta.Name)
    PlotSpectrum(SeisData.ZZdata, Para, Source_sta.Name, Receive_sta.Name,'ZZ', Flag)
    #PlotSpectrum(SeisData.ZRdata, Para, Source_sta.Name, Receive_sta.Name,'ZR', Flag)
    #PlotSpectrum(SeisData.RZdata, Para, Source_sta.Name, Receive_sta.Name,'RZ', Flag)
    
    #Full Band
    Angle_corr_ZR, Angle_corr_RZ,Corr_I,angle_index,angle_Fullband=AngleSearch(SeisData, Para, Flag)
    PlotAngleCorr(Angle_corr_ZR, Angle_corr_RZ, 0, angle_Fullband, Para, Flag, Source_sta.Name, Receive_sta.Name)
    
    #Each Period
    Angle_T_ZR, Angle_T_RZ=Base_SearchT(SeisData, Para, Flag, Source_sta.Name, Receive_sta.Name)
    PlotTAngle(Angle_T_ZR, Angle_T_RZ, angle_Fullband[0], angle_Fullband[1], Source_sta.Name, Receive_sta.Name, Flag)
    #Remove outliers
    Angle_T_ZR_modify, ZR_angle_mean, modify_Ratio_ZR=ModifyAngle(Angle_T_ZR)
    Angle_T_RZ_modify, RZ_angle_mean, modify_Ratio_RZ=ModifyAngle(Angle_T_RZ)
    #PlotTAngle(Angle_T_ZR_modify, Angle_T_RZ_modify, angle_Fullband[0], angle_Fullband[1], Source_sta.Name, Receive_sta.Name)
    
    #print result
    print(f'Full band correction angle for ZR: {angle_Fullband[0]}°')
    print(f'Full band correction angle for RZ: {angle_Fullband[1]}°')
    print(f'Average correction angle for ZR: {ZR_angle_mean}°')
    print(f'Average correction angle for RZ: {RZ_angle_mean}°')
    print(f'Correlation coefficient: {Corr_I}')
    
    #Use corrected angle to rotate R/T component
    thet_m = Para.thet + angle_Fullband[1]
    psi_m = Para.psi + angle_Fullband[1]
    ZRdata_modify, RZdata_modify, RRdata_modify, TTdata_modify=CompRotate(SeisData, thet_m, psi_m)
    PlotContrast(SeisData.t_axis, SeisData.RRdata, RRdata_modify, SeisData.TTdata, TTdata_modify, Para, Flag, Source_sta.Name, Receive_sta.Name)
    CalculateLagTime(SeisData, Para, Flag, Source_sta.Name, Receive_sta.Name)
