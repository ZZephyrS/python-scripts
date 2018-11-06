import os
import sys
import mds
import datetime

import numpy as np
import matplotlib as matp
import scipy.signal as sig
import matplotlib.colors as col
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as dates

from scipy.io import loadmat
from scipy import fftpack
from scipy.signal import periodogram,detrend
from time import clock
from mpl_toolkits.basemap import Basemap
from datetime import date
from matplotlib import dates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter


path_5day    = '/data/soccom/SO3/optim2012/ITERATION105/OUTPUT_DICbdgt'
plotdir      = '/data/irosso/plots/BSOSE'
SOSEdir      = '/data/soccom'

grid3        = 'GRID_3'
griddir      = os.path.join(SOSEdir,grid3)
grid_file    = os.path.join(griddir, 'grid.mat')


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
filename1 = 'diag_bgc'        #'DIAG_BGC/diag_bgc'; % TRAC01 TRAC02 TRAC03 TRAC04 TRAC05 TRAC06 BLGPH3D BLGOMAR TRAC07 TRAC08  

filename2 = 'diag_state'      #now they are 5day avg %'DIAG_STATE/diag_state'; % THETA, SALT, UVEL, VVEL, WVEL, DRHODR (these are daily avg)

filename3 = 'diag_surf'       #'DIAG_SURF/diag_surf'; % ETAN BLGPCO2 SIarea SIheff PHIBOT 

filename4 = 'diag_dic_budget' #'ADVxTr01' 'ADVyTr01' 'ADVrTr01' 'DFxETr01' 'DFyETr01' 'DFrITr01' 'UTRAC01 ' 'VTRAC01 ' 'WTRAC01 ' 'BLGBIOA ' ‘ForcTr01'
#'DIAG_DIC_BUDGET/diag_dic_budget'; % ADVxTr01 ADVyTr01 ADVrTr01 DFxETr01 DFyETr01 DFrITr01 UTRAC01 VTRAC01 WTRAC01 ForcTr01

filename5 = 'diag_bgc_snaps'  #'TRAC01  ' 'TRAC02  ' 'TRAC03  ' 'TRAC04  ' 'TRAC05  ' ‘TRAC06  ' 
#forward:'DIAG_DIC_SNAPS/diag_dic_snaps'; % TRAC01 snapshots every 5 days

filename6 = 'diag_airsea'     #'TFLUX   ' 'SFLUX   ' 'BLGCFLX ' 'BLGOFLX ' 'oceTAUX ' 'oceTAUY ' 
#forward: 'DIAG_AIRSEA/diag_airsea'; % TFLUX SFLUX BLGCFLX BLGOFLX Add2EmP

#filename1 = 'DIAG_BGC/diag_bgc' # TRAC01 TRAC02 TRAC03 TRAC04 TRAC05 TRAC06 BLGPH3D BLGOMAR TRAC07 TRAC08  
#filename2 = 'DIAG_STATE/diag_state' # THETA, SALT, UVEL, VVEL, WVEL, DRHODR (these are daily avg)
#filename3 = 'DIAG_SURF/diag_surf' # ETAN BLGPCO2 SIarea SIheff PHIBOT 
#filename4 = 'DIAG_DIC_BUDGET/diag_dic_budget' # ADVxTr01 ADVyTr01 ADVrTr01 DFxETr01 DFyETr01 DFrITr01 UTRAC01 VTRAC01 WTRAC01 ForcTr01
#filename5 = 'DIAG_DIC_SNAPS/diag_dic_snaps' # TRAC01 snapshots every 5 days
#filename6 = 'DIAG_AIRSEA/diag_airsea' # TFLUX SFLUX BLGCFLX BLGOFLX Add2EmP
#filename7 = 'DiagBIOA/diag_dic_BIOA' # BLGBIOA

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# domain
DXC          = loadmat(grid_file)['DXC'].transpose()
DYC          = loadmat(grid_file)['DYC'].transpose()
XC           = loadmat(grid_file)['XC'].transpose()
YC           = loadmat(grid_file)['YC'].transpose()
bathy        = loadmat(grid_file)['Depth'].transpose()

yy           = 300#-2*3
lenx         = len(XC[0,:])
leny         = len(YC[:yy,0])

lon_0        = XC[0,0]

TimeStep     = np.array(range(0,87600+120,120))

DRF          = loadmat(grid_file)['DRF'].transpose()
RF           = loadmat(grid_file)['RF'].transpose()
RF           = RF[0,:]
zlev         = 32
zstar        = -RF[zlev]

DRF1         = np.zeros([zlev,leny,lenx])
for kk in range(zlev):
    DRF1[kk,...]=DRF[0,kk]

# bathymetry contour levels
vB      = np.linspace(0,3000,6) 
v0      = np.linspace(0,0,1)
# sea ice contour level
vSI     = np.linspace(0.15,0.15,1)
# flx (-0.3,0.5 * 1.e-7)
map_plot= 'spectral'

# field to plot
field   = 2
if field == 0:
   bb_lab    = 'W m$^{-2}$'                 # Heat flux
   bb_title  = 'Surface heat flux'
   c1        = -200
   c2        = 250
   fact      = 1.
   fileName  = filename6
   numFF     = 0
   ll        = 0
   fname     = 'HFlx_surf'
elif field == 1:
   bb_lab    = r'mol C m$^{-2}$ y$^{-1}$'   # Air-sea C flux
   bb_title  = 'Tendency of surface CO$_2$ flux'
   c1        = -1 
   c2        = 2
   fact      = 365.242199*86400.
   fileName  = filename6
   numFF     = 2
   ll        = 0
   fname     = 'CO2_flx_surf'
elif field == 2:
   bb_lab    = r'mol C m$^{-3}$'               # DIC  (range of DIC between surf and bottom should be between 2-2.2 mol / m3 = 2 mmol / L = 2000 micro mol / L)
   bb_title  = '650 m average DIC'
   c1        = 2#2.15
   c2        = 2.5#2.22 #2.195
   fact      = 1.
   fileName  = filename1
   numFF     = 0
   ll        = 0#range(0,zlev)
   fname     = 'DIC_avg650m'
elif field == 3:
   bb_lab    = r'mol C y$^{-1}$'            # BIO, SURF
   bb_title  = '650 m average biology activity'
   c1        = -0.015
   c2        = 0.008
   fact      = 365.242199*86400.
   fileName  = filename4
   numFF     = 9
   ll        = range(0,zlev)
   fname     = 'BIO_avg650m'
elif field == 4:
   bb_lab    = 'm d$^{-1}$'                 # vertical velocity
   bb_title  = '650 m average vertical velocity'
   c1        = -5
   c2        = 5
   fact      = 86400.
   fileName  = filename2
   numFF     = 4
   ll        = range(0,zlev)
   fname     = 'Wvel_avg650m'
   map_plot  = 'RdBu_r'
   

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
latcorners   = YC[:yy,0]
loncorners   = XC[0,:]
lons, lats   = loncorners, latcorners

latcorners1  = YC[:-10,0]
lons1, lats1 = loncorners, latcorners1

SI           = np.zeros((364,leny,lenx),'>f4')
tmpff        = np.zeros((364,leny,lenx),'>f4')
for tt in range(365):
    # load the field
    print tt
    if ll!=0:
        tmp      = mds.rdmds(os.path.join(path_5day,fileName),TimeStep[tt],rec=numFF,lev=0)#ll)
        tmp2     = tmp[:yy,:]
        # average over depth
        #tmp2     = np.cumsum(DRF1*tmp[:,:yy,:],axis=0)[-1]/zstar #np.sum(DRF1[:,0,0])
        #tmp2     = np.nanmean(tmp[:,:yy,:],axis=0)
    else:
        tmp      = mds.rdmds(os.path.join(path_5day,fileName),TimeStep[tt],rec=numFF)
        tmp2     = tmp[:yy,:]
    tmpff[tt,...]      = tmp2
    del tmp
    
    # load sea ice
    tmpSI        = mds.rdmds(os.path.join(path_5day,filename3),TimeStep[tt],rec=2)
    SI[tt,...]= tmpSI[:yy,:]
        
del tmp2, tmpSI

tmpff_msk = np.ma.masked_equal(tmpff,0.)
del tmpff

"""
# DIC has an unrealistic trend: we need to remove it before computing the mean
if field == 2:
    # detrend and compute climatology (avg over 8 years) of the field: the detrend function removes also the mean
    tmpff2  = sig.detrend(tmpff_msk,axis=0,type='linear')
    # add the mean back, to not have the field centered in zero (which for DIC doesn't make any sense..)
    dumBar  = np.nanmean(tmpff_msk)
    tmpff2  = tmpff2 + dumBar
else:
"""
tmpff2  = tmpff_msk
del tmpff_msk

# climatology (avg over 5 years)
ff_clim = np.zeros((73,leny,lenx),'>f4')
SI_clim = np.zeros((73,leny,lenx),'>f4')
for i in range(73):
    ff_clim[i,...] = np.nanmean(tmpff2[i::73,...],axis=0)
    SI_clim[i,...] = np.nanmean(SI[i::73,...],axis=0)
#del tmpff2, SI

print YC[-1,0]
# plot figure
fig     = plt.figure(10,figsize=(10,15))
season  = ['Summer (DJF)','Autumn (MAM)','Winter (JJA)','Spring (SON)']
for iFig in range(1,5):
    if iFig == 1:
        # summer (jan + feb + dec)
        #ff_seas = np.sum(tmpff2[:12,...],axis=0) + np.sum(tmpff2[-6:,...],axis=0)
        #SI_seas = np.sum(SI[:12,...],axis=0) + np.sum(SI[-6:,...],axis=0)
        ff_seas = np.nansum(ff_clim[:12,...],axis=0) + np.nansum(ff_clim[-6:,...],axis=0)
        SI_seas = np.nansum(SI_clim[:12,...],axis=0) + np.nansum(SI_clim[-6:,...],axis=0)
    else:
        #ff_seas = np.sum(tmpff2[12+18*(iFig-2):12+18*(iFig-1),...],axis=0)
        #SI_seas = np.sum(SI[12+18*(iFig-2):12+18*(iFig-1),...],axis=0)
        ff_seas = np.nansum(ff_clim[12+18*(iFig-2):12+18*(iFig-1),...],axis=0)
        SI_seas = np.nansum(SI_clim[12+18*(iFig-2):12+18*(iFig-1),...],axis=0)
    ff_seas = ff_seas/(18.) 
    SI_seas = SI_seas/(18.) 
    print ff_seas.min(), ff_seas.max()
    ff_msk  = np.ma.masked_array(ff_seas,bathy[:yy,:]==0.)
    del ff_seas
    #plot fig
    ax      = fig.add_subplot(2,2,iFig)
    #m       = Basemap(projection='spstere',boundinglat=-40,lon_0=90)
    m          = Basemap(projection='ortho', lat_0=-90, lon_0=lon_0)
    x, y    = m(*np.meshgrid(lons, lats))
    xb, yb  = m(*np.meshgrid(lons1, lats1))
    levels  = np.linspace(c1,c2,70)
    im      = m.contourf(x, y, ff_msk*fact, cmap='spectral', levels=levels, extend='both')  
    #if field == 0:
    #    m.contourf(x, y, ff_msk*fact, cmap='spectral_r', levels=np.linspace(-450,c1,2))
    if field in [0,1,3]:
        m.contour(x, y, ff_msk*fact, v0, colors='k')
    m.contour(xb, yb, bathy[:-10,:],vB,colors='gray',linewidts=0.5)
    m.contour(x, y, SI_seas,vSI,colors='k',linewidths=2,alpha=0.8)
    m.contour(xb, yb, bathy[:-10,:],v0,colors='gray',linewidts=2)
    m.drawparallels(np.arange(-90, YC[-1,0], 20))#,labels=[True, False, True, True])
    if iFig == 1:
        labmer = [True, False, True, True]
    elif iFig in [2,3]:
        labmer = [False, False, True, True]
    else:
        labmer = [False, True, True, True]
    print iFig, labmer
    m.drawmeridians(np.arange(0, 360, 30),labels=[False,False,False,False])#labmer)
    cax  = fig.add_axes([0.2, 0.05, 0.6, 0.02])
    cbar = fig.colorbar(im, cax=cax,orientation='horizontal')
		
    cbar.set_ticks((np.linspace(c1,c2,5)))
    ax.set_title('%s\n' %(season[iFig-1]),fontsize=16)
    #plt.suptitle('%s\n' %(bb_title),fontsize=16)
    cbar.ax.set_xlabel('%s' %(bb_lab), rotation='horizontal',fontsize=14)
    iFig += 1

plt.figure(10,figsize=(10,15))
outfile = os.path.join(plotdir,'%s_seas_clim.png' %(fname))
print outfile
plt.savefig(outfile, bbox_inches='tight',dpi=500)

plt.show()
sys.exit()

# plot Takahashi CO2 flux
takafile= os.path.join(SOSEdir,'data/Takahashi_CO2/CO2.txt')
f       = open(takafile, 'r')
lonCO2  = []
latCO2  = []
takaCO2 = []
for line in f.readlines():
    if not line.strip().startswith('LAT'):
        data = line.split()
        if float(data[0])<=-30.:
            lonCO2  = np.append(lonCO2,float(data[1]))
            latCO2  = np.append(latCO2,float(data[0]))
            takaCO2 = np.append(takaCO2,float(data[5]))

f.close()

xnew    = 0.
xnew    = np.append(xnew,np.arange(2.5,362.5,5.))
xnew    = np.append(xnew,360.)
ynew    = np.arange(-76.,-30.,4.)

CO2Taknew= np.zeros((len(ynew), len(xnew)))
for ii in range(len(xnew)):
    for jj in range(len(ynew)):
        for kk in range(len(lonCO2)):
            if xnew[ii] == lonCO2[kk] and ynew[jj] == latCO2[kk]:
                CO2Taknew[jj,ii] = takaCO2[kk]

CO2Taknew[:,0]  = (CO2Taknew[:,1]+CO2Taknew[:,-2])/2.
CO2Taknew[:,-1] = CO2Taknew[:,0]
CO2_msk         = np.ma.masked_equal(CO2Taknew,0.)

"""
jjmax  = np.min(np.where(lats>=np.max(latCO2)))-1
jjmin  = np.min(np.where(lats>=np.min(latCO2)))-1
ymax   = lats[jjmax]
ymin   = lats[jjmin]

ax_ext = [lons[0], lons[-1], ymin, ymax]
plt.figure()
plt.contour(bathy[jjmin:jjmax,:], vB, origin='lower', extent=ax_ext, colors='gray')
plt.contourf(CO2TakInt,levels,origin='lower',extent=ax_ext)
#plt.scatter(lonCO2, latCO2, c=takaCO2, cmap=map_plot, marker='D',s=100, edgecolors='none')
plt.xlim(np.min(lonCO2),np.max(lonCO2))
plt.ylim(np.min(latCO2),np.max(latCO2))
"""

# plot flux
fig     = plt.figure(20,figsize=(8,8))
ax      = fig.add_subplot(111)
m       = Basemap(projection='spstere',boundinglat=-40,lon_0=90)
xCO2, yCO2 = m(*np.meshgrid(xnew, ynew))
#xCO2, yCO2 = m(*np.meshgrid(lonCO2, latCO2))
c1      = -3
c2      = 2
levels  = np.linspace(c1,c2,100)
im      = m.contourf(xCO2, yCO2, CO2_msk, cmap=map_plot, levels=levels)#, extend='max')
#im      = m.plot(lonCO2, latCO2, 'ro')#c=takaCO2, cmap=map_plot, marker='o',s=100, edgecolors='none')
x, y    = m(*np.meshgrid(lons, lats))
m.contour(x, y, bathy[:yy,:],vB,colors='gray')
m.drawparallels(np.arange(-90, YC[-1,0], 20))#,labels=[True, False, True, True])
m.drawmeridians(np.arange(0, 360, 30),labels=[True,True,True,True])
cax  = fig.add_axes([0.2, 0.05, 0.6, 0.02])
cbar = fig.colorbar(im, cax=cax,orientation='horizontal')
cbar.set_ticks((np.linspace(c1,1.5,5)))
#ax.set_title('Takahashi et al. (2009) %s\n' %(season[iFig-1]))
plt.suptitle('Takahashi et al. (2009) CO$_2$ flux \n',fontsize=16)
cbar.ax.set_xlabel('mol C m$^{-2}$ y$^{-1}$', rotation='horizontal',fontsize=12)


plt.figure(20,figsize=(8,8))
outfile = os.path.join(plotdir,'Takahashi_2009_CO2Flx_clim.png')
print outfile
plt.savefig(outfile, bbox_inches='tight',dpi=500)

plt.show()

