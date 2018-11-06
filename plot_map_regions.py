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

from matplotlib.lines import Line2D
from scipy.io import loadmat
from numpy import linalg as la
from scipy import fftpack
from scipy.signal import periodogram,detrend
from time import clock
#from mpl_toolkits.basemap import Basemap
from datetime import date
from matplotlib import dates
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

plt_net      = True
net_fig      = True
onlyReg      = False

HOMEdir       '/data/irosso'
plotdir      = os.path.join(HOMEdir,'plots')
SOSEdir      = '/data/soccom'

grid3        = 'GRID_3'
griddir      = os.path.join(SOSEdir,grid3)
grid_file    = os.path.join(griddir, 'grid.mat')

local        = '/data/irosso/data/BSOSE'

DXC          = loadmat(grid_file)['DXC'].transpose()
DYC          = loadmat(grid_file)['DYC'].transpose()
XC           = loadmat(grid_file)['XC'].transpose()
YC           = loadmat(grid_file)['YC'].transpose()
bathy        = loadmat(grid_file)['Depth'].transpose()
DRF          = loadmat(grid_file)['DRF'][0]

x1           = None
x2           = 1080
y1           = None
y2           = 300

XC           = XC[y1:y2,x1:x2]
YC           = YC[y1:y2,x1:x2]
DXC          = DXC[y1:y2,x1:x2]
DYC          = DYC[y1:y2,x1:x2]
bathy        = bathy[y1:y2,x1:x2]

lenx         = len(XC[0,:])
leny         = len(YC[:300,:])
nx           = len(XC[0,x1:x2])
ny           = len(YC[y1:y2,0])
print YC[0,0],YC[-1,0]

# land indexes
[topo_y, topo_x] = np.where(bathy==0.)

v0           = np.linspace(0,0,1)

# text
xtt          = [75,104,170,217,291,305,192,10,319,184,105,163,39]
ytt          = [-49,-58,-59,-62,-59,-68,-70,-75,-32,-40,-43,-30,-32]
text         = ['Kerguelen\nPlateau', 'South East\nIndian Ridge',
                'Macquarie\nRidge', 'Pacific-Antarctic\nRidge',
                'Drake Passage','Weddell Gyre','Ross Gyre',
                'Antarctica', 'South\nAmerica','New Zealand',
                'Tasmania','Australia','South\nAfrica']

# domain
dom          = ['ACC','NorthACC','WS']#,RS'SouthACC']
col          = ['k','g','r','c','m','b']
colReg       = ['r','b','k']
pattern      = [None,'|','x']

tit          = ['Subtropical Region', 'ACC', 'Antarctic Region']

xlabels      = ['DIC tendency', 'diffusion', 'biology', 'dilution', 'air-sea CO$_2$ flux', 'advection','horizontal advection','vertical advection']

NET_DIC      = np.zeros((3,8),'>f4')


NET_DIC[0,:] = [0.177379, 0.077144, -0.178062, 0.131912, 0.153438, -0.084196, 1.062427, -1.060959]
NET_DIC[1,:] = [0.026254, 0.025952, -0.194623, -0.046864, 0.030353, 0.185483, -2.402074, 2.336971]
NET_DIC[2,:] = [0.063670, 0.000456, -0.112979, -0.017764, 0.057283, 0.136218, -5.774339, 5.496866]

"""
# SO3 
NET_DIC[0,:] = [0.157585,0.079130,-0.084367,0.095727,0.198832,-0.210869,2.569190,-2.785992]
NET_DIC[1,:] = [0.279157,0.021293,-0.108969,-0.156294,0.112788,0.389047,-5.269756,5.650727]
NET_DIC[2,:] = [0.106297,-0.001838,-0.055708,-0.039740,0.095989,0.109432,-4.703841,4.806846]
"""
print dom 
if onlyReg:
    fig     = plt.figure(1,frameon=False,figsize=(15,5))
else:
    fig     = plt.figure(1,figsize=(15,10))
ax3     = plt.subplot(111)
if onlyReg:
    ax3.axis('off')
elif not onlyReg :
    plt.pcolormesh(XC[0,:],YC[:,0],bathy,cmap='Greys',alpha=0.7)
plt.contour(XC[0,:],YC[:,0],bathy,v0,colors='k',origin='lower')

for domain in dom:
    if domain == 'WS':
        file         = os.path.join(local,'WScoord.data')
        Rcoord       = np.fromfile(file,'int32')
        Rcoord       = np.reshape(Rcoord,(leny,lenx))
        cmap         = 'Greens'
    elif domain == 'RS':
        file         = os.path.join(local,'RScoord.data')
        Rcoord       = np.fromfile(file,'int32')
        Rcoord       = np.reshape(Rcoord,(leny,lenx))
        cmap         = 'Oranges'
    elif domain == 'ACC':
        file         = os.path.join(local,'ACCcoord.data')
        Rcoord       = np.fromfile(file,'int32')
        Rcoord       = np.reshape(Rcoord,(leny,lenx))
        cmap         = 'Blues'
    elif domain == 'NorthACC':
        file         = os.path.join(local,'Northcoord.data')
        Rcoord       = np.fromfile(file,'int32')
        Rcoord       = np.reshape(Rcoord,(leny,lenx))
        cmap         = 'Reds'
    elif domain == 'SouthACC':
        file         = os.path.join(local,'Southcoord.data')
        Rcoord       = np.fromfile(file,'int32')
        Rcoord       = np.reshape(Rcoord,(leny,lenx))
        cmap         = 'Greys'            
    xcoor   = np.where(Rcoord==0)[1]
    ycoor   = np.where(Rcoord==0)[0]
    
    bathy2  = bathy.copy()
    bathy2[ycoor,xcoor]  = 0.
    bathy2[topo_y,topo_x]= 0.
    bathymsk = np.ma.masked_equal(bathy2,0.)
    plt.pcolormesh(XC[0,:],YC[:,0],bathymsk,cmap=cmap)

    plt.xlim(XC[0,0],XC[0,-1])
    plt.ylim(YC[0,0],YC[-1,0])
    if not plt_net:
        for itt in range(13):
            if itt == 7:
                cc = 'k'
            else:
                cc = 'w'
            if onlyReg:
                ax3.text(xtt[itt], ytt[itt], '%s' %(text[itt]),fontsize=18, color='w',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.8))
            else:
                ax3.text(xtt[itt], ytt[itt], '%s' %(text[itt]),fontsize=20, color=cc,bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.2))
    ii1      = np.min(np.where(XC[0,:]>=30.))
    ii2      = np.min(np.where(XC[0,:]>=330.))
    xtix     = np.linspace(XC[0,ii1],XC[0,ii2],11)
    xtix_l   = [(str(int(f)) + 'E') for f in xtix[0:5]] + ['180'] + [(str(int(f)) + 'W') for f in xtix[4::-1]]

    jj1      = np.min(np.where(YC[:,0]>=-30.))
    jj2      = np.min(np.where(YC[:,0]>=-70.))
    ytix     = np.linspace(YC[jj2,0],YC[jj1,0],5)
    ytix_l   = [(str(int(round(-f))) + 'S') for f in ytix]
    
    if not onlyReg:
        ax3.set_xticks(xtix)#np.linspace(0,leny-1,13)[1:-1])
        ax3.set_xticklabels(xtix_l)
        ax3.set_yticks(ytix)#np.linspace(0,leny-1,6)[1:-1])
        ax3.set_yticklabels(ytix_l)

if plt_net:
    if net_fig:
        fig      = plt.figure(2,figsize=(17,5))
        ylim     = 0.25
        xPos     = [0.3,0.57,0.85]
    else:
        yPos     = [0.64,0.38,0.12]
        ylim     = 0.5

    ylim2    = 6
    barwidth = 0.15 #2
    xdata    = np.linspace(0,1,6)
    #xdata    = np.arange(1,27,5)
    ydata    = np.linspace(-ylim,ylim,5) 
    ydata2   = np.linspace(-ylim2,ylim2,3) 
    xx       = [0,1,2]

    for rr in range(3):
        # white square around
        if not net_fig:
            axx      = fig.add_axes([0.48, yPos[rr], 0.27, 0.25])   
            axx.axes.get_xaxis().set_visible(False)
            axx.axes.get_yaxis().set_visible(False)
            axx.set_frame_on(True)
            axx.patch.set_edgecolor(colReg[rr])
            axx.patch.set_alpha(0.7)
            # bars
            ax4      = fig.add_axes([0.55, yPos[rr]+0.02, 0.2, 0.2 ])
            ax4.set_frame_on(True)
            ax4.get_yaxis().tick_left()
            ax4.axes.get_yaxis().set_visible(True)
            ax4.axes.get_xaxis().set_visible(False)
            ax4.add_artist(Line2D((-0.2,-0.2), (-ylim-0.001, ylim), color='black', linewidth=4))
            ax5      = fig.add_axes([0.7, yPos[rr]+0.02, 0.03, 0.1 ])
        else:
            ax4      = fig.add_subplot(1,3,rr+1)
            plt.title('%s' %tit[rr], fontsize=14)
            ax5      = fig.add_axes([xPos[rr], 0.62, 0.045, 0.25 ]) #y=0.62
            ax4.get_yaxis().tick_left()
            ax4.axes.get_yaxis().set_visible(True)
            ax4.axes.get_xaxis().set_visible(False)
            
        # hor and ver ADV
        ax5.set_frame_on(True)
        ax5.get_yaxis().tick_left()
        ax5.axes.get_yaxis().set_visible(True)
        ax5.axes.get_xaxis().set_visible(False)
        ax5.add_artist(Line2D((0,0), (-ylim2-0.001, ylim2), color='black', linewidth=1))
        ax5.bar(0.5, NET_DIC[rr,6], facecolor='w',alpha=0.8,edgecolor='k', width=0.8,hatch='-',align='center',label='%s' %(xlabels[6]),linewidth=4)
        ax5.bar(1.5, NET_DIC[rr,7], facecolor='w',alpha=0.8,edgecolor='k', width=0.8,hatch='|',align='center',label='%s' %(xlabels[7]),linewidth=4)
        ax5.grid('on')
        if rr == 0:
            ax5.set_ylabel('[Pg C y$^{-1}$]',fontsize=14)
        ax5.set_xticks(xdata+1)
        ax5.set_xticklabels(xlabels)
        ax5.set_yticks(ydata2)

        for nn in range(6):
            bb = ax4.bar([xdata[nn]], NET_DIC[rr,nn], facecolor=col[nn],alpha=1, edgecolor=col[nn],width=barwidth, align='center',label='%s' %(xlabels[nn]),linewidth=2)# colReg[rr]
            #bb = ax4.bar([xdata[nn]+xx[rr]], NET_DIC[rr,nn], edgecolor=colReg[rr],facecolor=col[nn],alpha=0.5, width=barwidth, align='center',label='%s' %(xlabels[nn]),hatch=pattern[rr],linewidth=2)

        ax4.set_xticks(xdata+1)
        ax4.set_xticklabels(xlabels)
        ax4.set_yticks(ydata)
        ax4.grid('on')
        #ax4.patch.set_visible(False)
        if rr == 0 :
            ax4.set_ylabel('[Pg C y$^{-1}$]',fontsize=16)

        if rr == 2:
            hh, ll   = ax4.get_legend_handles_labels()
            print len(hh),len(ll)
            if net_fig:
                ax6      = fig.add_axes([0.8, 0.07, 0.05, 0.05 ])
                aa       = 0
            else:
                ax6      = fig.add_axes([0.2, 0.09, 0.05, 0.05 ])
                aa       = 0.5
            ax6.set_frame_on(False)
            ax6.axes.get_xaxis().set_visible(False)
            ax6.axes.get_yaxis().set_visible(False)
            leg=plt.legend(hh[:6],ll[:6],loc=8,fancybox=True,framealpha=aa,fontsize=14)
            leg.get_frame().set_linewidth(0.0)

            hh, ll   = ax5.get_legend_handles_labels()
            print len(hh),len(ll)
            if net_fig:
                ax7      = fig.add_axes([0.75, 0.7, 0.02, 0.04 ])#([0.85, 0.4, 0.02, 0.02 ])
                aa       = 0
            else:
                ax7      = fig.add_axes([0.4, 0.1, 0.02, 0.02 ])
                aa       = 0.5
            ax7.set_frame_on(False)
            ax7.axes.get_xaxis().set_visible(False)
            ax7.axes.get_yaxis().set_visible(False)
            leg=plt.legend(hh[:6],ll[:6],loc=8,fancybox=True,framealpha=aa,fontsize=14)
            leg.get_frame().set_linewidth(0.0)

plt.figure(1)
outfile = os.path.join(plotdir,'map_bathy_regions.png')
if plt_net:
    plt.figure(2)
    outfile = os.path.join(plotdir,'net_terms.png')

print outfile   
plt.savefig(outfile, bbox_inches='tight',dpi=200)
plt.show()
