import glob,os,sys,math
import numpy as np
import matplotlib as matp
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# parameters
case      = 'AABW_BACK'

ylimN     = -25.
yIni      = -25.
yFin      = -80.

folder    = 'glued'
plotdir   = 'plots'

# 3D files
file_lon3D= os.path.join(folder,'lon3D_%s.data' %(case))
file_lat3D= os.path.join(folder,'lat3D_%s.data' %(case))
file_dep3D= os.path.join(folder,'dep3D_%s.data' %(case))
#file_x3D  = os.path.join(folder,'x3D_%s.data' %(case))
#file_y3D  = os.path.join(folder,'y3D_%s.data' %(case))
#file_z3D  = os.path.join(folder,'z3D_%s.data' %(case))

#you can glue one case or multi-cases at the same time, 'DIMES_0001' etc. are casenames
casenames = '%s' %(case) #%04i' %ii for ii in range(1,10,1)] 

#specify particle numbers and case numbers (the NPP value in the namelist) here:
if 'AABW' in case:
    npts = 9443  # P18: 70; P17: 92; P17E: 35
npp       = 60

files     = [f for f in os.listdir(folder) if f.startswith('%s' %(casenames)) and 'pickup' not in f]# and '0000046081' in f] #%s_%04i.XYZ.0000046081.0002142897.data' %(casenames,ii) for ii in range(1,10,1)] #put a glued XYZ data filename 
files.sort()

res       = 6
grid      = 'SO6_Iter100'

cmap      = plt.cm.jet
cmaplist  = [cmap(i) for i in range(cmap.N)]
col       = ['k','r','b','g','c','m','orange','purple','grey']
xc        = 0.2
yc        = 0.16


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_grid(res,griddir):
    from scipy.io import loadmat
    grid_file    = os.path.join(griddir, 'grid.mat')
    if res in [3,6]:
        read_grid.XC = loadmat(grid_file)['XC'].transpose()
        read_grid.YC = loadmat(grid_file)['YC'].transpose()
        read_grid.RC = loadmat(grid_file)['RC'].transpose()
        read_grid.RF = loadmat(grid_file)['RF'].transpose()
        read_grid.DRC= loadmat(grid_file)['DRC'].transpose()
        read_grid.DRF= loadmat(grid_file)['DRF'].transpose()
        read_grid.depth= loadmat(grid_file)['Depth'].transpose()
       
    else:
        grid         = hdf.File(grid_file,'r')
        read_grid.XC = grid['XC'][:] 
        read_grid.YC = grid['YC'][:]
        read_grid.RC = grid['RC'][:]
        read_grid.depth= grid['Depth'][:]

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# load grid
read_grid(res,grid)
XC         = read_grid.XC
YC         = read_grid.YC
#RR         = read_grid.RC
#RR         = RR[0,:]
#DR         = read_grid.DRC
#DR         = DR[0,:]
bathy      = read_grid.depth

Lx         = len(XC[0,:])
Ly         = len(YC[:,0])
# bathymetry contour levels
vB         = np.linspace(1000,5000,5) 
v0         = np.linspace(0,0,1)

# create figure and axes instances
cgr        = matp.cm.Greys_r
cgr.set_bad(color='k')

xRoll      = 0
x1         = np.min(np.where(XC[0,:]>=xRoll))
XCr        = np.roll(XC,Lx-x1,axis=1)
XCr[np.where(XCr<xRoll)] = XCr[np.where(XCr<xRoll)]+360.
bathyr     = np.roll(bathy,Lx-x1,axis=1)
bathy_m    = np.ma.masked_equal(bathyr,0.)

"""
fig      = plt.figure(1,figsize=(15,10))
ax       = fig.add_subplot(111)
im       = ax.pcolor(XCr[0,:],YC[:,0],bathyr,cmap=cgr)#,vmin=0, vmax=0.2)
plt.contour(XCr[0,:],YC[:,0],bathyr,v0,colors='k')
plt.scatter([310],[-60], s=100,c='m')
plt.show()
sys.exit()
"""

xmin       = 0.
xmax       = 359.
yy1        = 0
yy2        = np.min(np.where(YC[:,0]>=-25.))
xx1        = np.min(np.where(XCr[0,:]>=xmin))
xx2        = np.min(np.where(XCr[0,:]>=xmax))

# load trajectory files
lon3D = np.fromfile(file_lon3D,'>f4')
lon3D = np.reshape(lon3D,(npts,2000,npp))
lat3D = np.fromfile(file_lat3D,'>f4')
lat3D = np.reshape(lat3D,(npts,2000,npp))
dep3D = np.fromfile(file_dep3D,'>f4')
dep3D = np.reshape(dep3D,(npts,2000,npp))
lon0   = lon3D[:,0,0]
lat0   = lat3D[:,0,0]

"""
fig      = plt.figure(1,figsize=(15,10))
ax       = fig.add_subplot(111)
im       = ax.pcolor(XCr[0,:],YC[:,0],bathyr,cmap=cgr)#,vmin=0, vmax=0.2)
plt.contour(XCr[0,:],YC[:,0],bathyr,v0,colors='k')
#plt.scatter([310],[-60], s=100,c='m')
#for ii in range(lon3D.shape[2]):
#    plt.plot(lon3D[:,:500:10,ii],lat3D[:,:500:10,ii],linewidth=0.2)
plt.scatter(lon0,lat0, s=50,c='m',zorder=10000000)

outfile = os.path.join(plotdir,'prova.png' )
plt.savefig(outfile, bbox_inches='tight',dpi=200)#rasterized=True)
print "\nPlots saved in: ", outfile

sys.exit()
"""


print '\nlon, lat and depth loaded..'
"""
dep3D = np.fromfile(file_dep3D,'>f4')
dep3D = np.reshape(dep3D,(npts,2000,npp))
x3D     = np.fromfile(file_x3D,'>f4')
x3D     = np.reshape(x3D,(npts,2000,npp))
y3D     = np.fromfile(file_y3D,'>f4')
y3D     = np.reshape(y3D,(npts,2000,npp))
z3D     = np.fromfile(file_z3D,'>f4')
z3D     = np.reshape(z3D,(npts,2000,npp))
"""

#prob2D  = np.zeros((Ly,Lx),'>f4')
Source    = []
countZini = 0
countRS   = 0
countWS   = 0

# initial depths are: 1971.48, 2512.02, 3000, 3500, 4000, 4500, 5000
zMax1     = 2990
zMax2     = 3010
tMax      = 1108
zzz       = np.where(dep3D[:,0,:]<=zMax2)
indZ      = zip(zzz[0],zzz[1])
print zzz
#zRel      = dep3D[indZ[0][0],0,indZ[0][1]]
print '\nStarting the count..'


for ii in range(len(indZ)):
   ind1 = indZ[ii][0]
   ind2 = indZ[ii][1]
   if lon3D[ind1,0,ind2]<150.: # this should then change, I don't know why there's another cluster of points at lon >150...
	   if dep3D[ind1,0,ind2] >=zMax1:
		   zRel      = dep3D[ind1,0,ind2]
		   countZini = countZini + 1
		   for tt in range(tMax):
				if ~np.isnan(lon3D[ind1,tt,ind2]) and lat3D[ind1,tt,ind2]>=YC[0,0] and lat3D[ind1,tt,ind2]<=-65. and lon3D[ind1,tt,ind2]>=160. and lon3D[ind1,tt,ind2]<=210.:
				   countRS += 1
				   print 'qui RS'
				   Source = np.append(Source,[ind1,ind2])
				   for t2 in range(tt,tMax):
					   if ~np.isnan(lon3D[ind1,t2,ind2]) and lat3D[ind1,t2,ind2]>=YC[0,0] and lat3D[ind1,t2,ind2]<=-60. and lon3D[ind1,t2,ind2]>=310.:
						   countWS += 1
						   Source = np.append(Source,[ind1,ind2])
						   print 'qui WS'
						   break
				   break
			
				elif ~np.isnan(lon3D[ind1,tt,ind2]) and lat3D[ind1,tt,ind2]>=YC[0,0] and lat3D[ind1,tt,ind2]<=-60. and lon3D[ind1,tt,ind2]>=310.:
				   countWS += 1
				   Source = np.append(Source,[ind1,ind2])
				   print 'qui WS2'
				   break
			
				elif np.isnan(lon3D[ind1,tt,ind2]):
				   break
			   
"""
for ii in range(npts):
    for nn in range(npp):
        if dep3D[ii,0,nn] <=2000.:
            if ii == 0 and nn == 0:
                zRel = dep3D[ii,0,nn]
            countZini += 1            
            # look at probability after 5years (365*5/10days)
            for tt in range(1108):
                #print lat3D[ii,tt,nn], YC[0,0],lon3D[ii,tt,nn]
                if ~np.isnan(lon3D[ii,tt,nn]) and lat3D[ii,tt,nn]>=YC[0,0] and lat3D[ii,tt,nn]<=-65. and lon3D[ii,tt,nn]>=160. and lon3D[ii,tt,nn]<=210.:
                #if ~np.isnan(x3D[ii,tt,nn]) and y3D[ii,tt,nn]<Ly and x3D[ii,tt,nn]<Lx:
                    countRS += 1
                    Source = np.append(Source,[ii,nn])
                    break
                    #prob2D[y3D[ii,tt,nn],x3D[ii,tt,nn]] = prob2D[y3D[ii,tt,nn],x3D[ii,tt,nn]] + 1
                elif ~np.isnan(lon3D[ii,tt,nn]) and lat3D[ii,tt,nn]>=YC[0,0] and lat3D[ii,tt,nn]<=-60. and lon3D[ii,tt,nn]>=290.:
                    countWS += 1
                    Source = np.append(Source,[ii,nn])
                    break
                elif np.isnan(lon3D[ii,tt,nn]):
                    break
"""
Source   = np.reshape(Source,[countWS+countRS,2])

print Source
print 'number of particles coming from the Weddell Sea = ', countWS
print 'number of particles coming from the Ross Sea = ', countRS
print 'Total number of particles released at -%02.f m = %i' %(zRel,countZini)
print 'Percentage of particles from Weddell Sea: ',float(countWS)/float(countZini)*100., '%' 
print 'Percentage of particles from Ross Sea: ',float(countRS)/float(countZini)*100., '%' 



fig      = plt.figure(1,figsize=(15,10))
ax       = fig.add_subplot(111)
im       = ax.pcolor(XCr[0,:],YC[:,0],bathyr,cmap=cgr)#,vmin=0, vmax=0.2)
plt.contour(XCr[0,:],YC[:,0],bathyr,v0,colors='k')
for jj in range(len(Source)):
    jj1 = int(Source[jj][0])
    jj2 = int(Source[jj][1])
    ax.plot(lon3D[jj1,:,jj2],lat3D[jj1,:,jj2],linewidth=1,alpha=0.8) #,color=cmaplist[iP2*cmap.N/nptsSouth]

#prob2D   = prob2D/float(count)*100
#prob2D_r = np.roll(prob2D,Lx-x1,axis=1)
#prob2D_m = np.ma.masked_equal(prob2D_r,0)
# save plot
outfile = os.path.join(plotdir,'Traj_AABW_WS_RS_%im.png' %(int(zRel)))
print "\nPlots saved in: ", outfile
plt.savefig(outfile, bbox_inches='tight',dpi=200)#rasterized=True)

plt.show()

sys.exit()


"""
# plot
fig      = plt.figure(ii+1,figsize=(15,10))
ax       = fig.add_subplot(111)
im       = ax.pcolor(XCr[0,:],YC[:,0],prob2D_m,cmap='GnBu',vmin=0, vmax=0.2)
plt.contour(XCr[0,:],YC[:,0],bathyr,v0,colors='k')

print np.min(prob2D), np.max(prob2D)

params = {'lines.linewidth': 4}
plt.rcParams.update(params)
plt.plot(lon_fi[:tmax],lat_fi[:tmax],'r',alpha=0.8,linewidth=4)
params = {'lines.linewidth': 0.5}
plt.rcParams.update(params)
plt.plot(lon_fi[:100],lat_fi[:100],'w',alpha=0.5,linewidth=0.5)
plt.scatter(lon_fi[0],lat_fi[0],s=300,color='k',marker="*",zorder=10000)

#print '\nCoordinates of SOCCOM float at 100 days: ',lon_fi[99],lat_fi[99]
#print 'Probability of SOCCOM_100dd :', prob2D_r[(lat_fi[99]+77.875)*6.,(lon_fi[99]-150.)*6.]

# plot Orsi's fronts
for ff in fronts:
    lon_FF = []
    lat_FF = []
    dataFile = os.path.join(frontdir, '%s.txt' %(ff))
    data     = open(dataFile, 'r')
    for line in data.readlines():
        if not line.strip().startswith('%'):
            coord  = line.split()
            f_lon  = float(coord[0])
            if f_lon <= 0.:
                f_lon = f_lon + 360.
            lon_FF = np.append(lon_FF, f_lon)
            lat_FF = np.append(lat_FF, float(coord[1]))
        else:
            if len(lon_FF)!= 0 : 
                # roll
                if np.array(np.where(lon_FF>=xRoll)).size != 0:
                    xx       = np.min(np.where(lon_FF>=xRoll))
                    lon_FF   = np.roll(lon_FF,len(lon_FF)-xx)
                    lat_FF   = np.roll(lat_FF,len(lat_FF)-xx)
                if np.array(np.where(lon_FF<xRoll)).size != 0:
                    lon_FF[np.where(lon_FF<xRoll)] = lon_FF[np.where(lon_FF<xRoll)]+360.
                # the stf makes a mess with the plotting.. need to split it
                if np.array(np.where(np.abs(np.diff(lon_FF))>30.)).size!=0:
                    indNaN = np.where(np.abs(np.diff(lon_FF))>30.)
                    lon_FF2 = np.insert(lon_FF,indNaN[0][:]+1,np.nan)
                    lat_FF2 = np.insert(lat_FF,indNaN[0][:]+1,np.nan)
                else:
                    lon_FF2 = lon_FF
                    lat_FF2 = lat_FF

                params = {'lines.linewidth': 2}
                plt.rcParams.update(params)
                ax.plot(lon_FF2, lat_FF2,'k-',alpha=0.2,linewidth=2)

                lon_FF = []
                lat_FF = []

    data.close()

# plot properties
plt.title('Probability of particle location after 5 years',fontsize=14)
cax  = fig.add_axes([xc, yc, 0.15, 0.02])
cbar = plt.colorbar(im,cax=cax,orientation='horizontal')
cbar.ax.set_xlabel('[%]', rotation='horizontal', fontsize=12, color='k')
cbar.set_ticks(np.linspace(0,0.2,5))
cbar.ax.xaxis.set_tick_params(color='k')
plt.setp(plt.getp(cbar.ax.axes, 'xticklabels'), color='k')

ax.set_xticks(xtix)
ax.set_xticklabels(xtix_l)    
ax.set_yticks(ytix)
ax.set_yticklabels(ytix_l)

ax.set_xlim(XC[0,0],XC[0,-1])  
ax.set_ylim(YC[0,0],YC[0,-1])

# save plot
outfile = os.path.join(plotdir,'prob_%s_5yy_2500m.pdf' %(case))
print "\nPlots saved in: ", outfile
plt.savefig(outfile, bbox_inches='tight',dpi=200)#rasterized=True)
#plt.show()
            
"""
