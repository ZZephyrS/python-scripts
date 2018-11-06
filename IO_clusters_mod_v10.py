import os,sys
import numpy as np
import netCDF4 as nc
	
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from scipy.io import loadmat
from scipy import interpolate

from datetime import datetime

# run a K-means clustering for T/S
from sklearn.preprocessing import Imputer	
from sklearn import preprocessing#, mixture
#from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from palettable.colorbrewer.diverging import PRGn_10 as prg
from palettable.colorbrewer.sequential import GnBu_9 as gb
from palettable.colorbrewer.sequential import YlOrRd_9 as ord
from palettable.colorbrewer.sequential import YlGnBu_9 as ord2
from palettable.colorbrewer.diverging import RdYlBu_10 as ryb
from palettable.colorbrewer.diverging import RdBu_10 as rb
from palettable.colorbrewer.qualitative import Pastel2_5 as past
# import cm colorbars
sys.path.append('/home/irosso/cmocean-master/cmocean')
import cm

# import gsw library
sys.path.append('/home/irosso/gsw-3.0.3')
import gsw

regions     = range(1,9)

# depth at which to run the kmean or gmm
zlev        = 50
kmns        = True
plot_traj   = False
save_bin_dens_floats= False
save_bin_dens_clim = False
save_bin_dens_Argo = False
save_argo_files = False

HOMEdir     = '/data/irosso'
plotdir     = os.path.join(HOMEdir,'plots/floats')
ETOPO       = '/data/irosso/ETOPO/ETOPO1_Ice_g_gmt4.grd'
data        = nc.Dataset(ETOPO)
XC          = data.variables['x'][:]
YC          = data.variables['y'][:]
bathy       = data.variables['z'][:]
bathy       = np.ma.masked_greater(bathy,0.)
bathy       = -bathy 
v           = np.linspace(0,6,7)
x1          = np.min(np.where(XC>=0))
x2          = np.min(np.where(XC>=180))
y1          = np.min(np.where(YC>=-70))
y2          = np.min(np.where(YC>=-30))

fname       = ['G2D','Practical salinity','Conserv temperature','Potential temperature','NO','Chl a','Oxygen','TAlk','DIC','pCO2','pH','Nitrate','Spiciness'];
G_var       = ['AABW','LNADW','UNADW','MSOW','RSOW','AAIW','NPIW','ALBW','RSBW','WSW','RSBWFR'] #names of the var in G_wm
wm_index    = [2,5,9]
labZ        = ['Subtropical Zone','Subantarctic Zone','Polar Front Zone','Antarctic Zone','Sea Ice Zone']  
prop        = ['SA','TE','OXY','NO3','DIC']
axtit       = ['SA',r'$\theta$','O$_2$','NO$_3$','DIC']
xlab        = ['', '[$^{\circ}C$]','[$\mu$mol kg$^{-1}$]','[$\mu$mol kg$^{-1}$]','[$\mu$mol kg$^{-1}$]']
regTit      = ['West','Upstream','Downstream','East']

folder      = os.path.join(HOMEdir,'SOCCOM_floats_IO')
files       = [f for f in os.listdir(folder) if 'QC' in f]
files.sort()

#col   = ['y','m','c','orange','k','b','r','gray','pink','green','lime','royalblue','slateblue','crimson','yellowgreen','peru']
col         = ['r','c','m','green','b','orange','yellowgreen','gray','pink','lime','royalblue','slateblue','crimson','yellowgreen','peru']

labT        = ['LCDW', 'UCDW', 'AAIW', 'SAMW', 'WW', 'AASW', 'PFSW', 'SASW', 'ARC', 'STSW']
WM_SPR      = np.nan*np.ones((2,len(labT),100000))
WM_SUM      = np.nan*np.ones((2,len(labT),100000))
WM_AUT      = np.nan*np.ones((2,len(labT),100000))
WM_WIN      = np.nan*np.ones((2,len(labT),100000))

all_floats  = ['9313','0690','9600','9637','9650','9749','0508','9260','0692','0691','9645']
zoneset= range(1,6)

#----------------------------------------------------
def plot_Orsi(fig,ax,c):
	frontdir  = '/data/irosso/Orsi_fronts'
	fronts    = ['pf', 'stf','saf', 'saccf', 'sbdy']
	col_FF    = ['k', 'k', 'k', 'k', 'k']
	ll1       = [100,161,170,120,140]
	ll2       = [-50, -47, -57, -64, -65]
	xRoll     = 0.
	# plot Orsi's fronts
	for ii,ff in enumerate(fronts[1:-1]):
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
						indNaN   = np.where(np.abs(np.diff(lon_FF))>30.)
						lon_FF2  = np.insert(lon_FF,indNaN[0][:]+1,np.nan)
						lat_FF2  = np.insert(lat_FF,indNaN[0][:]+1,np.nan)
					else:
						lon_FF2  = lon_FF
						lat_FF2  = lat_FF
				
					if len(c)!=0:
						col = c
					else:
						col = col_FF[ii]
					if ii == 0:
						style = ':'
					else:
						style = '-'
					ax.plot(lon_FF2, lat_FF2,col,alpha=0.5,linewidth=1)
					lon_FF = []
					lat_FF = []
		
		data.close()
		#ax.annotate('%s' %(fronts[ii+1]),xy=(ll1[ii+1],ll2[ii+1]),fontsize=12)
	
	return fig
#----------------------------------------------------
def plot_PF(fig,ax,c):
	file = '/data/irosso/PolarFront/Polar_Front_weekly_NFreeman.nc'
	data = nc.Dataset(file)
	PF   = data.variables['PFw'][:]
	long = data.variables['longitude'][:]
	PFm  = np.ma.masked_invalid(PF)
	PFmean= np.nanmean(PFm,axis=0)
	PFvar= np.nanvar(PFm,axis=0)
	PFstd= np.sqrt(PFvar)
	
	x1     = np.min(np.where(long>=0))
	x2     = np.min(np.where(long>=180))
	
	PFmean = PFmean[x1:x2]
	PFstd  = PFstd[x1:x2]
	
	ax.plot(long[x1:x2],PFmean,color=c,alpha=0.5,linewidth=1)
	#ax.plot(long[x1:x2],PFmean+PFstd,color=c,linestyle='--',alpha=0.3,linewidth=1)
	#ax.plot(long[x1:x2],PFmean-PFstd,color=c,linestyle='--',alpha=0.3,linewidth=1)
	ax.set_xlim(long[x1],long[x2])
	#ax.annotate('pf', xy=(135,-54),fontsize=12)
	
	return fig
#----------------------------------------------------
def plot_clusters(data,oxy,no3,pr,col,WMtot,labels,vd):
	plt.figure(figsize=(15,20))
	for ii,id in enumerate(idtot[:n_clust]):
		#	ax = plt.subplot(131)
		ax = plt.subplot(331)	
		im = ax.scatter(data[id,1], data[id,0], c=col[ii],marker='.',edgecolors='face',s=10,alpha=0.4)
		ax.set_xlabel('SP', fontsize=12)
		ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		ax.set_xlim(np.min(ss),36)#np.max(ss))#(33.5,35.5)#
		#ax.set_ylim(0,18)#(-2,4)#(-2,10)
		if ii == 0:
			cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
			plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
		#ax.scatter(centers[ii,1],centers[ii,0],c=col[ii],marker='*',edgecolor='k',s=200,zorder=10000)

		#	ax = plt.subplot(132)		
		ax = plt.subplot(332)
		im = ax.scatter(oxy[id],data[id,0],c=col[ii],s=10,marker='.',edgecolors='face',alpha=0.4)
		ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		ax.set_xlabel(r'O$_2$ [$\mu$mol kg$^{-1}$]', fontsize=12)
		#ax.set_xlim(33.5,36)
		plt.grid('on')
	
		ax = plt.subplot(333)
		im = ax.scatter(no3[id],data[id,0],c=col[ii],s=10,marker='.',edgecolors='face',alpha=0.4)
		ax.plot(np.nan,np.nan,c=col[ii],label='%i' %ii,linewidth=4)
		ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		ax.set_xlabel(r'NO$_3$ [$\mu$mol kg$^{-1}$]', fontsize=12)
		#ax.set_xlim(33.5,36)
		plt.grid('on')

	ax = plt.subplot(333)	
	plt.legend(loc=1,ncol=3,fancybox=True)

	ax=plt.subplot(334)
	im=ax.scatter(data[:,1],data[:,0],c=pr,edgecolors='face',alpha=0.4)
	im.set_clim(0,30)
	plt.colorbar(im)
	ax.set_xlabel('SP', fontsize=12)
	ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)

	ax=plt.subplot(335)
	im=ax.scatter(oxy,data[:,0],c=pr,edgecolors='face',alpha=0.4)
	im.set_clim(0,30)
	ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
	ax.set_xlabel(r'O$_2$ [$\mu$mol kg$^{-1}$]', fontsize=12)		
	plt.grid('on')
	
	ax=plt.subplot(336)
	im=ax.scatter(no3,data[:,0],c=pr,edgecolors='face',alpha=0.4)
	im.set_clim(0,30)
	ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
	ax.set_xlabel(r'NO$_3$ [$\mu$mol kg$^{-1}$]', fontsize=12)
	plt.grid('on')
	

	for ii,id in enumerate(WMtot):
		ax = plt.subplot(337)	
		im = ax.scatter(data[id,1], data[id,0], c=col[ii],marker='.',edgecolors='face',s=10,alpha=0.4)
		ax.plot(np.nan,np.nan,color=col[ii],label='%s' %lab[ii],linewidth=4)
		ax.set_xlabel('SP', fontsize=12)
		ax.set_xlim(np.min(ss),np.max(ss)+0.5)
		ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		if ii == 0:
			cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
			plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
		#ax.set_xlim(33.5,35.5)
		#ax.set_ylim(-2,4)#(0,12)#(-2,10)

		ax = plt.subplot(338)
		im = ax.scatter(oxy[id],data[id,0],c=col[ii],s=10,marker='.',edgecolors='face',alpha=0.4)
		ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		ax.set_xlabel(r'O$_2$ [$\mu$mol kg$^{-1}$]', fontsize=12)
		plt.grid('on')

		ax = plt.subplot(339)
		im = ax.scatter(no3[id],data[id,0],c=col[ii],s=10,marker='.',edgecolors='face',alpha=0.4)
		ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		ax.set_xlabel(r'NO$_3$ [$\mu$mol kg$^{-1}$]', fontsize=12)
		plt.grid('on')
		
	ax = plt.subplot(335)	
	plt.legend(loc=4,fancybox=True)

	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_zone_%i.png' %zone)
	print outfile

	plt.savefig(outfile, bbox_inches='tight',dpi=200)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def load_GLODAP():
	# domain limits:
	# lat: -70, -30
	# lon: 0,180
	# lonSource: 0,50
	# lonUpStream: 50, 68
	# lonDownStream: 68, 110
	# lonEast: 110, 180
	climVar = ['salinity','temperature','oxygen','NO3']
	for ii,cc in enumerate(climVar):
		file   = '/data/averdy/datasets/GLODAP/v2/GLODAPv2.2016b_MappedClimatologies/GLODAPv2.2016b.%s.nc' %cc
		dataCl = nc.Dataset(file)
		if ii == 0:
			depC    = dataCl.variables['Depth'][:]	
			lonC    = dataCl.variables['lon'][:]	
			latC    = dataCl.variables['lat'][:]
			zC1     = np.min(np.where(depC>=50.))
			zC2     = np.min(np.where(depC>=2000.))
			yC1     = np.min(np.where(latC>=-70))
			yC2     = np.min(np.where(latC>=-30))
			xC1     = 0
			xC2     = np.min(np.where(lonC>=180))  
			saClim  = dataCl.variables[cc][:zC2+1,yC1:yC2,xC1:xC2]	
		elif ii == 1:
			teClim  = dataCl.variables[cc][:zC2+1,yC1:yC2,xC1:xC2]	
			mapErr  = dataCl.variables['temperature_relerr'][:zC2+1,yC1:yC2,xC1:xC2]	
			inN     = dataCl.variables['Input_N'][0,yC1:yC2,xC1:xC2]
		elif ii == 2:
			oxyClim = dataCl.variables[cc][:zC2+1,yC1:yC2,xC1:xC2]	
		elif ii == 3:
			no3Clim = dataCl.variables[cc][:zC2+1,yC1:yC2,xC1:xC2]	

	return [saClim,teClim,oxyClim,no3Clim,mapErr,inN,lonC,latC,depC]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def load_SO_argo():
	from datetime import datetime,date, timedelta
	
	programs = os.listdir('/data/irosso/argo/South_Indian_Ocean_single_files')
	
	tot_fl   = 0
	iniDate  = datetime(1950,1,1)
	minDate  = datetime(2019,1,1).toordinal()
	maxNprof = 0
	# initialize big arrays
	raw_dataSO  = np.nan*np.ones((2,822*2000*310))
	coor_dataSO = np.nan*np.ones((3,822*2000*310))
	pr_dataSO   = np.nan*np.ones((822*2000*310))
	s0_dataSO   = np.nan*np.ones((822*2000*310))
	seasonsSO   = np.zeros((822*2000*310),'int32')
	years_dataSO= np.zeros((822*2000*310),'int32')
	nnprofSO    = np.zeros((822*2000*310),'int32')
	IDSO        = np.zeros((822*2000*310),'int32')
	# initialize arrays with only a certain depth (50) values
	PT_surfSO   = np.nan*np.ones((822*310))
	SP_surfSO   = np.nan*np.ones((822*310))
	id_surfSO   = np.zeros((822*310),'int32')
	nn_surfSO   = np.zeros((822*310),'int32')
	lon_surfSO  = np.nan*np.ones((822*310))
	lat_surfSO  = np.nan*np.ones((822*310))
	
	dateFlNum   = []
	ll          = 0
	ll1         = 0
	l0          = 0
		
	"""
	fig,ax   = plt.subplots(figsize=(12,7))
	im       = ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
	ax.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')
	ax.set_xlim(XC[x1],XC[x2])
	ax.set_ylim(YC[y1],YC[y2])
	ax.set_title('Argo floats in the Indian sector of the Southern Ocean \n(822 floats since Dec 12, 2010 for a maximum of 725 profiles)',fontsize=16)
	cax     = fig.add_axes([0.15, 0.15, 0.2, 0.01])
	cbar    = plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('[km]')
	cbar.ax.xaxis.set_tick_params(color='k')
	"""
	for pp in programs:
		SO_floats = os.listdir('/data/irosso/argo/South_Indian_Ocean_single_files/%s' %pp)
		for ff_SO in SO_floats:
			print ff_SO
			SO_prof = [f for f in os.listdir('/data/irosso/argo/South_Indian_Ocean_single_files/%s/%s' %(pp,ff_SO)) if 'prof.nc' in f]
			tot_fl += 1
			file    = '/data/irosso/argo/South_Indian_Ocean_single_files/%s/%s/%s' %(pp,ff_SO,SO_prof[0])
			data    = nc.Dataset(file)
			time    = data.variables['JULD'][:]
			lon     = data.variables['LONGITUDE'][:]
			lat     = data.variables['LATITUDE'][:]
			pressPre= data.variables['PRES_ADJUSTED'][:].transpose()
			tempPre = data.variables['TEMP_ADJUSTED'][:].transpose()
			tempQC  = data.variables['TEMP_ADJUSTED_QC'][:].transpose()
			# THIS IS NOT THETA!!! GOTTA TRANSFORM INTO THETA LATER!!
			psaPre   = data.variables['PSAL_ADJUSTED'][:].transpose()
			psaQC    = data.variables['PSAL_ADJUSTED_QC'][:].transpose()
			psaPre[np.where(psaQC!='1')] = np.nan
			tempPre[np.where(tempQC!='1')] = np.nan
			# let's get rid of some profiles that are out of the indian sector
			msk1    = np.where(np.logical_and(lon>=0,lon<180))[0][:]
			msk2    = np.where(np.logical_and(lat[msk1]>-70,lat[msk1]<-30))[0][:]
			
			lon     = lon[msk1][msk2]
			lat     = lat[msk1][msk2]
			pressPre= pressPre[:,msk1[msk2]]
			tempPre = tempPre[:,msk1[msk2]]
			psaPre  = psaPre[:,msk1[msk2]]
			if time[0] + iniDate.toordinal() < minDate:
				minDate = time[0]
			if len(lon) > maxNprof:
				maxNprof = len(lon[~np.isnan(lon)])
				flN      = ff_SO
			# plot trajectories
			#f1 = lon[~np.isnan(lon)]
			#f2 = lat[~np.isnan(lon)]
			#ax.scatter(f1,f2,c='DodgerBlue',s=1,edgecolor='face',alpha=0.3)
			
			# interpolate and prepare the data like the SOCCOM floats
			lat       = np.ma.masked_less(lat,-90)
			lon       = np.ma.masked_less(lon,-500)
			lon[lon>360.] = lon[lon>360.]-360.
			Nprof     = np.linspace(1,len(lat),len(lat))
			# turn the variables upside down, to have from the surface to depth and not viceversa
			if any(pressPre[:10,0]>500.):
				pressPre = pressPre[::-1,:]
				psaPre   = psaPre[::-1,:]
				tempPre  = tempPre[::-1,:]
			# interpolate data on vertical grid with 1db of resolution (this is fundamental to then create profile means)
			fields    = [psaPre,tempPre]
			press     = np.nan*np.ones((2000,pressPre.shape[1]))
			for kk in range(press.shape[1]):
				press[:,kk] = np.arange(2,2002,1)
			psa   = np.nan*np.ones((press.shape),'>f4')
			temp  = np.nan*np.ones((press.shape),'>f4')	
			for ii,ff in enumerate(fields):
				for nn in range(pressPre.shape[1]):
					# only use non-nan values, otherwise it doesn't interpolate well
					try:
						f1 = ff[:,nn][ff[:,nn].mask==False] #ff[:,nn][~np.isnan(ff[:,nn])]
						f2 = pressPre[:,nn][ff[:,nn].mask==False]
					except:
						f1 = ff[:,nn]
						f2 = pressPre[:,nn]
					if len(f1)==0:
						f1 = ff[:,nn]
						f2 = pressPre[:,nn]
					try:
						sp = interpolate.interp1d(f2,f1,kind='slinear', bounds_error=False, fill_value=np.nan)
						ff_int = sp(press[:,nn]) 
						if ii == 0:
							psa[:,nn]   = ff_int
						elif ii == 1:
							temp[:,nn] = ff_int
					except:
						print 'At profile number %i, the float %s has only 1 record valid: len(f2)=%i' %(nn,ff_SO,len(f2))
							
			# I need to compute theta and sigma0. 
			# To compute theta, I need absolute salinity [g/kg] from practical salinity (PSS-78) [unitless] and conservative temperature.
			# To compute sigma0, I need conservative temperature and absolute salinity
			sa = np.nan*np.ones((press.shape),'>f4') 
			for kk in range(press.shape[1]):
				sa[:,kk] = gsw.SA_from_SP(psa[:,kk], press[:,0], lon[kk], lat[kk])	
			ptemp     = gsw.pt_from_CT(sa, temp)			
			sigma0    = gsw.sigma0(sa,temp)
			
			# mask out the profiles with :
			msk       = np.where(lat<-1000)
			lat[msk]  = np.nan
			lon[msk]  = np.nan
			sa[msk]   = np.nan
			sa[sa==0.]= np.nan
			ptemp[msk] = np.nan
			ptemp[temp==0.]= np.nan
			sigma0[msk]= np.nan
			sigma0[sigma0==0.]= np.nan
			
			# save the nprofiles
			NN        = np.ones((temp.shape),'int32')
			for ii in range(len(Nprof)):
				NN[:,ii]=Nprof[ii]
				
			floatID   = int(ff_SO)*np.ones((sa.shape),'int32')
			coor_dataSO[0,ll:ll+len(sa.flatten())] = np.array(sa.shape[0]*[lon]).flatten()
			coor_dataSO[1,ll:ll+len(sa.flatten())] = np.array(sa.shape[0]*[lat]).flatten()
			raw_dataSO[0,ll:ll+len(sa.flatten())]  = ptemp.flatten()
			raw_dataSO[1,ll:ll+len(sa.flatten())]  = sa.flatten()
			pr_dataSO[ll:ll+len(sa.flatten())]  = press.flatten()
			s0_dataSO[ll:ll+len(sa.flatten())]  = sigma0.flatten()
			nnprofSO[ll:ll+len(sa.flatten())]   = NN.flatten()
			IDSO[ll:ll+len(sa.flatten())]       = floatID.flatten()
			
			# separate seasons
			dateFl   = []
			for dd in time:
				floatDate = iniDate + timedelta(float(dd))
				floatDate.strftime('%Y/%m/%d %H:%M:%S%z')
				dateFl.append(floatDate)
				dateFlNum = np.append(dateFlNum,floatDate.toordinal())

			yearsSO   = np.array([int(dd.year) for dd in dateFl])
			monthsSO  = np.array([int(dd.month) for dd in dateFl])
			SPR       = np.where(np.logical_and(monthsSO >= 9,monthsSO <=11))
			SUM       = np.where(np.logical_or(monthsSO == 12,monthsSO <=2))
			AUT       = np.where(np.logical_and(monthsSO >= 3,monthsSO <=5))
			WIN       = np.where(np.logical_and(monthsSO >= 6,monthsSO <=8))
			mmFlSO    = monthsSO.copy()
			mmFlSO[SPR] = 1
			mmFlSO[SUM]  =2
			mmFlSO[AUT] = 3
			mmFlSO[WIN] = 4	
			
			mm2DSO      = np.zeros((sa.shape))
			yy2DSO      = np.zeros((sa.shape))
			for jj in range(mm2DSO.shape[1]):
				mm2DSO[:,jj] = mmFlSO[jj]
				yy2DSO[:,jj] = yearsSO[jj]
			seasonsSO[ll:ll+len(sa.flatten())] = mm2DSO.flatten()
			years_dataSO[ll:ll+len(sa.flatten())] = yy2DSO.flatten()
			
			ll  = ll + len(sa.flatten())
			ll1 = ll1 + len(lon)
			
			# surface data:
			try:
				[np.max(np.where(press[:,ii]<=zlev)) for ii in range(len(Nprof))]
				surf = [np.max(np.where(press[:,ii]<=zlev)) for ii in range(len(Nprof))]
				imax = len(Nprof)
			except:                   
				print ii
				surf  = [np.max(np.where(press[:,jj]<=zlev)) for jj in range(ii)]
				imax  = ii
			PT_surfSO[l0:l0+imax] = [ptemp[surf[ii],ii] for ii in range(imax)]
			SP_surfSO[l0:l0+imax] = [sa[surf[ii],ii] for ii in range(imax)]
			id_surfSO[l0:l0+imax] = int(ff_SO)*np.ones((imax),'int32')
			nn_surfSO[l0:l0+imax]  = [NN[surf[ii],ii] for ii in range(imax)]
			lon_surfSO[l0:l0+imax] = lon[:imax]
			lat_surfSO[l0:l0+imax] = lat[:imax]
			l0   = l0 + imax
				
	#outfile = os.path.join(plotdir,'Indian_Ocean_Argo_floats.png')
	#print outfile
	#plt.savefig(outfile, bbox_inches='tight',dpi=200)
	#plt.close()
	
	PT_surfSO[np.where(SP_surfSO==99999.)] = np.nan 
	SP_surfSO[np.where(SP_surfSO==99999.)] = np.nan
	PT_surfSO[np.where(SP_surfSO<30.)] = np.nan 
	SP_surfSO[np.where(SP_surfSO<30.)] = np.nan
	
	plt.figure()
	f1 = SP_surfSO
	f2 = PT_surfSO
	plt.scatter(f1[~np.isnan(f1)],f2[~np.isnan(f1)])
	# print data.variables
	print 'There are %i floats crossing the boundaries of the Indian Ocean sector of the SO' %(tot_fl)
	minArgoDate = timedelta(float(minDate))+iniDate
	minArgoDate.strftime('%Y/%m/%d %H:%M:%S%z')
	print "The earliest Argo profile available is on %s" %minArgoDate
	print "There are maximum %i profiles (float #%s)" %(maxNprof,flN)
	
	surfaceSO = [lon_surfSO,lat_surfSO,PT_surfSO,SP_surfSO,id_surfSO,nn_surfSO]
	profSO    = [coor_dataSO,raw_dataSO,pr_dataSO,s0_dataSO,nnprofSO,IDSO]
	temporalSO= [seasonsSO,years_dataSO,dateFlNum]
	
	return [surfaceSO,profSO,temporalSO]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def load_Landsch():
	file    = '/data/averdy/datasets/Landschutzer/spco2_1982-2015_MPI_SOM-FFN_v2016.nc'
	data    = nc.Dataset(file)
	lon     = data.variables['lon'][:]	
	lat     = data.variables['lat'][:]
	var     = data.variables['fgco2_smoothed'][:] #"mol m^{-2} yr^{-1}" 
	yL1     = np.min(np.where(lat>=-70))
	yL2     = np.min(np.where(lat>=-30))
	xL1     = np.min(np.where(lon>=0))
	xL2     = None
	
	lon     = lon[xL1:xL2]
	lat     = lat[yL1:yL2]
	var     = var[:,yL1:yL2,xL1:xL2]
	
	return [lon,lat,var]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def load_CO2_flx(freq):
	#file    = '/data/irosso/data/SOCCOM/CO2_flux/soccom_float_flux_28Aug18.nc'
	file    = '/data/irosso/data/SOCCOM/CO2_flux/soccom_float_flux_19Oct18.nc'
	data    = nc.Dataset(file)
	time    = data.variables['Time_%s' %(freq)][:]
	IDflx   = data.variables['Float_%s' %(freq)][:]	
	flx     = data.variables['Air-sea CO2 flux_%s' %(freq)][:]
	lat     = data.variables['Lat_%s' %(freq)][:]	
	lon     = data.variables['Lon_%s' %(freq)][:]	
	xlim    = np.where(lon<=180)
	time    = time[xlim]
	lon     = lon[xlim]
	flx     = flx[xlim]
	IDflx   = IDflx[xlim]
	lat     = lat[xlim]
		
	# convert time from days since 1/1/1950 to dd/mm/yy
	from datetime import datetime,date, timedelta
	iniDate  = datetime(1950,1,1)
	dateFl   = []
	micros   = []
	dateFlNum= []
	for dd in time:
		floatDate = iniDate + timedelta(float(dd))
		floatDate.strftime('%Y/%m/%d %H:%M:%S%z')
		dateFl.append(floatDate)
		dateFlNum = np.append(dateFlNum,floatDate.toordinal())
		# extracting the microseconds allows me to select only the real float's profiles, 
		#and not the 6 hourly interpoleted ones
		micros.append(int(floatDate.microsecond))
	micros=np.array(micros)
					
	months    = np.array([int(dd.month) for dd in dateFl])
	SPR       = np.where(np.logical_and(months >= 9,months <=11))
	SUM       = np.where(np.logical_or(months == 12,months <=2))
	AUT       = np.where(np.logical_and(months >= 3,months <=5))
	WIN       = np.where(np.logical_and(months >= 6,months <=8))
	mmFl      = months.copy()
	mmFl[SPR] = 1
	mmFl[SUM] = 2
	mmFl[AUT] = 3
	mmFl[WIN] = 4	
	
	all_set   = ['0506','0507','0508','0510','0690','0691','0692','9096','9260','9313','9600','9602','9637','9645','9650','9749','9757','12537','12558']
	all_set   = [float(ii) for ii in all_set]
	flxFl     = np.zeros((6,len(np.unique(IDflx))*300))
	ll        = 0
	if freq == 'month':
		# interpolated, monthly
		fig     = plt.figure(100,figsize=(12,12))
		ax1     = fig.add_subplot(211)
		ax2     = fig.add_subplot(212)
		axx     = [ax1,ax2]
		seastit   = ['Spring + Summer','Autumn + Winter']
		for ii,ax in enumerate(axx):
			im      = ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
			ax.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
			plot_Orsi(fig,ax,'k')
			plot_PF(fig,ax,'k')
			ax.set_xlim(XC[x1],XC[x2])
			ax.set_ylim(YC[y1],YC[y2])
			ax.set_title('%s' %seastit[ii],fontsize=16)
		cax     = fig.add_axes([0.15, 0.15, 0.2, 0.01])
		cbar    = plt.colorbar(im,cax=cax,orientation='horizontal')
		cbar.ax.set_xlabel('[km]')
		cbar.ax.xaxis.set_tick_params(color='k')
		for ii,id in enumerate(np.unique(IDflx)):
			if id in all_set:
				idx   = np.where(IDflx==id)
				warm  = np.where(np.logical_or(mmFl[idx]==1,mmFl[idx]==2))
				cold  = np.where(np.logical_or(mmFl[idx]==3,mmFl[idx]==4))
				im  = ax1.scatter(lon[idx][warm],lat[idx][warm],c=flx[idx][warm],marker='.',edgecolors='face',s=200,alpha=0.7,cmap='coolwarm')
				im.set_clim(-5,5)
				im  = ax2.scatter(lon[idx][cold],lat[idx][cold],c=flx[idx][cold],marker='.',edgecolors='face',s=200,alpha=0.7,cmap='coolwarm')
				im.set_clim(-5,5)
				ax1.plot(lon[idx],lat[idx],linewidth=0.5,color='k',alpha=0.4)
				ax2.plot(lon[idx],lat[idx],linewidth=0.5,color='k',alpha=0.4)
		cax     = fig.add_axes([0.65, 0.15, 0.2, 0.01])
		cbar    = plt.colorbar(im,cax=cax,orientation='horizontal',extend='both')
		cbar.ax.set_xlabel('[mol m$^{-2}$ y$^{-1}$]')
		cbar.ax.xaxis.set_tick_params(color='k')

		ax2.text(150,-61,'outgassing',fontsize=14, color='w',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.8))
		ax2.text(120,-61,'uptake',fontsize=14, color='w',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.8))

		outfile = os.path.join(plotdir,'Indian_Ocean_SOCCOM_floats_CO2flx_%s_Gray_seas.png' %freq)
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
		plt.close()
		
		# build array with monthly flux data (only floats in Indian Ocean)
		for ii,id in enumerate(np.unique(IDflx)):
			if id in all_set:
				idx   = np.where(IDflx==id)[0][:]
				# only the real profiles
				flxFl[0,ll:ll+len(idx)] = len(idx)*[id]
				flxFl[1,ll:ll+len(idx)] = flx[idx]
				flxFl[2,ll:ll+len(idx)] = lon[idx]
				flxFl[3,ll:ll+len(idx)] = lat[idx]
				flxFl[4,ll:ll+len(idx)] = dateFlNum[idx]
				ll += len(idx) 	
	else:
		# build an array with floatId, # profile, flux (NOT 6-hr interpolated)
		for ii,id in enumerate(np.unique(IDflx)):
			if id in all_set:
				idx   = np.where(IDflx==id)[0][:]
				# only the real profiles
				noInterp = np.where(micros[idx]!=0)[0][:]
				flxFl[0,ll:ll+len(noInterp)] = len(noInterp)*[id]
				flxFl[1,ll:ll+len(noInterp)] = flx[idx][noInterp]
				flxFl[2,ll:ll+len(noInterp)] = lon[idx][noInterp]
				flxFl[3,ll:ll+len(noInterp)] = lat[idx][noInterp]
				flxFl[4,ll:ll+len(noInterp)] = dateFlNum[idx][noInterp]
				ll += len(noInterp) 

		### just an easy plot to check
		"""
		plt.figure()
		idxLon = np.argsort(lon)
		mmFls  = mmFl[idxLon]
		lonFls = lon[idxLon]
		flxFls = flx[idxLon]
		idx    = np.where(np.logical_or(mmFls==1,mmFls==2))
		f1     = lonFls[idx]
		f2     = flxFls[idx]
		f1[np.where(np.diff(f1)>=0.1)]=np.nan
		f2[np.where(np.diff(f1)>=0.1)]=np.nan
		plt.plot(f1,f2,'r',alpha=0.5,linewidth=2)
		idx    = np.where(np.logical_or(mmFls==3,mmFls==4))
		f1     = lonFls[idx]
		f2     = flxFls[idx]
		f1[np.where(np.diff(f1)>=0.1)]=np.nan
		f2[np.where(np.diff(f1)>=0.1)]=np.nan
		plt.plot(f1,f2,'b',alpha=0.5,linewidth=2)
		plt.show()
		"""
		###
	
		fig     = plt.figure(100,figsize=(12,12))
		ax1     = fig.add_subplot(211)
		ax2     = fig.add_subplot(212)
		axx     = [ax1,ax2]
		seastit   = ['Spring + Summer','Autumn + Winter']
		for ii,ax in enumerate(axx):
			im      = ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
			ax.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
			plot_Orsi(fig,ax,'k')
			plot_PF(fig,ax,'k')
			ax.set_xlim(XC[x1],XC[x2])
			ax.set_ylim(YC[y1],YC[y2])
			ax.set_title('%s' %seastit[ii],fontsize=16)
		cax     = fig.add_axes([0.15, 0.15, 0.2, 0.01])
		cbar    = plt.colorbar(im,cax=cax,orientation='horizontal')
		cbar.ax.set_xlabel('[km]')
		cbar.ax.xaxis.set_tick_params(color='k')
		for ii,id in enumerate(np.unique(IDflx)):
			if id in all_set:
				idx   = np.where(IDflx==id)
				warm  = np.where(np.logical_or(mmFl[idx]==1,mmFl[idx]==2))
				cold  = np.where(np.logical_or(mmFl[idx]==3,mmFl[idx]==4))
				# only the real profiles
				noInterpW = np.where(micros[idx][warm]!=0)
				noInterpC = np.where(micros[idx][cold]!=0)
				#im  = ax1.scatter(lon6hr[idx][warm],lat6hr[idx][warm],c=flx6hr[idx][warm],marker='.',edgecolors='face',s=200,alpha=0.4,cmap='coolwarm')
				im  = ax1.scatter(lon[idx][warm][noInterpW],lat[idx][warm][noInterpW],c=flx[idx][warm][noInterpW],marker='.',edgecolors='face',s=200,alpha=0.4,cmap='coolwarm')
				im.set_clim(-5,5)
				#im  = ax2.scatter(lon6hr[idx][cold],lat6hr[idx][cold],c=flx6hr[idx][cold],marker='.',edgecolors='face',s=200,alpha=0.4,cmap='coolwarm')
				im  = ax2.scatter(lon[idx][cold][noInterpC],lat[idx][cold][noInterpC],c=flx[idx][cold][noInterpC],marker='.',edgecolors='face',s=200,alpha=0.4,cmap='coolwarm')
				im.set_clim(-5,5)
				ax1.plot(lon[idx],lat[idx],linewidth=0.5,color='k',alpha=0.4)
				ax2.plot(lon[idx],lat[idx],linewidth=0.5,color='k',alpha=0.4)
		cax     = fig.add_axes([0.65, 0.15, 0.2, 0.01])
		cbar    = plt.colorbar(im,cax=cax,orientation='horizontal',extend='both')
		cbar.ax.set_xlabel('[mol m$^{-2}$ y$^{-1}$]')
		cbar.ax.xaxis.set_tick_params(color='k')

		ax2.text(150,-61,'outgassing',fontsize=14, color='w',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.8))
		ax2.text(120,-61,'uptake',fontsize=14, color='w',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.8))

		outfile = os.path.join(plotdir,'Indian_Ocean_SOCCOM_floats_CO2flx_%s_Gray_seas_noInterp.png' %freq)
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
		
		fig     = plt.figure(101,figsize=(10,5))
		ax      = fig.add_subplot(111)
		im      = ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=1) # plt.cm.Greys cm.deep
		ax.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k',alpha=0.5)
		plot_Orsi(fig,ax,'k')
		plot_PF(fig,ax,'k')
		ax.set_xlim(XC[x1],XC[x2])
		ax.set_ylim(YC[y1],YC[y2])
		ax.set_title('SOCCOM floats CO$_2$ flux (6-hourly interpolated)',fontsize=16)
		cax     = fig.add_axes([0.15, 0.2, 0.2, 0.02])
		cbar    = plt.colorbar(im,cax=cax,orientation='horizontal')
		cbar.ax.set_xlabel('[km]')
		cbar.ax.xaxis.set_tick_params(color='k')
		for ii,id in enumerate(np.unique(IDflx)):
			if id in all_set:
				idx   = np.where(IDflx==id)
				warm  = np.where(np.logical_or(mmFl[idx]==1,mmFl[idx]==2))
				cold  = np.where(np.logical_or(mmFl[idx]==3,mmFl[idx]==4))
				# only the real profiles
				noInterpW = np.where(micros[idx][warm]!=0)
				noInterpC = np.where(micros[idx][cold]!=0)
				skip = 8
				im  = ax.scatter(lon[idx][warm][::skip],lat[idx][warm][::skip],c=flx[idx][warm][::skip],marker='*',edgecolors='None',linewidths=0.2,s=100,alpha=0.9,cmap=rb.mpl_colormap)
				#im  = ax1.scatter(lon[idx][warm][noInterpW],lat[idx][warm][noInterpW],c=flx[idx][warm][noInterpW],marker='.',edgecolors='face',s=200,alpha=0.4,cmap='coolwarm')
				im.set_clim(-10,10)
				im  = ax.scatter(lon[idx][cold][::skip],lat[idx][cold][::skip],c=flx[idx][cold][::skip],marker='o',edgecolors='None',linewidths=0.2,s=100,alpha=0.9,cmap=rb.mpl_colormap)
				#im  = ax1.scatter(lon[idx][cold][noInterpC],lat[idx][cold][noInterpC],c=flx[idx][cold][noInterpC],marker='.',edgecolors='face',s=200,alpha=0.4,cmap='coolwarm')
				im.set_clim(-10,10)
				#ax.plot(lon[idx],lat[idx],linewidth=0.5,color='k',alpha=0.4)
		cax     = fig.add_axes([0.59, 0.2, 0.3, 0.02],frameon=True)
		cbar    = plt.colorbar(im,cax=cax,orientation='horizontal',extend='both')
		cbar.ax.set_xlabel('[mol m$^{-2}$ y$^{-1}$]')
		cbar.ax.xaxis.set_tick_params(color='k')
		
		im1, = ax.plot([np.nan],color='gray',marker='*',label='Spring + Summer',linewidth=1)
		im2, = ax.plot([np.nan],color='gray',marker='o',label='Autumn + Winter',linewidth=1)
		leg  = ax.legend(loc=9,ncol=2,handles=[im1,im2],fancybox=True)
		leg.get_frame().set_alpha(0.7)

		ax.text(148,-62,'outgassing',fontsize=14, color='w',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.8))
		ax.text(122,-62,'uptake',fontsize=14, color='w',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.8))

		outfile = os.path.join(plotdir,'Indian_Ocean_SOCCOM_floats_CO2flx_%s_Gray_seas2_2days_no_outline.png' %freq)
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	# identify zone of the profile based on GLODAP clusters
	idxNOzero =  np.where(flxFl[2,:]!=0)[0][:]
	labgr = np.reshape(labTr,[len(latC[yC1:yC2]),len(lonC[xC1:xC2])])
	labm  = np.ma.masked_where(np.ma.getmask(dataTT),labgr)

	for inz in idxNOzero:
		xx = np.min(np.where(lonC[xC1:xC2]>=flxFl[2,inz]))
		yy = np.min(np.where(latC[yC1:yC2]>=flxFl[3,inz]))
		# I need to force some values because GLODAP and floats differ in lat/lon. The cluster algorithm 
		# is based on T/S, not lat/lon. Also, GLODAP's minimum lon is 20E and floats go to 0!
		if np.array(labm[yy,xx])==0.:
			flxFl[5,inz] = 1
		elif flxFl[0,inz]==9313. and flxFl[2,inz]<=10.:
			if flxFl[3,inz]>-40.:			
				flxFl[5,inz] = 1	
			else:		
				flxFl[5,inz] = 2
		elif flxFl[0,inz]==9260. and flxFl[2,inz]<=31:
			flxFl[5,inz] = 3
		elif flxFl[0,inz]==9260. and flxFl[2,inz]<=47 and flxFl[2,inz]>=31 and flxFl[3,inz]<=-46:
			flxFl[5,inz] = 4	
		elif flxFl[0,inz]==9260. and labm[yy,xx]==1.:
			flxFl[5,inz] = 2
		else:
			flxFl[5,inz] = labm[yy,xx]
		
		
	# plot spatial map of fluxes
	fig    = plt.figure(100,figsize=(12,7))
	ax     = fig.add_subplot(111)
	im     = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
	plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
	cax    = fig.add_axes([0.15, 0.2, 0.2, 0.02])
	cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('[km]')
	cbar.ax.xaxis.set_tick_params(color='k')
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')
	ax.set_xlim(XC[x1],XC[x2])
	ax.set_ylim(YC[y1],YC[y2])
	ax.set_title('k-means cluster of 6-hourly interpolated CO$_2$ fluxes',fontsize=16)
	for ii in range(1,6):
		i3  = np.where(flxFl[5,:]== ii)
		im  = ax.scatter(flxFl[2,i3],flxFl[3,i3],c=col[ii-1],marker='.',edgecolors='face',s=200,alpha=0.4,cmap='coolwarm')
		
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_SOCCOM_floats_CO2flx_%s_Gray.png' %(freq))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	plt.show()
		
	return flxFl
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def label_clusters(labels,predict,ensamble):
	id0 = np.where(labels==0)
	id1 = np.where(labels==1)
	id2 = np.where(labels==2) 
	id3 = np.where(labels==3)
	id4 = np.where(labels==4)
	id5 = np.where(labels==5)
	id6 = np.where(labels==6)
	id7 = np.where(labels==7)
	idtot = [id0,id1,id2,id3,id4,id5,id6,id7]

	# group the clusters by region
	lab2      = 1*np.ones(labels.shape)
	lab2[id2] = 2
	lab2[id6] = 2
	lab2[id0] = 3
	lab2[id3] = 4
	lab2[id7] = 5
	
	if predict and ensamble=='SOCCOM':
		# force few points to the right cluster
		tmp = np.where(np.logical_and(lab2==5,lat_surf>-62))[0][:]
		lab2[tmp] = 4
		tmp = np.where(np.logical_and(lab2==4,lat_surf<-62))[0][:]
		lab2[tmp] = 5
		tmp = np.where(np.logical_and(lon_surf<70,id_surf==691))[0][:]
		lab2[tmp] = 3
		tmp = np.where(np.logical_and(lon_surf<70,id_surf==692))[0][:]
		lab2[tmp] = 3
		tmp = np.where(np.logical_and(lat_surf>-40,id_surf==9313))[0][:]
		lab2[tmp] = 1
		tmp = np.where(np.logical_and(lon_surf<8,id_surf==9313))[0][:]
		lab2[tmp] = 2
		tmp = np.where(np.logical_and(lat_surf>-44.3,id_surf==9749))[0][:]
		lab2[tmp] = 1
		tmp = np.where(np.logical_and(lab2==1,np.logical_and(lat_surf<-45,id_surf==9600)))[0][:]
		lab2[tmp] = 2
		tmp = np.where(np.logical_and(lab2==1,id_surf==9637))[0][:]
		lab2[tmp] = 2
		tmp = np.where(np.logical_and(id_surf==9650,lab2==1))[0][:]
		lab2[tmp] = 2
		tmp = np.where(id_surf==507)[0][:]
		lab2[tmp] = 4

	STZ = np.where(lab2==1)
	SAZ = np.where(lab2==2)
	PFZ = np.where(lab2==3)
	AZ  = np.where(lab2==4)
	SZ  = np.where(lab2==5)
	
	return [idtot,lab2,STZ,SAZ,PFZ,AZ,SZ]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def label_argo(labels,data,dens,lat,PT):
	data[data==0.] = np.nan
	idT = np.where(np.isnan(data[:,0]))[0][:]
	idS = np.where(np.isnan(data[:,1]))[0][:]

	labels[idT] = 10000
	labels[idS] = 10000

	id0 = np.where(labels==0)
	id1 = np.where(labels==1)
	id2 = np.where(labels==2) 
	id3 = np.where(labels==3)
	id4 = np.where(labels==4)
	id5 = np.where(labels==5)
	id6 = np.where(labels==6)
	id7 = np.where(labels==7)
	#id8 = np.where(labels==8)
	#id9 = np.where(labels==9)
	#id10 = np.where(labels==10)
	#id11 = np.where(labels==11)
	idtot = [id0,id1,id2,id3,id4,id5,id6,id7]#,id8,id9,id10,id11]

	# group the clusters by region
	lab      = 1*np.ones(labels.shape)
	
	lab[np.where(labels==10000)] = 10000

	#lab[id0] = 2
	lab[id5] = 2
	lab[id7] = 2
	#lab[id9] = 2
	#lab[id10] = 3
	lab[id3] = 4
	lab[id2] = 5
	#lab[id11] = 5

	tmp = np.where(np.logical_and(lab==5,dens<=27.3))[0][:]
	lab[tmp] = 4
	#tmp = np.where(np.logical_and(lab==3,PT>=6))[0][:]
	#lab[tmp] = 2
	#tmp = np.where(np.logical_and(lab==2,PT<6.))[0][:]
	#lab[tmp] = 3
	tmp = np.where(np.logical_and(lab==4,dens<=27.1))[0][:]#26.9))[0][:]
	lab[tmp] = 3
	tmp = np.where(np.logical_and(lab==5,lat>=-63))[0][:]
	lab[tmp] = 4

	STZ = np.where(lab==1)
	SAZ = np.where(lab==2)
	PFZ = np.where(lab==3)
	AZ  = np.where(lab==4)
	SZ  = np.where(lab==5)

	WMtot = [STZ,SAZ,PFZ,AZ,SZ]
	
	return [idtot,lab,STZ,SAZ,PFZ,AZ,SZ]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def extract_profiles(s1,s2,lab,id_surffl,IDfl,raw,coor,oxy,no3,dens,seasons,dic,years,nn_surffl,nnproffl,dateNum,dataset):
	tenn  = np.nan*np.ones((5,s1,s2))
	sann  = np.nan*np.ones((5,s1,s2))
	latnn = np.nan*np.ones((5,s1,s2))
	lonnn = np.nan*np.ones((5,s1,s2))
	ididnn= np.nan*np.ones((5,s1,s2))
	nnprnn= np.nan*np.ones((5,s1,s2))
	if dataset=='SOCCOM':
		oxynn = np.nan*np.ones((5,s1,s2))
		no3nn = np.nan*np.ones((5,s1,s2))
		dicnn = np.nan*np.ones((5,s1,s2))
		yynn  = np.nan*np.ones((5,s1,s2))
		seasnn= np.nan*np.ones((5,s1,s2))
		datesnn = np.nan*np.ones((5,s1))
	else:
		oxynn = []
		no3nn = []
		dicnn = []
		yynn  = []
		seasnn= []
		datesnn=[]

	for ii,id in enumerate(np.unique(lab[np.where(lab!=10000)])):
		jj = 0
		ll = 0
		# extract floats:
		print ii
		fl_list = np.unique(id_surffl[np.where(lab==id)])
		for ff in fl_list:
			if ff != 0.:			
				# extract the data for the whole depth profile
				# 1. extract indexes for the float ID
				idff    = np.where(IDfl==ff)[0][:]
				saff    = raw[1,idff]
				teff    = raw[0,idff]
				if dataset == 'SOCCOM':
					oxyeff  = oxy[idff]
					no3eff  = no3[idff]
					diceff  = dic[idff]
					yyeff   = years[idff]
					seaseff = seasons[idff]
				denseff = dens[idff]
				loneff  = coor[0,idff]
				lateff  = coor[1,idff]
				# extract list of profiles for the float
				nn_list = nn_surffl[np.where(np.logical_and(lab==id,id_surffl==ff))]
				if dataset == 'SOCCOM':
					datesnn[ii,ll:ll+len(np.where(np.logical_and(lab==id,id_surf==ff))[0][:])]=dateNum[np.where(np.logical_and(lab==id,id_surf==ff))]
				ll += len(np.where(np.logical_and(lab==id,id_surffl==ff))[0][:])
				for nn in nn_list:
					# 2. extract the indexes with that nn profile, for the float ID
					idnn    = np.where(nnproffl[idff]==nn)[0][:]
					llN     = len(idnn)
					sann[ii,jj,:llN]  = saff[idnn]
					tenn[ii,jj,:llN]  = teff[idnn]
					if dataset == 'SOCCOM':
						oxynn[ii,jj,:llN] = oxyeff[idnn]
						no3nn[ii,jj,:llN] = no3eff[idnn]
						dicnn[ii,jj,:llN] = diceff[idnn]
						seasnn[ii,jj,:llN]= seaseff[idnn]
						yynn[ii,jj,:llN]  = yyeff[idnn]
					latnn[ii,jj,:llN] = lateff[idnn]
					lonnn[ii,jj,:llN] = loneff[idnn]
					ididnn[ii,jj,:llN]= llN*[ff]
					nnprnn[ii,jj,:llN]= llN*[nn]
					# plot the profiles by region
					#ax.plot(sann[ii,jj,:llN],tenn[ii,jj,:llN],c=col[ii],alpha=0.2,linewidth=0.2)#marker='.',edgecolors='face',s=200,alpha=0.4)
					jj+=1
					
	return [tenn,sann,oxynn,no3nn,dicnn,latnn,lonnn,seasnn,ididnn,nnprnn,yynn,datesnn]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 	
def app_O2(S,T,O2):
	"""
	converted from https://www.mbari.org/products/research-software/matlab-scripts-oceanographic-calculations/
	Input:   S = Salinity (pss-78)
       		 T = Potential Temp (deg C)
             O2 = Measured Oxygen Conc (umol/kg)

 	Output:      Apparant Oxygen Utilization (umol/kg).

    AOU = app_O2(S,T,O2).
	"""
	# DEFINE CONSTANTS, ETC FOR SATURATION CALCULATION
	# The constants -177.7888, etc., are used for units of ml O2/kg.
	
  	T1 = (T + 273.15) / 100
	OSAT = -177.7888 + 255.5907 / T1 + 146.4813*np.log(T1) - 22.2040 * T1
	OSAT = OSAT + S* (-0.037362 + T1* (0.016504 - 0.0020564* T1))
	OSAT = np.exp(OSAT)

	# CONVERT FROM ML/KG TO UM/KG
  	OSAT = OSAT * 1000 / 22.392
	# CALCULATE AOU
  	AOU = OSAT - O2
        Os  = O2/OSAT*100.	
  	return [AOU,Os]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def plot_map_region(lon,lat,lab,WMtot,dataset):
	fig    = plt.figure(100,figsize=(10,5))
	axx    = fig.add_subplot(111)
	im     = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
	plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
	cax    = fig.add_axes([0.15, 0.2, 0.2, 0.02])
	cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('[km]')
	cbar.ax.xaxis.set_tick_params(color='k')
	plot_Orsi(fig,axx,'k')
	plot_PF(fig,axx,'k')

	for ii,id in enumerate(WMtot):
		axx.scatter(lon[np.where(lab==ii+1)],lat[np.where(lab==ii+1)],c=col[ii],marker='.',edgecolors='face',s=200,alpha=0.4) 
		axx.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=4)
		if ii == 4:
			axx.legend(loc=1,fancybox=True)	
	
	axx.plot([40,40],[YC[y1],YC[y2]],'k--',linewidth=2,zorder=10000)
	axx.plot([68,68],[YC[y1],YC[y2]],'k--',linewidth=2,zorder=10000)
	axx.plot([120,120],[YC[y1],YC[y2]],'k--',linewidth=2,zorder=10000)
	axx.set_xlim(XC[x1],XC[x2])
	axx.set_ylim(YC[y1],YC[y2])
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_map_%s_%im_2.png' %(dataset,zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def plot_TS_region(idtot,data,WMtot,dataset):
	"""
	# 1. lat/lon color coded by clusters
	fig1   = plt.figure(figsize=(12,7))
	ax1    = fig1.add_subplot(111)
	im     = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
	plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
	cax    = fig.add_axes([0.15, 0.2, 0.2, 0.02])
	cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('[km]')
	cbar.ax.xaxis.set_tick_params(color='k')
	plot_Orsi(fig1,ax1)
	plot_PF(fig1,ax1)

	for ii,id in enumerate(idtot):
		ax1.scatter(lon_surf[id],lat_surf[id],c=col[ii],marker='.',edgecolors='face',s=200,alpha=0.4) 
		ax1.plot(np.nan,np.nan,color=col[ii],label='%s' %ii,linewidth=4)
	plt.legend(loc=4,fancybox=True)	
	plt.xlim(XC[x1],XC[x2])
	plt.ylim(YC[y1],YC[y2])
	"""
	# 2. T/S cluster figure
	plt.figure(figsize=(17,10))
	for ii,id in enumerate(idtot):
		#	ax = plt.subplot(131)
		ax = plt.subplot(121)	
		im = ax.scatter(data[id,1], data[id,0], c=col[ii],marker='.',edgecolors='face',s=100,alpha=0.4)
		ax.plot(np.nan,np.nan,c=col[ii],label='%i' %ii,linewidth=4)
		ax.set_xlabel('SP', fontsize=12)
		ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		ax.set_xlim(np.min(ss),36)#np.max(ss))#(33.5,35.5)#
		ax.set_ylim(-3,23)
		#ax.set_ylim(0,18)#(-2,4)#(-2,10)
		if ii == 0:
			cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
			plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
		#ax.scatter(centers[ii,1],centers[ii,0],c=col[ii],marker='*',edgecolor='k',s=200,zorder=10000)
		plt.legend(loc=2,fancybox=True)	
		
	for ii,id in enumerate(WMtot):
		ax = plt.subplot(122)	
		im = ax.scatter(data[id,1], data[id,0], c=col[ii],marker='.',edgecolors='face',s=100,alpha=0.4)
		ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=4)
		ax.set_xlabel('SP', fontsize=12)
		ax.set_xlim(np.min(ss),36)
		#ax.set_ylim(-3,23)
		ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		if ii == 0:
			cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
			plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
		plt.legend(loc=4,fancybox=True)	

	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_TS_%s_clusters_%im.png' %(dataset,zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def plot_TS_tr_pr(dataCl,data,WMtotTr):
	fig,ax = plt.subplots()
	for ii,id in enumerate(WMtotTr):
		im = ax.scatter(dataCl[id,1], dataCl[id,0], c=col[ii],marker='.',edgecolors='face',s=100,alpha=0.4)
		ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=4)
		ax.set_xlabel('SP', fontsize=12)
		ax.set_xlim(np.min(ss),36)
		ax.set_ylim(-3,23)
		ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		if ii == 0:
			cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
			plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
		plt.legend(loc=4,fancybox=True)		
	
	for ii,id in enumerate(WMtot):
		im = ax.scatter(data[id,1], data[id,0], c=col[ii],marker='o',edgecolors='k',s=200,alpha=0.2)

	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_train_pred_%s.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
def plot_map_GLODAP(lonC,latC,labTr,inN):
	labgr = np.reshape(labTr,[len(latC[yC1:yC2]),len(lonC[xC1:xC2])])
	labm  = np.ma.masked_where(np.ma.getmask(dataTT),labgr)

	fig,ax = plt.subplots(figsize=(12,7))
	for ii in range(5):
		labm2 = labm.copy()		
		msk   = np.where(labm!=ii+1)
		labm2[msk] = np.nan
		im    = plt.contourf(lonC[xC1:xC2],latC[yC1:yC2],labm2,v=np.linspace(ii+0.5,ii+1+0.5,2),colors=col[ii],alpha=0.8)#,np.arange(-6,0,1),cmap='jet')
	#cbar = fig.colorbar(im, ticks=np.arange(-5.5,0))
	#cbar.ax.set_yticklabels(['%s' %ll for ll in labZ[::-1]])  
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')
	x,y = np.meshgrid(lonC[xC1:xC2],latC[yC1:yC2])
	plt.scatter(x,y,s=50.*mapErr[zC1,...],marker='o',facecolor="None",alpha=0.5,edgecolor='k')
	ax.set_xlim(lonC[xC1],lonC[xC2-1])
	ax.set_ylim(latC[yC1],latC[yC2-1])
	plt.title('k-means algorithm trained with gridded GLODAPv2 temperature and salinity at 50 m')

	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_GLODAPv2_%s.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	"""
	fig,ax = plt.subplots(figsize=(12,7))
	im  = plt.contourf(lonC[xC1:xC2],latC[yC1:yC2],-1*labm,np.arange(-6,0,1),cmap='jet')
	cbar = fig.colorbar(im, ticks=np.arange(-5.5,0))
	cbar.ax.set_yticklabels(['%s' %ll for ll in labZ[::-1]])
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')
	x,y = np.meshgrid(lonC[xC1:xC2],latC[yC1:yC2])
	plt.scatter(x,y,s=50.*mapErr[zC1,...],marker='o',facecolor="None",alpha=0.5,edgecolor='k')
	ax.set_xlim(lonC[xC1],lonC[xC2-1])
	ax.set_ylim(latC[yC1],latC[yC2-1])
	plt.title('k-means algorithm trained with GLODAPv2 temperature and salinity at 50 m')
    """
	
	# only where GLODAP has data
	i1,i2 = np.where(inN>0)
	ll    = labm[i1,i2]
	lon   = lonC[xC1:xC2][i2]
	lat   = latC[yC1:yC2][i1]
	
	fig,ax = plt.subplots(figsize=(12,7))
	plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
	for ii in range(1,6):
		i3  = np.where(ll == ii)
		im  = plt.scatter(lon[i3],lat[i3],c=col[ii-1],s=100,marker='.',edgecolors='face',alpha=0.4,label='%s' %labZ[ii-1])
	plt.legend(loc=4,fancybox=True)			
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')
	ax.set_xlim(lonC[xC1],lonC[xC2-1])
	ax.set_ylim(latC[yC1],latC[yC2-1])
	plt.title('k-means algorithm trained with GLODAPv2 temperature and salinity at 50 m')

	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_GLODAPv2_raw_%s.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	# plot TS raw
	saSub=saClim[zC1,i1,i2]
	teSub=teClim[zC1,i1,i2]
	fig,ax = plt.subplots(figsize=(7,7))
	for ii in range(1,6):
		i3  = np.where(ll == ii)
		im = ax.scatter(saSub[i3], teSub[i3], c=col[ii-1],marker='.',edgecolors='face',s=100,alpha=0.4)
	ax.set_xlabel('SP', fontsize=12)
	ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
	ax.set_title('T/S clusters of GLODAPv2 at 50 m (only obs)')
	ax.set_xlim(np.min(ss),36)
	cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
	plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)

	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_TS_GLODAPv2_raw_%s.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 	
def plot_cruise(idinn,lonnn,latnn):
	HEOBI   	= ['9749','9645','9757']
	I8S     	= ['0564','0510','9602','9637','9650','9600']
	SOE10   	= ['0690','12734','0693','12755','12730','12781','12757']
	ACE         = ['0691','0692','12557','12558','12537'] #or 12537 is actually 12557
	SR03        = ['12782','12779','12748','12741','12736','12709','12702','12370','12784','0688']#12760']
	KAxis       = ['0507','0506'] 
	Eddy        = ['9631','9744']
	A12         = ['0508','9260','9096','9313']
	
	fl_list     = np.unique(coor_data[2,:])[~np.isnan(np.unique(coor_data[2,:]))]
	fl_list     = [str(int(ff)) for ff in fl_list]
	
	fig     	= plt.figure(100,figsize=(10,5))
	ax      	= fig.add_subplot(111)
	im      	= ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.2)
	ax.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k',alpha=0.5)
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')
	ax.set_xlim(XC[x1],XC[x2])
	ax.set_ylim(YC[y1],YC[y2])
	cax     = fig.add_axes([0.15, 0.2, 0.2, 0.02])
	cbar    = plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('[km]')
	cbar.ax.xaxis.set_tick_params(color='k')
	for ff in fl_list:
		print ff
		cc='k'
		ff = ff.zfill(4)
		if ff in HEOBI:
			cc  = '#D099E1'
			st  = '-'
			labC= 'HEOBI'
			mark = 'o'
		elif ff in SR03:
			cc  = '#58BDC7'
			st  = '--'
			labC= 'SR03'
			mark = 'o'
			print 'sr3!!'
		elif ff in ACE:
			cc  = 'b'
			st  = '-'
			labC= 'ACE'
			mark = 'o'
		elif ff in SOE10:
			cc  = 'g'
			st  = '-'
			labC= 'SOE10'
			mark = 'o'
		elif ff in KAxis:
			cc  = 'c'
			st  = '--'	
			labC= 'K-Axis'
			mark = '*'
		elif ff in A12:
			cc  = 'purple'
			st  = '-'
			labC= 'A12'
			mark = 'o'
		elif ff in I8S:
			cc  = 'orange'
			st  = '-'
			labC= 'I8S'
			mark = 'o'
			
		idxFl1,idxFl2 = np.where(ididnn[:,:,0]==float(ff))
		im = ax.scatter(lonnn[idxFl1,idxFl2,0],latnn[idxFl1,idxFl2,0],c=cc,marker=mark,s=50,edgecolors='None',alpha=0.4)
		print cc, ff
	labList = ['HEOBI','ACE','SOE10','K-Axis','A12','I08S']
	colLab  = ['#D099E1','b','g','c','purple','orange']
	for ll,labs in enumerate(labList):
		ax.plot(np.nan,np.nan,color=colLab[ll],label='%s' %labList[ll],linewidth=2)
	leg = ax.legend(loc=1,fancybox=True)
	leg.get_frame().set_alpha(0.7)
	ax.set_xlim(XC[x1],XC[x2])
	ax.set_ylim(YC[y1],YC[y2])
	
	outfile = os.path.join(plotdir,'Indian_Ocean_SOCCOM_traj_cruise.png')
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 	
def smooth(x,window_len,window):
	if x.ndim != 1:
		raise ValueError, "smooth only accepts 1 dimension arrays."
	if x.size < window_len:
		raise ValueError, "Input vector needs to be bigger than window size."
	if window_len<3:
		return x
	if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
		raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"
	s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
	#print(len(s))
	if window == 'flat': #moving average
		w=np.ones(window_len,'d')
	else:
		w=eval('np.'+window+'(window_len)')
	y=np.convolve(w/w.sum(),s,mode='valid')
	return y
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ # 
	
if __name__ == "__main__":
	# map with trajectories
	if plot_traj:
		fig600 = plt.figure(600,figsize=(12,7))
		axTraj = fig600.add_subplot(111)
		fig601 = plt.figure(601,figsize=(12,7))
		axTrajs= fig601.add_subplot(111)
		axTr   = [axTraj,axTrajs]
		for ifig,fig in enumerate([fig600,fig601]):
			im     = axTr[ifig].contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
			axTr[ifig].contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
			cax    = fig.add_axes([0.15, 0.2, 0.2, 0.02])
			cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
			cbar.ax.set_xlabel('[km]')
			cbar.ax.xaxis.set_tick_params(color='k')
			plot_Orsi(fig,axTr[ifig])
			plot_PF(fig,axTr[ifig])
		
	# initialize big arrays
	raw_data  = np.nan*np.ones((2,19*2000*310))
	coor_data = np.nan*np.ones((3,19*2000*310))
	oxy_data  = np.nan*np.ones((19*2000*310))
	dic_data  = np.nan*np.ones((19*2000*310))
	no3_data  = np.nan*np.ones((19*2000*310))
	pr_data   = np.nan*np.ones((19*2000*310))
	s0_data   = np.nan*np.ones((19*2000*310))
	seasons   = np.zeros((19*2000*310),'int32')
	years_data= np.zeros((19*2000*310),'int32')
	nnprof    = np.zeros((19*2000*310),'int32')
	ID        = np.zeros((19*2000*310),'int32')
	dates_all = np.zeros((19*2000*310),'int32')
	# initialize arrays with only a certain depth (50) values
	PT_surf   = np.nan*np.ones((19*310))
	SP_surf   = np.nan*np.ones((19*310))
	O2_surf   = np.nan*np.ones((19*310))
	NO3_surf  = np.nan*np.ones((19*310))
	DIC_surf  = np.nan*np.ones((19*310)) 
	id_surf   = np.zeros((19*310),'int32')
	nn_surf   = np.zeros((19*310),'int32')
	lon_surf  = np.nan*np.ones((19*310))
	lat_surf  = np.nan*np.ones((19*310))
	dateNum   = []
	ll        = 0
	ll1       = 0
	l0        = 0
	for zone in regions:
		useO2   = False
		useNO3  = False
		useAll  = False
		usePr   = True
		#['0506','0507','0508','0510','0690','0691','0692','9096','9260','9313','9600','9602','9631','9637','9645','9650','9749','9757','12537','12558','9744']
		if zone == 1:
			subset = ['9313']
		elif zone == 2:
			subset = ['0690']
		elif zone == 3:
			subset = ['9600','9637','9650','9749']
			useO2 = True
			usePr = False
			useAll= False
		elif zone == 4:
			subset = ['0508']
		elif zone == 5:
			subset = ['9260','0692','0691','9645']
		elif zone == 6:
			subset = ['9096','12781','12734','12757']
			useO2  = False
			useNO3 = True
		elif zone == 7:
			subset = ['0510','9602','9757','12537','12558']
		elif zone == 8:
			subset = ['0506','0507','12730','12755']
		# these are really in the Pacific Ocean... 
		#elif zone == 9:
		#	subset = ['9631','9744']
		
		# ~~~~~~~~~ MAIN ~~~~~~~~~ #
		for ff in files:	
			"""	
			fig = plt.figure(100)
			axD1= fig.add_subplot(131)
			axD2= fig.add_subplot(132)
			axD3= fig.add_subplot(133)
			"""
			#print '\n', ff
			file      = os.path.join(folder,ff)
			floatN    = ff.split('_')[1]
			if floatN in subset:
				print floatN
				WMOID     = loadmat(file)['WMOID'][0]
				press_pre = loadmat(file)['pr']
				lat       = loadmat(file)['lat'].transpose()
				lon       = loadmat(file)['lon'].transpose()
				date      = loadmat(file)['date'][:]
				datee     = [datetime.strptime(da,'%m/%d/%Y') for da in date]
				for ii in range(len(datee)):
					dateNum = np.append(dateNum,datee[ii].toordinal())
				months    = np.array([int(dd.month) for dd in datee])
				years     = np.array([int(dd.year) for dd in datee])
				SPR       = np.where(np.logical_and(months >= 9,months <=11))
				SUM       = np.where(np.logical_or(months == 12,months <=2))
				AUT       = np.where(np.logical_and(months >= 3,months <=5))
				WIN       = np.where(np.logical_and(months >= 6,months <=8))
				mm        = months.copy()
				mm[SPR]   = 1
				mm[SUM]   = 2
				mm[AUT]   = 3
				mm[WIN]   = 4
			
				#press     = press
				lat       = lat[:,0]
				lon       = lon[:,0]
				lat       = np.ma.masked_less(lat,-90)
				lon       = np.ma.masked_less(lon,-500)
				lon[lon>360.] = lon[lon>360.]-360.
				Nprof     = np.linspace(1,len(lat),len(lat))

				# plot traj
				if zone in zoneset:
					lonm      = np.ma.masked_less(lon,-1000)
					latm      = np.ma.masked_less(lat,-1000)
					if plot_traj:
						axTraj.plot(lonm,latm,linewidth=0.5,color='k')
				# load data
				#SP        = loadmat(file)['var_3']
				SP_pre     = loadmat(file)['sa']
				PT_pre     = loadmat(file)['th']
				O2_pre     = loadmat(file)['ox']
				DIC_pre    = loadmat(file)['DIC']
				NO3_pre    = loadmat(file)['NO3']
				
				"""
				# check data: with plt.plot the data get truncated, with scatter it does not
				# this is because there are no 2 consecutive points not nan --> idea of Paul: 
				# loop by profile and extract only the non-nan values
				if floatN == '9637':
					for i in range(len(lat)):
						a1 = PT_pre[:,i][~np.isnan(PT_pre[:,i])]
						a2 = O2_pre[:,i][~np.isnan(O2_pre[:,i])]
						a3 = NO3_pre[:,i][~np.isnan(NO3_pre[:,i])]
						b1 = press_pre[:,i][~np.isnan(PT_pre[:,i])]
						b2 = press_pre[:,i][~np.isnan(O2_pre[:,i])]
						b3 = press_pre[:,i][~np.isnan(NO3_pre[:,i])]
						axD1.plot(a1,b1)
						axD2.plot(a2,b2)
						axD3.plot(a3,b3)
					axD1.invert_yaxis()
					axD1.set_title('PT %s' %(floatN))
					axD2.invert_yaxis()
					axD2.set_title('O2 %s' %(floatN))
					axD3.invert_yaxis()
					axD3.set_title('NO3 %s' %(floatN))
					plt.show()
				"""
				# turn the variables upside down, to have from the surface to depth and not viceversa
				if any(press_pre[:10,0]>500.):
					press_pre = press_pre[::-1,:]
					SP_pre    = SP_pre[::-1,:]
					PT_pre    = PT_pre[::-1,:]
					O2_pre    = O2_pre[::-1,:]
					DIC_pre   = DIC_pre[::-1,:]
					NO3_pre   = NO3_pre[::-1,:]
					
				# interpolate data on vertical grid with 1db of resolution (this is fundamental to then create profile means)
				fields    = [SP_pre,PT_pre,O2_pre,DIC_pre,NO3_pre]
				press     = np.nan*np.ones((2000,press_pre.shape[1]))
				for kk in range(press.shape[1]):
					press[:,kk] = np.arange(2,2002,1)
				SP  = np.nan*np.ones((press.shape),'>f4')
				PT  = np.nan*np.ones((press.shape),'>f4')
				O2  = np.nan*np.ones((press.shape),'>f4')
				DIC = np.nan*np.ones((press.shape),'>f4')
				NO3 = np.nan*np.ones((press.shape),'>f4')

				# I made few plots to understand which kind in the interpolation is the best, and the linear is the winner! 
				# this preserves the shape and the extremes in the original data, without inducing weird steps (quadratic is terrible,
				# nearest gives a lot of unreal steps
				for ii,ff in enumerate(fields):
					for nn in range(press_pre.shape[1]):
						# only use non-nan values, otherwise it doesn't interpolate well
						#mask = (~np.isnan(ff[:,nn]))&(~np.isnan(press_pre[:,nn]))
						if floatN == '9749' and nn == 46 and ii == 0:
							idSpk = np.where(np.abs(np.diff(ff[:,nn]))>=0.08)[0][:]
							ff[idSpk[1:],nn] = np.nan
						f1 = ff[:,nn][~np.isnan(ff[:,nn])]
						f2 = press_pre[:,nn][~np.isnan(ff[:,nn])]
						if len(f1)==0:
							f1 = ff[:,nn]
							f2 = press_pre[:,nn]
						sp = interpolate.interp1d(f2,f1,kind='slinear', bounds_error=False, fill_value=np.nan)
						ff_int = sp(press[:,nn]) 
						
						if ii == 0:
							SP[:,nn] = ff_int
						elif ii == 1:
							PT[:,nn] = ff_int
						elif ii == 2:
							O2[:,nn] = ff_int
						elif ii == 3:
							DIC[:,nn] = ff_int
						elif ii == 4:
							NO3[:,nn] = ff_int			
				sigma0    = gsw.sigma0(SP,PT) #loadmat(file)['sigmaT']
				"""
				if floatN == '9637':
					PT = np.ma.masked_equal(PT,0)
					axD1.plot(PT,press)
					axD2.plot(O2,press)
					axD3.plot(NO3,press)
					axD1.invert_yaxis()
					axD1.set_title('PT %s' %(floatN))
					axD2.invert_yaxis()
					axD2.set_title('O2 %s' %(floatN))
					axD3.invert_yaxis()
					axD3.set_title('NO3 %s' %(floatN))
					plt.show()
				"""
				# mask out the profiles with :
				msk       = np.where(lat<-1000)
				lat[msk]  = np.nan
				lon[msk]  = np.nan
				SP[msk]   = np.nan
				SP[SP==0.]= np.nan
				PT[msk]   = np.nan
				PT[PT==0.]= np.nan
				O2[msk]   = np.nan
				O2[O2==0.]= np.nan
				DIC[msk]  = np.nan
				DIC[DIC==0.]= np.nan
				NO3[msk]  = np.nan
				NO3[NO3==0.]= np.nan
				sigma0[msk]= np.nan
				sigma0[sigma0==0.]= np.nan

				# save the nprofiles
				NN        = np.ones((O2.shape),'int32')
				for ii in range(len(Nprof)):
					NN[:,ii]=Nprof[ii]
					
				floatID   = int(floatN)*np.ones((O2.shape),'int32')	
				# save the data into a bigger file:
				raw_data[0,ll:ll+len(PT.flatten())]  = PT.flatten()
				raw_data[1,ll:ll+len(PT.flatten())]  = SP.flatten()
				coor_data[0,ll:ll+len(PT.flatten())] = np.array(PT.shape[0]*[lon]).flatten()
				coor_data[1,ll:ll+len(PT.flatten())] = np.array(PT.shape[0]*[lat]).flatten()
			 	coor_data[2,ll:ll+len(PT.flatten())] = int(floatN)*np.ones((len(lon)*PT.shape[0]),'int32')
				
				mm2D      = np.zeros((PT.shape))
				yy2D      = np.zeros((PT.shape))
				for jj in range(mm2D.shape[1]):
					mm2D[:,jj] = mm[jj]
					yy2D[:,jj] = years[jj]
				seasons[ll:ll+len(PT.flatten())] = mm2D.flatten()
				years_data[ll:ll+len(PT.flatten())] = yy2D.flatten()
			
				oxy_data[ll:ll+len(PT.flatten())] = O2.flatten()
				dic_data[ll:ll+len(PT.flatten())] = DIC.flatten()
				no3_data[ll:ll+len(PT.flatten())] = NO3.flatten()
				pr_data[ll:ll+len(PT.flatten())]  = press.flatten()
				s0_data[ll:ll+len(PT.flatten())]  = sigma0.flatten()
				nnprof[ll:ll+len(PT.flatten())]   = NN.flatten()
				ID[ll:ll+len(PT.flatten())]       = floatID.flatten()

				ll  = ll + len(PT.flatten())
				ll1 = ll1 + len(lon)
				
				# save the surface data somewhere else:
				# at the surface
				#surf = np.nanargmin(press,axis=0)
				# inside mld
				try:
					[np.max(np.where(press[:,ii]<=zlev)) for ii in range(len(Nprof))]
					surf = [np.max(np.where(press[:,ii]<=zlev)) for ii in range(len(Nprof))]
					imax = len(Nprof)
				except:                   
					print ii
					surf  = [np.max(np.where(press[:,jj]<=zlev)) for jj in range(ii)]
					imax  = ii
				PT_surf[l0:l0+imax] = [PT[surf[ii],ii] for ii in range(imax)]
				SP_surf[l0:l0+imax] = [SP[surf[ii],ii] for ii in range(imax)]
				O2_surf[l0:l0+imax] = [O2[surf[ii],ii] for ii in range(imax)]
				NO3_surf[l0:l0+imax] = [NO3[surf[ii],ii] for ii in range(imax)]
				DIC_surf[l0:l0+imax] = [DIC[surf[ii],ii] for ii in range(imax)]
				id_surf[l0:l0+imax] = int(floatN)*np.ones((imax),'int32')
				nn_surf[l0:l0+imax]  = [NN[surf[ii],ii] for ii in range(imax)]
				lon_surf[l0:l0+imax] = lon[:imax]
				lat_surf[l0:l0+imax] = lat[:imax]
				l0   = l0 + imax

	# !!!!!!!!!!!!
	# 12 July 2018: INI
	# train the kmeans algorithm using GLODAPv2
	[saClim,teClim,oxyClim,no3Clim,mapErr,inN,lonC,latC,depC] = load_GLODAP()
	
	# use only the 50 m climatology
	zC1     = np.min(np.where(depC>=50.))
	zC2     = np.min(np.where(depC>=2000.))
	yC1     = np.min(np.where(latC>=-70))
	yC2     = np.min(np.where(latC>=-30))
	xC1     = 0
	xC2     = np.min(np.where(lonC>=180)) 
	
	dataSS = saClim[zC1,...]
	dataTT = teClim[zC1,...]
	dataTfl= dataTT.flatten()
	dataSfl= dataSS.flatten()
	dataCl = np.zeros((len(dataSfl),2))
	dataCl[:,0] = dataTfl
	dataCl[:,1] = dataSfl
	dataCl[dataCl==-999]=np.nan

	# train the k-means algorithm with gridded GLODAPv2
	imp       = Imputer(missing_values='NaN')
	imputer   = imp.fit(dataCl)
	# normalize data (remove mean and divide by std)
	imp_data  = imputer.transform(dataCl)
	scaled    = preprocessing.scale(imp_data)
	n_digits  = len(np.unique(scaled))
	n_clust   = 8 # this is a good number!!

	est       = KMeans(n_clusters=n_clust, n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=3425, copy_x=True, n_jobs=1)
	est.fit(scaled)
	labels    = est.labels_
	
	app       = label_clusters(labels,predict=False,ensamble='GLODAP')
	idtotTr   = app[0]
	labTr     = app[1]
	WMtotTr   = app[2:]
	del app 

	# predict the clusters for the SOCCOM floats data
	data      = np.zeros((len(PT_surf),2))
	data[:,0] = PT_surf
	data[:,1] = SP_surf
	
	# for plotting
	ss        = np.linspace(np.nanmin(data[:,1])-0.1,36.5,15)#np.nanmax(data[:,1])+0.05,15)
	tt        = np.linspace(-3,25,15)
	ss2, tt2  = np.meshgrid(ss,tt)
	s0        = gsw.sigma0(ss2,tt2)
	vd        = np.arange(23,30,0.5)

	# predict SOCCOM
	imputer   = imp.fit(data)
	imp_data  = imputer.transform(data)
	scaled    = preprocessing.scale(imp_data)

	predNew   = est.predict(scaled)
	app       = label_clusters(predNew,predict=True,ensamble='SOCCOM')
	idtot     = app[0]
	lab2      = app[1]
	WMtot     = app[2:]	
	
	# plot traj color coded by cruise
	#plot_cruise(coor_data)
	
	"""
	# plot traj and T/S float 9645
	idxFl     = np.where(coor_data[2,:]==9645)[0][:]
	fig     = plt.figure(101,figsize=(7,5))
	ax      = fig.add_subplot(111)
	im      = ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.2)
	ax.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k',alpha=0.5)
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')
	ax.set_xlim(XC[x1],XC[x2])
	ax.set_ylim(YC[y1],YC[y2])
	ax.set_title('SOCCOM float #9645',fontsize=16)
	cax     = fig.add_axes([0.15, 0.2, 0.2, 0.01])
	cbar    = plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('[km]')
	cbar.ax.xaxis.set_tick_params(color='k')
	im = plt.scatter(coor_data[0,idxFl],coor_data[1,idxFl],c=np.linspace(1,len(idxFl)/2000,len(idxFl)),marker='o',edgecolors='None',linewidths=0.2,s=100,alpha=0.4,cmap=ord2.mpl_colormap)
	cax     = fig.add_axes([0.65, 0.2, 0.2, 0.01],frameon=True)
	cbar    = plt.colorbar(im,cax=cax,orientation='horizontal',extend='both')
	cbar.ax.set_xlabel('# profile')
	cbar.ax.xaxis.set_tick_params(color='k')
	ax.set_xlim(60,140)
	ax.set_ylim(-60,-40)
	"""
			
	# predict all SO argo
	#surfaceSO = [lon_surfSO,lat_surfSO,PT_surfSO,SP_surfSO,id_surfSO,nn_surfSO]
	#profSO    = [coor_dataSO,raw_dataSO,pr_dataSO,s0_dataSO,nnprofSO]
	# load SO argo
	[surfaceSO,profSO,temporalSO] = load_SO_argo()
	# surface data
	lon_surfSO = surfaceSO[0]
	lat_surfSO = surfaceSO[1]
	PT_surfSO  = surfaceSO[2]
	SP_surfSO  = surfaceSO[3]
	id_surfSO  = surfaceSO[4]
	nn_surfSO  = surfaceSO[5]
	# all depth profiles
	coor_dataSO= profSO[0]
	raw_dataSO = profSO[1]
	pr_dataSO  = profSO[2]
	s0_dataSO  = profSO[3]
	nnprofSO   = profSO[4]
	IDSO       = profSO[5]
	# dates and seasons
	seasonsSO  = temporalSO[0]
	years_dataSO = temporalSO[1]
	dateFlNum  = temporalSO[2]
	
	# I need to train the algorithm again for the Argo.. somehow it doesn't work with the GLODAP, still not sure why
	# kmeans
	dataSO      = np.zeros((len(PT_surfSO),2))
	# normalize
	dataSO[:,0] = (PT_surfSO - np.nanmean(PT_surfSO)) / np.nanstd(PT_surfSO)
	dataSO[:,1] = (SP_surfSO- np.nanmean(SP_surfSO)) / np.nanstd(SP_surfSO)

	# take only a smaller part of the dataset to train the algorithm
	data2SO   = dataSO[::30,:]

	# get rid of NaNs (basically, it assignes a different value to these)
	imp       = Imputer(missing_values='NaN')
	imputer   = imp.fit(data2SO)
	imp_data  = imputer.transform(data2SO)

	# this is the normalization part, which I don't really need anymore..
	scaled    = preprocessing.scale(imp_data)

	# kmeans algorithm
	n_digits  = len(np.unique(scaled))
	n_clust   = 8
	est       = KMeans(n_clusters=n_clust, n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=3425, copy_x=True, n_jobs=1)
	est.fit(scaled)

	# clusters
	labelsSO  = est.labels_

	"""
	dataSONew = data2SO.copy()
	dataSONew[:,0] = dataSONew[:,0]*np.nanstd(PT_surfSO) + np.nanmean(PT_surfSO)
	dataSONew[:,1] = dataSONew[:,1]*np.nanstd(SP_surfSO) + np.nanmean(SP_surfSO)

	[idtotSOtrain,lab2SOtrain,STZ,SAZ,PFZ,AZ,SZ] = label_argo(labelsSO,data2SO,lat_surfSO[::30],PT_surfSO[::30])
	"""

	# predict all argo using the trained model
	imputer       = imp.fit(dataSO)
	imp_dataPred  = imputer.transform(dataSO)
	scaledPred    = preprocessing.scale(imp_dataPred)

	#dataSONew      = dataSO.copy()
	#dataSONew[:,0] = dataSONew[:,0]*np.nanstd(PT_surfSO) + np.nanmean(PT_surfSO)
	#dataSONew[:,1] = dataSONew[:,1]*np.nanstd(SP_surfSO) + np.nanmean(SP_surfSO)
	dataSONew      = np.zeros((len(PT_surfSO),2))
	dataSONew[:,0] = PT_surfSO
	dataSONew[:,1] = SP_surfSO
	
	densSO        = gsw.sigma0(dataSONew[:,1],dataSONew[:,0])
	predNew       = est.predict(scaledPred)
	app           = label_argo(predNew,dataSO,densSO,lat_surfSO,PT_surfSO)
	idtotSO       = app[0]
	lab2SO        = app[1]
	WMtotSO       = app[2:]	
	
	# plot some figures
	# 1. T/S plot color coded by region
	# plot_TS_region(idtot,data,WMtot,'floats')
	# GLODAP
	# plot_TS_region(idtotTr,dataCl,WMtotTr,'GLODAPv2')
	# SO argo
	# plot_TS_region(idtotSO,dataSONew,WMtotSO,'SO_Argo')
	# 3. lat/lon color coded by region
	# SOCCOM
	# plot_map_region(lon_surf,lat_surf,lab2,WMtot,'floats')
	# SO Argo
	# plot_map_region(lon_surfSO,lat_surfSO,lab2SO,WMtotSO,'SO_Argo')
	# 4. map of trained clusters
	# plot_map_GLODAP(lonC,latC,labTr,inN)
	# 5. trained and predict together
	# plot_TS_tr_pr(dataCl,data,WMtotTr)

	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# extract all the profiles that belong to each zone
	# 1. SOCCOM
	profilesnn = extract_profiles(1000,2000,lab2,id_surf,ID,raw_data,coor_data,oxy_data,no3_data,s0_data,seasons,dic_data,years_data,nn_surf,nnprof,dateNum,'SOCCOM')
	tenn       = profilesnn[0]
	sann       = profilesnn[1]
	oxynn      = profilesnn[2]
	no3nn      = profilesnn[3]
	dicnn      = profilesnn[4]
	latnn      = profilesnn[5]
	lonnn      = profilesnn[6]
	seasnn     = profilesnn[7]
	ididnn     = profilesnn[8]
	nnprnn     = profilesnn[9]
	yynn       = profilesnn[10]
	datesnn    = profilesnn[11]
	
	densnn     = gsw.sigma0(sann,tenn)
	# not sure why, but there are some negative densities..
	densnn[np.where(densnn<10)]=np.nan
	
	# plot o2 vs te (searching for the oxygen minimum)
	sann[np.where(sann<33)] =np.nan
	tenn[np.where(sann<33)] =np.nan
	oxynn[np.where(sann<33)] =np.nan
	no3nn[np.where(sann<33)] =np.nan
	dicnn[np.where(sann<33)] =np.nan
	latnn[np.where(sann<33)] =np.nan
	lonnn[np.where(sann<33)] =np.nan
	
	"""
	fig = plt.figure(figsize=(12,7))
	ax1 = plt.subplot(121)
	ax2 = plt.subplot(122)
	for ii in range(5):
         for jj in range(1000):
             try:
                 f1=sann[ii,jj,...][~np.isnan(sann[ii,jj,...])]
                 f2=tenn[ii,jj,...][~np.isnan(sann[ii,jj,...])] 
                 f3=oxynn[ii,jj,...][~np.isnan(sann[ii,jj,...])] 
                 ax1.plot(f1,f2,c=col[ii],linewidth=0.5,alpha=0.5)
                 ax2.plot(f3,f2,c=col[ii],linewidth=0.5,alpha=0.5)
             except:
                 print ii,jj
	cs  = ax1.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
	plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
	ax1.contour(ss,tt,s0,[27.68],colors='k',linestyles='solid',linewidths=1,alpha=0.8)
	ax1.set_ylim(-5,25)
	ax2.set_ylim(-5,25)
	ax1.plot([ss[0],ss[-1]],[1.5,1.5])
	ax1.plot([34.5,34.5],[tt[0],tt[-1]])
	ax2.grid("on")
	"""
	
	
	# plot all SOCCOM profiles by zone, together with the GLODAP mean by zone	
	fig2 = plt.figure(2)
	ax   = fig2.add_subplot(111)
	for ii in range(5):	
		for jj in range(1000):
			ax.plot(sann[ii,jj,:],tenn[ii,jj,:],c=col[ii],alpha=0.2,linewidth=0.2)
		i1,i2 = np.where(np.reshape(labTr,[len(latC[yC1:yC2]),len(lonC[xC1:xC2])])==ii+1)
		ax.plot(np.nanmean(saClim[:zC2+1,i1,i2],axis=1),np.nanmean(teClim[:zC2+1,i1,i2],axis=1),color=col[ii],alpha=0.8,linewidth=4,zorder=1000)
		ax.plot(np.nanmean(saClim[:zC2+1,i1,i2],axis=1),np.nanmean(teClim[:zC2+1,i1,i2],axis=1),c='k',alpha=1,linewidth=1,zorder=100000)
		ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=4)
		if ii == 4:
			ax.legend(loc=4,fancybox=True)	
		elif ii == 0:
			cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
			plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
	ax.set_xlim(np.min(ss),36)
	ax.set_ylim(-3,23)		
	ax.set_xlabel('SP', fontsize=12)
	ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_TS_SOCCOM_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	# 2. All IO Argo
	size1 = 1
	for ii in range(1,6):
		if len(lab2SO[np.where(lab2SO==ii)]) > size1:
			size1 = len(lab2SO[np.where(lab2SO==ii)])

	if save_argo_files:
		profilesnnSO = extract_profiles(size1,2000,lab2SO,id_surfSO,IDSO,raw_dataSO,coor_dataSO,[],[],s0_dataSO,seasonsSO,[],years_data,nn_surfSO,nnprofSO,dateFlNum,'Argo')
		tennSO     = profilesnnSO[0]
		sannSO     = profilesnnSO[1]
		latnnSO    = profilesnnSO[5]
		lonnnSO    = profilesnnSO[6]
		#seasnnSO   = profilesnnSO[7]
		ididnnSO   = profilesnnSO[8]
		nnprnnSO   = profilesnnSO[9]
		#yynnSO     = profilesnnSO[10]
		#datesnnSO  = profilesnnSO[11]
	
		outdir  = '/data/irosso/argo'
		fileout = os.path.join(outdir,'tennSO.data')
		tennSO.astype('>f4').tofile(fileout)
		fileout = os.path.join(outdir,'sannSO.data')
		sannSO.astype('>f4').tofile(fileout)
		fileout = os.path.join(outdir,'latnnSO.data')
		latnnSO.astype('>f4').tofile(fileout)
		fileout = os.path.join(outdir,'lonnnSO.data')
		lonnnSO.astype('>f4').tofile(fileout)
		fileout = os.path.join(outdir,'ididnnSO.data')
		ididnnSO.astype('>f4').tofile(fileout)
		fileout = os.path.join(outdir,'nnprnnSO.data')
		nnprnnSO.astype('>f4').tofile(fileout)
	else:
		outdir   = '/data/irosso/argo'
		fileout  = os.path.join(outdir,'tennSO.data')
		tennSO   = np.fromfile(fileout,'>f4')
		tennSO   = np.reshape(tennSO,[5,size1,2000])
		fileout  = os.path.join(outdir,'sannSO.data')
		sannSO   = np.fromfile(fileout,'>f4')
		sannSO   = np.reshape(sannSO,[5,size1,2000])
		fileout  = os.path.join(outdir,'latnnSO.data')
		latnnSO  = np.fromfile(fileout,'>f4')
		latnnSO  = np.reshape(latnnSO,[5,size1,2000])
		fileout  = os.path.join(outdir,'lonnnSO.data')
		lonnnSO  = np.fromfile(fileout,'>f4')
		lonnnSO  = np.reshape(lonnnSO,[5,size1,2000])
		#fileout  = os.path.join(outdir,'ididnnSO.data')
		#ididnnSO = np.fromfile(fileout,'>f4')
		#ididnnSO = np.reshape(ididnnSO,[5,size1,2000])
		#fileout  = os.path.join(outdir,'nnprnnSO.data')
		#nnprnnSO = np.fromfile(fileout,'>f4')
		#nnprnnSO = np.reshape(nnprnnSO,[5,size1,2000])
		
	# compute density
	densnnSO = gsw.sigma0(sannSO,tennSO)
	# not sure why, but there are some negative densities..
	densnnSO[np.where(densnnSO<10)]=np.nan	
	
	# adding this because somehow after label_argo() the labels became float numbers
	lab2SO  = np.array([int(ii) for ii in lab2SO])	
	
	# plot all Argo profiles by zone, together with the GLODAP mean by zone	
	fig2 = plt.figure(2)
	ax   = fig2.add_subplot(111)
	for ii in range(5):	
		for jj in range(0,size1,30):
			ax.plot(sannSO[ii,jj,:],tennSO[ii,jj,:],c=col[ii],alpha=0.2,linewidth=0.2)
		i1,i2 = np.where(np.reshape(labTr,[len(latC[yC1:yC2]),len(lonC[xC1:xC2])])==ii+1)
		ax.plot(np.nanmean(saClim[:zC2+1,i1,i2],axis=1),np.nanmean(teClim[:zC2+1,i1,i2],axis=1),color=col[ii],alpha=0.8,linewidth=4,zorder=1000)
		ax.plot(np.nanmean(saClim[:zC2+1,i1,i2],axis=1),np.nanmean(teClim[:zC2+1,i1,i2],axis=1),c='k',alpha=1,linewidth=1,zorder=100000)
		ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=4)
		if ii == 4:
			ax.legend(loc=4,fancybox=True)	
		elif ii == 0:
			cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
			plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
	ax.set_xlim(np.min(ss),36)
	ax.set_ylim(-3,25)		
	ax.set_xlabel('SP', fontsize=12)
	ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_TS_subset_Argo_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)

	# plot all lat/lon, colorcoded by year
	fig  = plt.figure(figsize=(10,5))
	ax   = fig.add_subplot(111)
	im   = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.2)
	plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
	cax  = fig.add_axes([0.15, 0.2, 0.2, 0.02])
	cbar = plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('[km]',fontsize=14)
	cbar.ax.xaxis.set_tick_params(color='k')
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')

	for ii,id in enumerate(np.unique(yynn[~np.isnan(yynn)])):
		#idx1,idx2 = np.where(yynn[:,:,0]==id)
		idx1W,idx2W = np.where(np.logical_and(yynn[:,:,0]==id,np.logical_or(seasnn[:,:,0]==1,seasnn[:,:,0]==2)))
		idx1C,idx2C = np.where(np.logical_and(yynn[:,:,0]==id,np.logical_or(seasnn[:,:,0]==3,seasnn[:,:,0]==4)))
		#im  = ax.scatter(lonnn[idx1,idx2,0],latnn[idx1,idx2,0],c=col[ii],marker='.',edgecolors='face',s=200,alpha=0.4,cmap='coolwarm')
		im  = ax.scatter(lonnn[idx1W,idx2W,0],latnn[idx1W,idx2W,0],c=col[ii],marker='o',edgecolors='gray',linewidth=0.3,s=50,alpha=0.4,cmap='coolwarm')
		im  = ax.scatter(lonnn[idx1C,idx2C,0],latnn[idx1C,idx2C,0],c=col[ii],marker='s',edgecolors='gray',linewidth=0.3,s=50,alpha=0.4,cmap='coolwarm')
		ax.plot(np.nan,np.nan,color=col[ii],linewidth=2,label='%i' %(int(id)))
	
	leg = ax.legend(loc=4,ncol=2,fancybox=True,fontsize=18)
	ax.add_artist(leg)
	leg.get_frame().set_alpha(0.7)
	ax.set_xlim(XC[x1],XC[x2])
	ax.set_ylim(YC[y1],YC[y2])
	
	im1, = ax.plot([np.nan],color='gray',marker='o',label='Spring + Summer',linewidth=1)
	im2, = ax.plot([np.nan],color='gray',marker='s',label='Autumn + Winter',linewidth=1)
	leg2 = ax.legend(loc=9,ncol=2,handles=[im1,im2],fancybox=True)
	leg2.get_frame().set_alpha(0.7)
	
	outfile = os.path.join(plotdir,'Indian_Ocean_map_profiles_years2.png')
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
		
	"""
	# checking the difference in nitrate
	for jj in range(1000):
		if np.any(np.abs(np.diff(no3nn[ii,jj,:]))>=2):
			print jj, np.diff(no3nn[ii,jj,:])
     	ff=no3nn[ii,jj,:]
		f1=ff[~np.isnan(ff)]
		if len(f1)!=0:
			f2=press[~np.isnan(ff)]
			plt.plot(f1,f2)
	"""
	
	# bin in density space
	#RHObin    = np.arange(24.64,27.7,0.01)
	RHObin    = np.arange(24.5,27.5,0.001)
	RHObin    = np.append(RHObin,np.arange(27.5,27.7,0.01))
	nbins     = len(RHObin)

	dirBin    = '/data/irosso/SOCCOM_floats_IO/binned_dens/'

	# GLODAP 	
	outfileSC = os.path.join(dirBin,'bin_ssCl2.data')
	outfileTC = os.path.join(dirBin,'bin_ttCl2.data')
	outfileOC = os.path.join(dirBin,'bin_oxyCl2.data')
	outfileNC = os.path.join(dirBin,'bin_no3Cl2.data')
	#outfileDC = os.path.join(dirBin,'bin_dicCl.data')
	
	# SOCCOM
	outfileS  = os.path.join(dirBin,'bin_ss2.data')
	outfileT  = os.path.join(dirBin,'bin_tt2.data')
	outfileO  = os.path.join(dirBin,'bin_oxy2.data')
	outfileN  = os.path.join(dirBin,'bin_no32.data')
	outfileD  = os.path.join(dirBin,'bin_dic2.data')
	
	# Argo
	outfileSSO = os.path.join(dirBin,'bin_ssSO.data')
	outfileTSO = os.path.join(dirBin,'bin_ttSO.data')
		
	# GLODAP
	densCl    = gsw.sigma0(saClim,teClim)
	if save_bin_dens_clim:
		bin_ttCl    = np.nan*np.ones((teClim.shape[1],teClim.shape[2],nbins))
		bin_ssCl    = np.nan*np.ones((teClim.shape[1],teClim.shape[2],nbins))
		bin_no3Cl   = np.nan*np.ones((teClim.shape[1],teClim.shape[2],nbins))	
		bin_oxyCl   = np.nan*np.ones((teClim.shape[1],teClim.shape[2],nbins))
		#bin_dicCl   = np.nan*np.ones((teClim.shape[0],teClim.shape[1],nbins))
		
		prop_nnCl   = [saClim,teClim,oxyClim,no3Clim]
		bin_ffCl    = [bin_ssCl,bin_ttCl,bin_oxyCl,bin_no3Cl]#,bin_dicCl]
		for ii in range(teClim.shape[1]):
			for jj in range(teClim.shape[2]):
				for pp in range(4):
					f0 = prop_nnCl[pp][:,ii,jj]
					f2 = densCl[:,ii,jj]
					f1 = f0[np.logical_and(~np.isnan(f0),~np.isnan(f2))]
					f2 = f2[np.logical_and(~np.isnan(f0),~np.isnan(f2))]
					# the following if is to avoid to extend further than the minimum of densCl
					if f2.count()!=0:
						dMin = np.min(np.where(RHObin>= np.nanmin(f2)))
						sp   = interpolate.interp1d(f2,f1,kind='slinear', bounds_error=False, fill_value=np.nan)
						bin_ffCl[pp][ii,jj,dMin:] = sp(RHObin[dMin:])
		
		# save binned files (it takes very long to bin the profiles)
		bin_ssCl.astype('>f4').tofile(outfileSC)
		bin_ttCl.astype('>f4').tofile(outfileTC)
		bin_oxyCl.astype('>f4').tofile(outfileOC)
		bin_no3Cl.astype('>f4').tofile(outfileNC)
		#bin_dic.astype('>f4').tofile(outfileDC)
	else:
		file = np.fromfile(outfileSC,dtype='>f4') 
		bin_ssCl = np.reshape(file,[teClim.shape[1],teClim.shape[2],nbins])
		file = np.fromfile(outfileTC,dtype='>f4') 
		bin_ttCl = np.reshape(file,[teClim.shape[1],teClim.shape[2],nbins])
		file = np.fromfile(outfileOC,dtype='>f4') 
		bin_oxyCl = np.reshape(file,[teClim.shape[1],teClim.shape[2],nbins])
		file = np.fromfile(outfileNC,dtype='>f4') 
		bin_no3Cl = np.reshape(file,[teClim.shape[1],teClim.shape[2],nbins])
		#file = np.fromfile(outfileDC,dtype='>f4') 
        #bin_dicCl = np.reshape(file,[teClim.shape[1],teClim.shape[2],nbins])
        
	# SOCCOM
	prop_nn   = [sann,tenn,oxynn,no3nn,dicnn]
	if save_bin_dens_floats:
		bin_tt    = np.nan*np.ones((5,1000,nbins))
		bin_ss    = np.nan*np.ones((5,1000,nbins))
		bin_no3	  = np.nan*np.ones((5,1000,nbins))	
		bin_oxy	  = np.nan*np.ones((5,1000,nbins))
		bin_dic   = np.nan*np.ones((5,1000,nbins))

		bin_ff    = [bin_ss,bin_tt,bin_oxy,bin_no3,bin_dic]
		for ii in range(5):
			for jj in range(1000):
				for pp in range(5):
					f0 = prop_nn[pp][ii,jj,:]
					f2 = densnn[ii,jj,:]
					f1 = f0[np.logical_and(~np.isnan(f0),~np.isnan(f2))]
					f2 = f2[np.logical_and(~np.isnan(f0),~np.isnan(f2))]
					# the following is to get rid of some densities that are negative!!!!
					f2[f2<RHObin[0]]=np.nan
					dm = np.ma.masked_invalid(f2)
					if dm.count()!=0 and len(f1)!=0:
						dMin = np.min(np.where(RHObin>= np.nanmin(dm)))#densnn[ii,jj,:])))
						sp   = interpolate.interp1d(f2,f1,kind='slinear', bounds_error=False, fill_value=np.nan)
						bin_ff[pp][ii,jj,dMin:] = sp(RHObin[dMin:])

		# save binned files (it takes very long to bin the profiles)
		bin_ss.astype('>f4').tofile(outfileS)
		bin_tt.astype('>f4').tofile(outfileT)
		bin_oxy.astype('>f4').tofile(outfileO)
		bin_no3.astype('>f4').tofile(outfileN)
		bin_dic.astype('>f4').tofile(outfileD)

	else:
		file = np.fromfile(outfileS,dtype='>f4') 
		bin_ss = np.reshape(file,[5,1000,nbins])
		file = np.fromfile(outfileT,dtype='>f4') 
		bin_tt = np.reshape(file,[5,1000,nbins])
		file = np.fromfile(outfileO,dtype='>f4') 
		bin_oxy = np.reshape(file,[5,1000,nbins])
		file = np.fromfile(outfileN,dtype='>f4') 
		bin_no3 = np.reshape(file,[5,1000,nbins])
		file = np.fromfile(outfileD,dtype='>f4') 
		bin_dic = np.reshape(file,[5,1000,nbins])
        
	# Argo
	prop_nnSO = [sannSO,tennSO]
	if save_bin_dens_argo:
		bin_ttSO    = np.nan*np.ones((5,size1,nbins))
		bin_ssSO    = np.nan*np.ones((5,size1,nbins))
	
		bin_ffSO    = [bin_ssSO,bin_ttSO]
		for ii in range(5):
			for jj in range(size1):
				for pp in range(2):
					f0 = prop_nnSO[pp][ii,jj,:]
					f2 = densnnSO[ii,jj,:]
					f1 = f0[np.logical_and(~np.isnan(f0),~np.isnan(f2))]
					f2 = f2[np.logical_and(~np.isnan(f0),~np.isnan(f2))]
					# the following is to get rid of some densities that are negative!!!!
					f2[f2<RHObin[0]]=np.nan
					dm = np.ma.masked_invalid(f2)
					if dm.count()!=0 and len(f1)!=0:
						dMin = np.min(np.where(RHObin>= np.nanmin(dm)))#densnn[ii,jj,:])))
						sp   = interpolate.interp1d(f2,f1,kind='slinear', bounds_error=False, fill_value=np.nan)
						bin_ffSO[pp][ii,jj,dMin:] = sp(RHObin[dMin:])
		
		# save binned files (it takes very long to bin the profiles)
		bin_ssSO.astype('>f4').tofile(outfileSSO)
		bin_ttSO.astype('>f4').tofile(outfileTSO)
	
	else:
		file = np.fromfile(outfileSSO,dtype='>f4') 
		bin_ssSO = np.reshape(file,[5,size1,nbins])
		file = np.fromfile(outfileTSO,dtype='>f4') 
		bin_ttSO = np.reshape(file,[5,size1,nbins])
		
    # mask binned files
    # GLODAP
	bin_ttCl  = np.ma.masked_equal(bin_ttCl,0.)
	bin_ssCl  = np.ma.masked_equal(bin_ssCl,0.)
	bin_oxyCl = np.ma.masked_equal(bin_oxyCl,0.)
	bin_no3Cl = np.ma.masked_equal(bin_no3Cl,-999.)
	#bin_dicCl = np.ma.masked_equal(bin_dicCl,0.)
	bin_ffCl  = [bin_ssCl,bin_ttCl,bin_oxyCl,bin_no3Cl]#,bin_dicCl]
	# SOCCOM
	bin_tt  = np.ma.masked_equal(bin_tt,0.)
	bin_ss  = np.ma.masked_equal(bin_ss,0.)
	bin_oxy = np.ma.masked_equal(bin_oxy,0.)
	bin_no3 = np.ma.masked_equal(bin_no3,0.)
	bin_dic = np.ma.masked_equal(bin_dic,0.)
	bin_ff  = [bin_ss,bin_tt,bin_oxy,bin_no3,bin_dic]
	# Argo
	bin_ttSO  = np.ma.masked_equal(bin_ttSO,0.)
	bin_ssSO  = np.ma.masked_equal(bin_ssSO,0.)
	bin_ffSO  = [bin_ssSO,bin_ttSO]
	
	sys.exit()

	fig = plt.figure(figsize=(15,7))
	ax1 = plt.subplot(131)
	ax2 = plt.subplot(132)
	ax3 = plt.subplot(133)
	message = 'Nothing'
	for ii in range(5):
         for jj in range(1000):
             try:
                 f1 = sann[ii,jj,...][~np.isnan(sann[ii,jj,...])]
                 f2 = tenn[ii,jj,...][~np.isnan(sann[ii,jj,...])] 
                 f3 = oxynn[ii,jj,...][~np.isnan(sann[ii,jj,...])] 
                 f4 = bin_oxy[ii,jj,...][~np.isnan(bin_tt[ii,jj,...])] 
                 f5 = RHObin[~np.isnan(bin_tt[ii,jj,...])] 
                 ax1.plot(f1,f2,c=col[ii],linewidth=0.5,alpha=0.5)
                 ax2.plot(f3,f2,c=col[ii],linewidth=0.5,alpha=0.5)
                 ax3.plot(f4,f5,c=col[ii],linewidth=0.5,alpha=0.5)
             except:
                 message='Something'
	cs  = ax1.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
	plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
	ax1.contour(ss,tt,s0,[27.3],colors='k',linestyles='solid',linewidths=1,alpha=0.8)
	ax1.set_ylim(-5,25)
	ax2.set_ylim(-5,25)
	ax1.plot([ss[0],ss[-1]],[1.5,1.5])
	ax1.plot([34.5,34.5],[tt[0],tt[-1]])
	ax3.plot([150,400],[27.2,27.2])
	ax3.invert_yaxis()
	ax2.grid("on")
	ax3.grid("on")
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_TSO2.png')
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	plt.show()
	
	# search oxygen minimum (for rho >27.2) [UCDW]
	message = 'Nothing'
	fig,ax  = plt.subplots(figsize=(12,7))
	im      = ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
	ax.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')
	ax.set_xlim(XC[x1],XC[x2])
	ax.set_ylim(YC[y1],YC[y2])
	ax.set_title('O$_2$ minimum depth',fontsize=16)
	cax     = fig.add_axes([0.15, 0.15, 0.2, 0.01])
	cbar    = plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('[km]')
	cbar.ax.xaxis.set_tick_params(color='k')
	for ii in range(5):
		idr1,idr2 = np.where(densnn[ii,...]>27.3)
		for jj,idd in enumerate(np.unique(idr1)):
			try:
				i_pr    = np.where(densnn[ii,idd,:]>27.3)
				ii_omin = np.nanargmin(oxynn[ii,idd,i_pr])
				#plt.scatter(oxynn[ii,idd,i_pr][0,ii_omin],press[i_pr,0][0,ii_omin],c=col[ii])
				ax.scatter(lonnn[ii,idd,i_pr][0,ii_omin],latnn[ii,idd,i_pr][0,ii_omin],
					c=press[i_pr,0][0,ii_omin],marker='.',facecolor='None',edgecolors=col[ii],s=400,
					alpha=0.2,zorder=10000)
				im = ax.scatter(lonnn[ii,idd,i_pr][0,ii_omin],latnn[ii,idd,i_pr][0,ii_omin],
					c=press[i_pr,0][0,ii_omin],marker='.',edgecolors='None',s=400,alpha=0.9,
					cmap=past.mpl_colormap,zorder=10000)
				im.set_clim(0,2000)
			except:
				message='Something'
				#print np.nanmin(oxynn[ii,idd,np.where(densnn[ii,idd,:]>27.2)])
		
	cax     = fig.add_axes([0.65, 0.15, 0.2, 0.01])
	cbar    = plt.colorbar(im,cax=cax,orientation='horizontal',ticks=np.linspace(0,2000,5))#,extend='both'
	cbar.ax.set_xlabel('pressure [db]')
	cbar.ax.xaxis.set_tick_params(color='k')
						
	# plot O2, NO3 concentration along longitude, on an isopycnal or depth (pressure) range
	# depth (pressure) range
	maxDB  = np.min(np.where(press[:,0]>=100))
	flxnn  = load_CO2_flx('6hr') #'month' or '6hr'
	ff_nn  = [sann,tenn,oxynn,no3nn,dicnn]
		
	####
	# !!!!!!!!!!!!!!!!!!!!!!!!!! there's still something weird with the profiles from the fluxes!!!!! 
	# probably it's better if I use the pCO2 from the floats file...
	ylims  = [[33.5,36.5],[-5,25],[160,400],[-2,40]]
	
	plt.figure(1,figsize=(10,10))
	ax1    = plt.subplot(411)
	ax2    = plt.subplot(412)
	ax3    = plt.subplot(413)
	ax4    = plt.subplot(414)
	axx    = [ax1,ax2,ax3,ax4]
	
	plt.figure(2,figsize=(8,5))
	ax5=plt.subplot(111)
	ax5.yaxis.grid(which="major")#, color='r', linestyle='-', linewidth=2)
	#ax5    = plt.axes([0.77, 0.8, 0.1, 0.1 ])
	#ax5.set_frame_on(False)
	#ax5.get_yaxis().tick_left()
	#ax5.axes.get_yaxis().set_visible(True)
	#ax5.axes.get_xaxis().set_visible(True)
	#ax5.grid('on')			
	
	xpos = 0
	xdata = [1.2,2.2,3.4,4.2]
	for ii in range(4):
		# extract flux by zone
		idZ    = np.where(flxnn[5,...]==ii+1)[0][:]
		flxtot = flxnn[1,...][idZ]
		# extract regions and compute mean and std
		flxW   = np.where(flxnn[2,...][idZ]<=40.)[0][:]
		flxU   = np.where(np.logical_and(flxnn[2,...][idZ]>40.,flxnn[2,...][idZ]<=68.))[0][:]
		flxD   = np.where(np.logical_and(flxnn[2,...][idZ]>68.,flxnn[2,...][idZ]<120.))[0][:]
		flxE   = np.where(flxnn[2,...][idZ]>=120.)[0][:]
		
		flx1   = np.nanmean(flxtot[flxW])
		flx2   = np.nanmean(flxtot[flxU])
		flx3   = np.nanmean(flxtot[flxD])
		flx4   = np.nanmean(flxtot[flxE])
		err1   = np.nanstd(flxtot[flxW])
		err2   = np.nanstd(flxtot[flxU])
		err3   = np.nanstd(flxtot[flxD])
		err4   = np.nanstd(flxtot[flxE])

		ax5.bar(1+xpos, flx1, facecolor=col[ii],yerr=err1,ecolor='black',alpha=0.5,edgecolor=col[ii], width=0.2, align='center',label='%s' %(labZ[0]),linewidth=4)
		ax5.bar(2+xpos, flx2, facecolor=col[ii],yerr=err2,ecolor='black',alpha=0.5,edgecolor=col[ii], width=0.2,align='center',linewidth=4)
		ax5.bar(3+xpos, flx3, facecolor=col[ii],yerr=err3,ecolor='black',alpha=0.5,edgecolor=col[ii], width=0.2,align='center',linewidth=4)
		ax5.bar(4+xpos, flx4, facecolor=col[ii],yerr=err4,ecolor='black',alpha=0.5,edgecolor=col[ii], width=0.2,align='center',linewidth=4)

		ax5.arrow(4.7,-0.2,0,-5,fc='b', ec="b",head_width=0.1,head_length=0.1)
		ax5.arrow(4.7,0.2,0,5,fc='r', ec="r",head_width=0.1,head_length=0.1)
		ax5.text(4.8, 4, 'outgassing', rotation=90, color='r', fontsize=18)
		ax5.text(4.8, -2, 'uptake', rotation=90, color='b', fontsize=18)
		
		ax5.fill_between([0.7,1.8],-6,6,facecolor='gray', alpha=0.2)
		ax5.fill_between([2.8,3.8],-6,6,facecolor='gray', alpha=0.2)
		ax5.set_ylim(-6,6)
		ax5.set_xlim(0.7,5)
		ax5.set_xticks(xdata)
        ax5.set_xticklabels(regTit,fontsize=18)
        ax5.set_ylabel('mean CO$_2$ flux [mol m$^{-2}$ y$^{-1}$]', fontsize=18)
		xpos += 0.2						
		#ax1.scatter(flxnn[2,np.where(flxnn[0,:]==ff)[0][:]][jjFl],flxnn[1,np.where(flxnn[0,:]==ff)[0][:]][jjFl],color=col[ii],s=50,marker='o',edgecolors='face',alpha=0.4)		

		listFlnn = np.unique(ididnn[ii,...][~np.isnan(ididnn[ii,...])])
		for ff in listFlnn:
			idxFlnn = np.where(ididnn[ii,:,0]==ff)[0][:]			
			for pp in range(4):
				fpre   = ff_nn[pp][ii,idxFlnn,:maxDB+1]
				f0     = np.nanmean(fpre,axis=1)
				f1     = lonnn[ii,idxFlnn,0]
				try:
					f2     = f0[~np.isnan(f0)]
					f3     = f1[~np.isnan(f0)]
					axx[pp].scatter(f3,f2,c=col[ii],s=50,marker='o',edgecolors='face',alpha=0.2)#,zorder=10000)
					if ii==0 and ff == listFlnn[-1]:
						axx[pp].fill_between([0,40],ylims[pp][0],ylims[pp][1],facecolor='gray', alpha=0.3)
						axx[pp].fill_between([68,120],ylims[pp][0],ylims[pp][1], facecolor='gray', alpha=0.3)
						axx[pp].grid('on')
						axx[pp].set_xlim(0,180)
						axx[pp].set_ylim(ylims[pp][0], ylims[pp][1])

				except:
					print 'no data for float #', ff, ' in zone ',ii+1
				"""
				if pp in [0,1]:
					# plot all IO Argo:
					ffSO   = prop_nnSO[pp][ii,:,:maxDB+1]
					f0     = np.nanmean(ffSO,axis=1)
					f1     = lonnnSO[ii,:,0]
					f2     = f0[~np.isnan(f0)]
					f3     = f1[~np.isnan(f0)]
					axx[pp].scatter(f3,f2,c=col[ii],s=10,marker='o',edgecolors='None',alpha=0.1)
				"""
	ax4.set_xlabel(r'longitude [$^{\circ}$E]', fontsize=16)
	ax2.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=16)
	ax3.set_ylabel(r'O$_2$ [$\mu$mol kg$^{-1}$]', fontsize=16)
	ax4.set_ylabel(r'NO$_3$[$\mu$mol kg$^{-1}$]', fontsize=16)
	
	ax1.text(5, 36, 'West', rotation=0, color='k', fontsize=18)
	ax1.text(40, 36, 'Upstream', rotation=0, color='k', fontsize=18)
	ax1.text(75, 36, 'Downstream', rotation=0, color='k', fontsize=18)
	ax1.text(125, 36, 'East', rotation=0, color='k', fontsize=18)
	
	im1, = ax4.plot([np.nan],color=col[0],marker='o',label='%s' %labZ[0],linewidth=1)
	im2, = ax4.plot([np.nan],color=col[1],marker='o',label='%s' %labZ[1],linewidth=1)
	im3, = ax4.plot([np.nan],color=col[2],marker='o',label='%s' %labZ[2],linewidth=1)
	im4, = ax4.plot([np.nan],color=col[3],marker='o',label='%s' %labZ[3],linewidth=1)
	leg  = ax4.legend(loc=8,ncol=2,handles=[im1,im2,im3,im4],fancybox=True)
	leg.get_frame().set_alpha(0.5)
	
	plt.figure(1,figsize=(10,10))
	#plt.suptitle('Property variability (surface CO$_2$ flux and property averaged over the top 100 db)', fontsize=16)
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_SOCCOM_Argo_prop_top_lon_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	plt.figure(2,figsize=(8,5))
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_prop_6hr_flux_regions_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	####
	# density range
	ylims  = [[34.,34.7],[0,9],[160,300],[25,35]]
	rhomin = np.min(np.where(RHObin>=26.9))
	rhomax = np.min(np.where(RHObin>=27.2))

	plt.figure(figsize=(10,10))
	ax1    = plt.subplot(411)
	ax2    = plt.subplot(412)
	ax3    = plt.subplot(413)
	ax4    = plt.subplot(414)
	axx    = [ax1,ax2,ax3,ax4]
	for ii in range(2):
		for pp in range(4):
			ff     = bin_ff[pp][ii,:,rhomin:rhomax]
			f0     = np.nanmean(ff,axis=1)
			f1     = lonnn[ii,:,0]
			f2     = f0[~np.isnan(f0)]
			f3     = f1[~np.isnan(f0)]
			f4     = seasnn[ii,:,0]
			f4     = f4[~np.isnan(f0)]
			idxW   = np.where(np.logical_or(f4==1,f4==2))[0][:]
			idxC   = np.where(np.logical_or(f4==3,f4==4))[0][:]
			axx[pp].scatter(f3,f2,c=col[ii],s=100,marker='o',edgecolors='k',alpha=0.3,zorder=100000)
			#axx[pp].scatter(f3[idxW],f2[idxW],c=col[ii],s=100,marker='o',edgecolors='k',alpha=0.3)
			#axx[pp].scatter(f3[idxC],f2[idxC],c=col[ii],s=100,marker='*',edgecolors='k',alpha=0.3)
			
			if pp in [0,1]:
				# plot all IO Argo:
				ff     = bin_ffSO[pp][ii,:,rhomin:rhomax]
				f0     = np.nanmean(ff,axis=1)
				f1     = lonnnSO[ii,:,0]
				f2     = f0[~np.isnan(f0)]
				f3     = f1[~np.isnan(f0)]
				axx[pp].scatter(f3,f2,c=col[ii],s=10,marker='o',edgecolors='None',alpha=0.1)
			if ii==0:
				axx[pp].fill_between([0,40],ylims[pp][0],ylims[pp][1],facecolor='gray', alpha=0.3)
				axx[pp].fill_between([68,120],ylims[pp][0],ylims[pp][1],facecolor='gray', alpha=0.3)
				axx[pp].grid('on')
				axx[pp].set_xlim(0,180)
				axx[pp].set_xticks(np.arange(0,200,20))
				axx[pp].set_xlabel(r'longitude [$^{\circ}$E]', fontsize=16)
				axx[pp].set_ylim(ylims[pp][0],ylims[pp][-1])
	
	ax1.text(5, 34.6, 'West', rotation=0, color='k', fontsize=18)
	ax1.text(40, 34.6, 'Upstream', rotation=0, color='k', fontsize=18)
	ax1.text(75, 34.6, 'Downstream', rotation=0, color='k', fontsize=18)
	ax1.text(125, 34.6, 'East', rotation=0, color='k', fontsize=18)
	
	#ax1 = plt.subplot(412)
	#im1, = plt.plot([np.nan],color='k',marker='o',label='Spring + Summer',linewidth=0.3)
	#im2, = plt.plot([np.nan],color='k',marker='*',label='Autumn + Winter',linewidth=0.3)
	#leg  = plt.legend(loc=4,handles=[im1,im2],fancybox=True)
	#leg.get_frame().set_alpha(0.5)
	
	ax1 = plt.subplot(414)
	im1, = plt.plot([np.nan],color='r',marker='o',label='Subtropical Zone',linewidth=1)
	im2, = plt.plot([np.nan],color='c',marker='o',label='Subantarctic Zone',linewidth=1)
	leg  = plt.legend(loc=8,ncol=2,handles=[im1,im2],fancybox=True)
	leg.get_frame().set_alpha(0.5)
	
	ax2.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=16)
	ax3.set_ylabel(r'O$_2$ [$\mu$mol kg$^{-1}$]', fontsize=16)
	ax4.set_ylabel(r'NO$_3$[$\mu$mol kg$^{-1}$]', fontsize=16)
	#plt.suptitle('Property variability averaged over 26.9-27.2 kg m$^{-3}$ isopycnals', fontsize=16)
		
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_prop_AAIW_SOCCOM_Argo_lon_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	# plot the same for gridded GLODAPv2
	labgr = np.reshape(labTr,[len(latC[yC1:yC2]),len(lonC[xC1:xC2])])
	labm  = np.ma.masked_where(np.ma.getmask(dataTT),labgr)
	ylims  = [[34.,34.7],[2,8],[160,300],[24,32]]

	plt.figure(figsize=(12,15))
	ax1    = plt.subplot(411)
	ax2    = plt.subplot(412)
	ax3    = plt.subplot(413)
	ax4    = plt.subplot(414)
	axx    = [ax1,ax2,ax3,ax4]
	for ii in range(2):
		for pp in range(4):
			ii1,ii2 = np.where(labm==ii+1)
			f0     = np.nanmean(bin_ffCl[pp][ii1,ii2,rhomin:rhomax],axis=1)
			f0[f0==-999.] = np.nan
			f1     = lonC[xC1:xC2][ii2]
			f2     = f0[~np.isnan(f0)]
			f3     = f1[~np.isnan(f0)]
			axx[pp].scatter(f3,f2,c=col[ii],s=100,marker='o',edgecolors='k',alpha=0.3)
			if ii==0:
				axx[pp].fill_between([0,40],ylims[pp][0],ylims[pp][1],facecolor='gray', alpha=0.3)
				axx[pp].fill_between([68,120],ylims[pp][0],ylims[pp][1], facecolor='gray', alpha=0.3)
				axx[pp].grid('on')
				axx[pp].set_xlim(0,180)
				axx[pp].set_xticks(np.arange(0,200,20))
				axx[pp].set_xlabel(r'longitude [$^{\circ}$E]', fontsize=12)
				axx[pp].set_ylim(ylims[pp][0],ylims[pp][-1])
	
	ax1.text(5, 34.75, 'West', rotation=0, color='k', fontsize=14)
	ax1.text(45, 34.75, 'Upstream', rotation=0, color='k', fontsize=14)
	ax1.text(75, 34.75, 'Downstream', rotation=0, color='k', fontsize=14)
	ax1.text(125, 34.75, 'East', rotation=0, color='k', fontsize=14)
	
	ax2.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
	ax3.set_ylabel(r'O$_2$ [$\mu$mol kg$^{-1}$]', fontsize=12)
	ax4.set_ylabel(r'NO$_3$[$\mu$mol kg$^{-1}$]', fontsize=12)
	plt.suptitle('Property variability averaged over 26.9-27.2 kg m$^{-3}$ isopycnals (gridded GLODAPv2)', fontsize=16)
		
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_GLODAP_prop_AAIW_lon_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	
	# plot map of depth for isopycnals 26.8 and 27.2
	from matplotlib.colors import LinearSegmentedColormap
	colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)] 
	mine = LinearSegmentedColormap.from_list('mine',colors,N=6)
	
	fig     = plt.figure(100,figsize=(12,15))
	ax1     = fig.add_subplot(211)
	ax2     = fig.add_subplot(212)
	axx     = [ax1,ax2]
	titles  = [r'Depth of isopycnal $\sigma_0=26.9$ kg m$^{-3}$',r'Depth of isopycnal $\sigma_0=27.2$ kg m$^{-3}$']
	for jj in range(2):
		if jj == 0:
			isop = 26.9
		else:
			isop = 27.2
		ax      = axx[jj]
		im      = ax.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
		ax.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
		if jj == 0:
			cax     = fig.add_axes([0.15, 0.15, 0.2, 0.01])
			cbar    = plt.colorbar(im,cax=cax,orientation='horizontal')
			cbar.ax.set_xlabel('[km]')
			cbar.ax.xaxis.set_tick_params(color='k')
		plot_Orsi(fig,ax,'k')
		plot_PF(fig,ax,'k')

		for ii in range(2):
			pr1,pr2 = np.where(densnn[ii,:,:]>=isop)
			for pp in (np.unique(pr1)):
				pr2 = np.min(np.where(densnn[ii,pp,:]>=isop))
				im  = ax.scatter(lonnn[ii,pp,0],latnn[ii,pp,0],c=press[pr2,0],marker='.',edgecolors='face',s=200,alpha=0.6,cmap=mine)#cm.deep)
				im.set_clim(0,1500)
		if jj == 1:
			cax     = fig.add_axes([0.6, 0.15, 0.25, 0.01])
			cbar    = plt.colorbar(im,cax=cax,orientation='horizontal',extend='both')
			cbar.ax.set_xlabel('[m]')
			cbar.ax.xaxis.set_tick_params(color='k')
			cbar.set_ticks(np.linspace(0,1500,7))
			
		ax.set_xlim(XC[x1],XC[x2])
		ax.set_ylim(YC[y1],YC[y2])
		ax.set_title(r'Depth of isopycnal $\sigma_0=$%.1f kg m$^{-3}$' %(isop))
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_SOCCOM_floats_depth_isopycnals.png')
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	zoneM = ['o','>']
	plt.figure(figsize=(12,7))
	ax    = plt.subplot(111)
	for ii in range(2):
		pr1,pr2 = np.where(np.logical_and(densnn[ii,...]>=26.9,densnn[0,...]<=27.2))
		for pp in np.unique(pr1):
			pr2 = np.where(np.logical_and(densnn[ii,pp,:]>=26.9,densnn[0,pp,:]<=27.2))
			press1 = np.min(pr2)
			press2 = np.max(pr2)
			print pp,ididnn[ii,pp,0],lonnn[ii,pp,0],densnn[ii,pp,press1]
			plt.scatter(lonnn[ii,pp,0],press[press1,0],c=col[ii],marker=zoneM[0],edgecolors='face',alpha=0.5,s=100)
			plt.scatter(lonnn[ii,pp,0],press[press2,0],c=col[ii],marker=zoneM[1],edgecolors='face',alpha=0.5,s=100)
			plt.plot([lonnn[ii,pp,0],lonnn[ii,pp,0]],[press[press1,0],press[press2,0]],'k',linewidth=0.3,alpha=0.4)
	ax.invert_yaxis()
	ax.grid('on')
	ax.set_xlim(0,180)
	ax.set_xlabel(r'longitude [$^{\circ}$E]', fontsize=12)
	ax.set_ylabel(r'pressure [db]', fontsize=12)
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_SOCCOM_floats_depth_isopycnals_2.png')
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	# check spikes on fields vs density
	plt.figure(figsize=(12,10))
	ax1=plt.subplot(231)
	ax2=plt.subplot(232)
	ax3=plt.subplot(233)
	ax4=plt.subplot(234)
	ax5=plt.subplot(235)
	ax6=plt.subplot(236)
	for ii in range(4):
		for jj in range(1000):
			f0  = bin_tt[ii,jj,:]
			f1  = f0[~np.isnan(f0)]
			idx = np.where(np.abs(np.diff(f1))>=1)[0][:]
			if len(idx)!=0:
				print ii, jj, ididnn[ii,jj,0]
				f2 = RHObin
				f2 = f2[~np.isnan(f0)]
				f3 = bin_ss[ii,jj,:][[~np.isnan(f0)]]
				ax3.plot(f1,f2,color=col[ii],linewidth=2,alpha=0.8)
				ax6.plot(f1,f2,color=col[ii],linewidth=2,alpha=0.8)
				f0 = tenn[ii,jj,:]
				f3 = sann[ii,jj,:][[~np.isnan(f0)]]
				f2 = densnn[ii,jj,:]
				f1 = f0[~np.isnan(f0)]
				f2 = f2[~np.isnan(f0)]
				fn = gsw.sigma0(f3,f1)
				ax2.plot(f1,f2,color=col[ii],linewidth=2,alpha=0.8)
				ax5.plot(f3,f2,color=col[ii],linewidth=2,alpha=0.8)
				f2 = press[:,0]
				f2 = f2[~np.isnan(f0)]
				ax1.plot(f1,f2,color=col[ii],linewidth=2,alpha=0.8,label='%i #%i' %(int(ididnn[ii,jj,0]),int(nnprnn[ii,jj,0])))
				ax4.plot(f3,f2,color=col[ii],linewidth=2,alpha=0.8)

	ax1.legend(loc=4, fancybox=True)
	ax2.set_title('raw temperature (vs density)')
	ax3.set_title('interpolated temperature')
	ax1.set_title('raw temperature (vs pressure)')
	ax5.set_title('raw salinity (vs density)')
	ax6.set_title('interpolated salinity')
	ax4.set_title('raw salinity (vs pressure)')
	ax1.invert_yaxis()	
	ax2.invert_yaxis()	
	ax3.invert_yaxis()	
	ax4.invert_yaxis()
	ax5.invert_yaxis()	
	ax6.invert_yaxis()	
	ax1.set_xlabel(r'$\theta$ [$^{\circ}$C]')
	ax2.set_xlabel(r'$\theta$ [$^{\circ}$C]')
	ax3.set_xlabel(r'$\theta$ [$^{\circ}$C]')
	ax1.set_ylabel(r'$\sigma_0$ [kg m$^{-3}$]')
	ax4.set_ylabel('pressure [db]')

	outfile = os.path.join(plotdir,'raw_binned_density.png')
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	plt.show()

	# plot GLODAP density versus floats density:
	lonCl     = lonC[xC1:xC2]
	latCl     = latC[yC1:yC2]
	
	fig,ax    = plt.subplots(figsize=(12,7))
	im1       = ax.pcolor(lonCl,latCl,densCl[0,...],cmap=cm.deep,alpha=1)
	im1.set_clim(23,27)
	cbar      = plt.colorbar(im1)
	cbar.ax.set_title(r'$\sigma_0$')
	cbar.ax.set_xlabel('kg m$^{-3}$')
	for ii in range(5):
		for jj in range(1000):
			try:
				d1 = densnn[ii,jj,:][~np.isnan(densnn[ii,jj,:])][0]
				im = ax.scatter(lonnn[ii,jj,0],latnn[ii,jj,0],c=d1,marker='o',edgecolors='None',s=100,alpha=1,cmap=cm.deep)
				im.set_clim(23,27)
				im = ax.scatter(lonnn[ii,jj,0],latnn[ii,jj,0],c='None',marker='o',edgecolors='w',s=100,alpha=0.2)
			except:
				print 'no density here: ',ii,jj
	plot_PF(fig,ax,'y')
	plot_Orsi(fig,ax,'y')			
	ax.set_xlim(lonC[xC1],lonC[xC2-1])
	ax.set_ylim(latC[yC1],latC[yC2-1])
	
	plt.title('GLODAPv2 and floats surface density')
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_dens_GLODAP_floats_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
			
	plt.show()	
	
	# plot GLODAP no3 averaged over the top 50 m:
	topNO3    = np.nanmean(no3Clim[:5,...],axis=0)
	fig,ax    = plt.subplots(figsize=(12,7))
	im1       = ax.pcolor(lonCl,latCl,topNO3,cmap=cm.deep,alpha=1)
	im1.set_clim(0,30)
	cbar      = plt.colorbar(im1)
	cbar.ax.set_title('$NO_3$')
	cbar.ax.set_xlabel(r'%s' %(xlab[2]))
	plot_PF(fig,ax,'y')
	plot_Orsi(fig,ax,'y')			
	ax.set_xlim(lonC[xC1],lonC[xC2-1])
	ax.set_ylim(latC[yC1],latC[yC2-1])
	
	plt.title('GLODAPv2: top 50 m NO$_3$')
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_GLODAP_NO3_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
			
	plt.show()	
	
	# top 50 mean of floats no3, divided into warm (spr+sum) and cold (aut+win) months 
	topNO3nn  = np.nanmean(no3nn[:,:,:99],axis=2)
	"""
	fig       = plt.figure(figsize=(12,12))
	seastit   = ['Spring + Summer','Autumn + Winter']
	for jj in range(1,3):
		ntot   = 0
		ax     = plt.subplot(2,1,jj)
		im     = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.2)
		plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k',alpha=0.3)
		if jj == 2:
			cax    = fig.add_axes([0.15, 0.15, 0.15, 0.01])
			cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
			cbar.ax.set_xlabel('ocean depth [km]')
			cbar.ax.xaxis.set_tick_params(color='k')
		plot_Orsi(fig,ax,'k')
		plot_PF(fig,ax,'k')
		for ii in range(5):
			if jj == 1:
				iSeas  = np.where(np.logical_or(seasnn[ii,:,0]==1,seasnn[ii,:,0]==2))[0][:]
			else:
				iSeas  = np.where(np.logical_or(seasnn[ii,:,0]==3,seasnn[ii,:,0]==4))[0][:]
			ntot= ntot + len(topNO3nn[ii,iSeas][~np.isnan(topNO3nn[ii,iSeas])])
			im1 = ax.scatter(lonnn[ii,iSeas,0],latnn[ii,iSeas,0],c=topNO3nn[ii,iSeas],marker='o',edgecolors='None',s=100,alpha=1,cmap=cm.deep)
			im1.set_clim(0,30)
			#ax.scatter(lonnn[ii,iSeas,0],latnn[ii,iSeas,0],c='None',marker='o',edgecolors='w',s=100,alpha=0.2)
			if jj == 1 and ii == 0:
				cax    = fig.add_axes([0.15, 0.6, 0.15, 0.01])
				cbar   = plt.colorbar(im1,cax=cax,orientation='horizontal')
				cbar.ax.set_xlabel(r'NO$_3$ %s' %(xlab[2]))
				cbar.ax.xaxis.set_tick_params(color='k')
				cbar.set_ticks(np.linspace(0,30,6))  
			
		ax.set_xlim(XC[x1],XC[x2])
		ax.set_ylim(YC[y1],YC[y2])
		ax.set_title('%s. N$_{prof}=$%i' %(seastit[jj-1],ntot))
	plt.suptitle('Top 50 m average of floats NO$_3$', fontsize=14)
	"""
	fig       = plt.figure(figsize=(12,7))
	labSeas   = ['Spring + Summer','Autumn + Winter']
	ntotW = ntotC = 0
	ax     = plt.subplot(111)
	im     = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.2)
	plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k',alpha=0.3)
	cax    = fig.add_axes([0.15, 0.2, 0.2, 0.01])
	cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('[km]')
	cbar.ax.xaxis.set_tick_params(color='k')
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')
	for ii in range(5):
		iW  = np.where(np.logical_or(seasnn[ii,:,0]==1,seasnn[ii,:,0]==2))[0][:]
		iC  = np.where(np.logical_or(seasnn[ii,:,0]==3,seasnn[ii,:,0]==4))[0][:]
		ntotW = ntotW + len(topNO3nn[ii,iW][~np.isnan(topNO3nn[ii,iW])])
		ntotC = ntotC + len(topNO3nn[ii,iC][~np.isnan(topNO3nn[ii,iC])])
		im1 = ax.scatter(lonnn[ii,iW,0],latnn[ii,iW,0],c=topNO3nn[ii,iW],marker='o',edgecolors='gray',linewidth=0.2,s=100,alpha=0.7,cmap=gb.mpl_colormap)
		im1.set_clim(0,30)
		im1 = ax.scatter(lonnn[ii,iC,0],latnn[ii,iC,0],c=topNO3nn[ii,iC],marker='s',edgecolors='gray',linewidth=0.2,s=100,alpha=0.7,cmap=gb.mpl_colormap)
		im1.set_clim(0,30)
		#ax.scatter(lonnn[ii,iSeas,0],latnn[ii,iSeas,0],c='None',marker='o',edgecolors='w',s=100,alpha=0.2)
		if ii == 0:
			cax    = fig.add_axes([0.65, 0.2, 0.2, 0.01])
			cbar   = plt.colorbar(im1,cax=cax,orientation='horizontal')
			cbar.ax.set_xlabel(r'%s' %(xlab[2]))
			cbar.ax.xaxis.set_tick_params(color='k')
			cbar.set_ticks(np.linspace(0,30,6))  
		
	im1, = ax.plot([np.nan],color='gray',marker='o',label='Spring + Summer: N$_{tot}$=%i' %ntotW,linewidth=1)
	im2, = ax.plot([np.nan],color='gray',marker='s',label='Autumn + Winter: N$_{tot}$=%i' %ntotC,linewidth=1)
	leg  = ax.legend(loc=9,ncol=2,handles=[im1,im2],fancybox=True)
	leg.get_frame().set_alpha(0.7)
	
	ax.set_xlim(XC[x1],XC[x2])
	ax.set_ylim(YC[y1],YC[y2])
	ax.set_title('Top 100 m averaged NO$_3$',fontsize=14)
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_floats_NO3_seas2_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
		
	plt.close(fig)	
	
	# top 50 mean of floats O2, divided into warm (spr+sum) and cold (aut+win) months 
	topO2nn  = np.nanmean(oxynn[:,:,:99],axis=2)
	"""
	fig       = plt.figure(figsize=(12,12))
	for jj in range(1,3):
		ntot   = 0
		ax     = plt.subplot(2,1,jj)
		im     = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.2)
		plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k',alpha=0.3)
		if jj == 2:
			cax    = fig.add_axes([0.15, 0.15, 0.15, 0.01])
			cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
			cbar.ax.set_xlabel('ocean depth [km]')
			cbar.ax.xaxis.set_tick_params(color='w')
		plot_Orsi(fig,ax,'k')
		plot_PF(fig,ax,'k')
		for ii in range(5):
			if jj == 1:
				iSeas  = np.where(np.logical_or(seasnn[ii,:,0]==1,seasnn[ii,:,0]==2))[0][:]
			else:
				iSeas  = np.where(np.logical_or(seasnn[ii,:,0]==3,seasnn[ii,:,0]==4))[0][:]
			ntot= ntot + len(topO2nn[ii,iSeas][~np.isnan(topO2nn[ii,iSeas])])
			im1 = ax.scatter(lonnn[ii,iSeas,0],latnn[ii,iSeas,0],c=topO2nn[ii,iSeas],marker='o',edgecolors='None',s=100,alpha=1,cmap=cm.deep)
			im1.set_clim(220,360)
			#ax.scatter(lonnn[ii,iSeas,0],latnn[ii,iSeas,0],c='None',marker='o',edgecolors='w',s=100,alpha=0.2)
			if jj == 1 and ii == 0:
				cax    = fig.add_axes([0.15, 0.6, 0.15, 0.01])
				#cax.text(0.5,0.8,'DO',fontsize=14, color='w',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.8))
				cbar   = fig.colorbar(im1,cax=cax,orientation='horizontal')
				cbar.ax.set_xlabel(r'DO %s' %(xlab[3]))
				cbar.ax.xaxis.set_tick_params(color='k')
				cbar.set_ticks(np.linspace(220,360,6))  
		ax.set_xlim(XC[x1],XC[x2])
		ax.set_ylim(YC[y1],YC[y2])
		ax.set_title('%s. N$_{prof}=$%i' %(seastit[jj-1],ntot))
	plt.suptitle('Top 50 m average of floats DO', fontsize=14)
	"""
	
	fig       = plt.figure(figsize=(12,7))
	ntotW = ntotC = 0
	ax     = plt.subplot(111)
	im     = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.2)
	plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k',alpha=0.3)
	cax    = fig.add_axes([0.15, 0.2, 0.2, 0.01])
	cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('[km]')
	cbar.ax.xaxis.set_tick_params(color='k')
	plot_Orsi(fig,ax,'k')
	plot_PF(fig,ax,'k')
	for ii in range(5):
		iW  = np.where(np.logical_or(seasnn[ii,:,0]==1,seasnn[ii,:,0]==2))[0][:]
		iC  = np.where(np.logical_or(seasnn[ii,:,0]==3,seasnn[ii,:,0]==4))[0][:]
		ntotW = ntotW + len(topO2nn[ii,iW][~np.isnan(topO2nn[ii,iW])])
		ntotC = ntotC + len(topO2nn[ii,iC][~np.isnan(topO2nn[ii,iC])])
		im1 = ax.scatter(lonnn[ii,iW,0],latnn[ii,iW,0],c=topO2nn[ii,iW],marker='o',edgecolors='gray',linewidth=0.2,s=100,alpha=0.7,cmap=ord.mpl_colormap)
		im1.set_clim(220,360)
		im1 = ax.scatter(lonnn[ii,iC,0],latnn[ii,iC,0],c=topO2nn[ii,iC],marker='s',edgecolors='gray',linewidth=0.2,s=100,alpha=0.7,cmap=ord.mpl_colormap)
		im1.set_clim(220,360)
		#ax.scatter(lonnn[ii,iSeas,0],latnn[ii,iSeas,0],c='None',marker='o',edgecolors='w',s=100,alpha=0.2)
		if ii == 0:
			cax    = fig.add_axes([0.65, 0.2, 0.2, 0.01])
			cbar   = plt.colorbar(im1,cax=cax,orientation='horizontal')
			cbar.ax.set_xlabel(r'%s' %(xlab[3]))
			cbar.ax.xaxis.set_tick_params(color='k')
			cbar.set_ticks(np.linspace(220,360,6))  
		
	im1, = ax.plot([np.nan],color='gray',marker='o',label='Spring + Summer: N$_{tot}$=%i' %ntotW,linewidth=1)
	im2, = ax.plot([np.nan],color='gray',marker='s',label='Autumn + Winter: N$_{tot}$=%i' %ntotC,linewidth=1)
	leg  = ax.legend(loc=9,ncol=2,handles=[im1,im2],fancybox=True)
	leg.get_frame().set_alpha(0.7)
	
	ax.set_xlim(XC[x1],XC[x2])
	ax.set_ylim(YC[y1],YC[y2])
	ax.set_title('Top 100 m averaged DO',fontsize=14)

	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_floats_DO_seas2_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
			
	plt.show()	
	
	
	# plot Landshuster co2Flux mean with floats:
	[lonL,latL,fco2] = load_Landsch()
	fig,ax    = plt.subplots(figsize=(12,7))
	plt.pcolor(lonL,latL,np.nanmean(fco2,axis=0),cmap=cm.balance)
	cbar = plt.colorbar()
	plt.clim(-2,2)
	#cbar.ax.set_title('CO$_2$ flux density smoothed')
	cbar.ax.set_xlabel(r'mol m$^{-2}$ yr$^{-1}$')
	
	for ii,id in enumerate(WMtot):
		ax.scatter(lon_surf[np.where(lab2==ii+1)],lat_surf[np.where(lab2==ii+1)],c=col[ii],marker='.',edgecolors='face',s=200,alpha=0.4) 
		ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=4)
		if ii == 4:
			ax.legend(loc=1,fancybox=True)	
	#ax.set_xlim(XC[x1],XC[x2])
	#ax.set_ylim(YC[y1],YC[y2])
	
	plot_PF(fig,ax,'k')
	plot_Orsi(fig,ax,'k')			
	ax.set_xlim(lonL[0],lonL[-1])
	ax.set_ylim(latL[0],latL[-1])
	
	plt.title(u'Floats clusters and 1982-2015 time average of Landschutzer smoothed CO$_2$ flux density')
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_Lands_fco2_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
			
	plt.show()	
	
	# plot property/density from all profiles, and GLODAP mean and floats mean
	labTr2D = np.reshape(labTr,[len(latC[yC1:yC2]),len(lonC[xC1:xC2])])
	labm  = np.ma.masked_where(np.ma.getmask(dataTT),labTr2D)

	fig = plt.figure(figsize=(15,17))
	ax1 = fig.add_subplot(221)
	ax2 = fig.add_subplot(222)
	ax3 = fig.add_subplot(223)
	ax4 = fig.add_subplot(224)
	axx = [ax1,ax2,ax3,ax4]
	for pp in range(4):
		for ii in range(5):
			# GLODAP
			ii1,ii2 = np.where(labm==ii+1)
			ff_Cl = bin_ffCl[pp][...]
			ff_Cl[ff_Cl==-999.] = np.nan
			mean_Cl = np.nanmean(ff_Cl[ii1,ii2,:],axis=0)
			std_Cl  = np.nanstd(ff_Cl[ii1,ii2,:],axis=0)
			f0 = mean_Cl
			f0[f0==0.] = np.nan
			f1 = f0[~np.isnan(f0)]
			f2 = RHObin[~np.isnan(f0)]
			axx[pp].plot(f1,f2,color=col[ii],alpha=0.6,linewidth=4)
			axx[pp].plot(f1,f2,color='k',alpha=0.9,linewidth=1)
			
			# smooth the mean of GLODAP profiles
			#win = 10
			#bmnew = smooth(bm1,win,'hanning')
			#axx[pp].plot(bmnew[win-1:],b2,color=col[ii],alpha=0.6,linewidth=6)
			#axx[pp].plot(bmnew[win-1:],b2,color='k',alpha=0.9,linestyle=':',linewidth=1)	
			
			# floats
			ff = bin_ff[pp][ii,:,:]		
			for jj in range(1000):
				f0 = ff[jj,:]
				f1 = f0[~np.isnan(f0)] 
				if len(f1)!=0:
					f2 = RHObin[~np.isnan(f0)]
					axx[pp].plot(f1,f2,color=col[ii],alpha=0.4,linewidth=0.2)	
			# flaots mean
			fm = np.nanmean(ff,axis=0)
			f1 = fm[~np.isnan(fm)] 
			f2 = RHObin[~np.isnan(fm)]
			axx[pp].plot(f1,f2,color=col[ii],linestyle='--',alpha=0.6,linewidth=4)
			axx[pp].plot(f1,f2,color='k',linestyle='--',alpha=0.9,linewidth=1)	
			
			if pp in [0,2]:
				axx[pp].set_ylabel(r'$\sigma_0$ [kg m$^{-3}$]',fontsize=14)
			elif pp == 3:
				axx[pp].plot(np.nan,np.nan,color=col[ii],linewidth=2,label=labZ[ii])
				if ii ==4:
					leg = axx[pp].legend(loc=1,fancybox=True)
					leg.get_frame().set_alpha(0.5)

		axx[pp].invert_yaxis()	
		axx[pp].set_title('%s' %axtit[pp],fontsize=14)
		axx[pp].set_xlabel('%s' %(xlab[pp]),fontsize=14)
		axx[pp].grid('on')
			
	ax1 = fig.add_subplot(221)		
	im1, = plt.plot([np.nan],color='k',linestyle='-',label='GLODAPv2',linewidth=2)
	im2, = plt.plot([np.nan],color='k',linestyle='--',label='SOCCOM floats',linewidth=2)
	leg  = plt.legend(loc=3,handles=[im1,im2],fancybox=True)
	leg.get_frame().set_alpha(0.5)
					
	plt.suptitle(r'properties interpolated on $\sigma_0$',fontsize=16)
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_prop_dens_bin_GLODAP_%im_2.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
			
	plt.show()	
	
	
	# property/density mean, by zone and region
	mean_fl_reg = np.nan*np.ones((5,4,4,nbins))
	std_fl_reg  = np.nan*np.ones((5,4,4,nbins))
	mean_Cl_reg = np.nan*np.ones((5,4,4,nbins))
	std_Cl_reg  = np.nan*np.ones((5,4,4,nbins))
	# GLODAP means by regions:
	WidxCl  = np.where(lonC[xC1:xC2]<=40)[0][:]
	UidxCl  = np.where(np.logical_and(lonC[xC1:xC2] > 40,lonC[xC1:xC2] <= 68.))[0][:]
	DidxCl  = np.where(np.logical_and(lonC[xC1:xC2] > 68,lonC[xC1:xC2] < 120.))[0][:]
	EidxCl  = np.where(lonC[xC1:xC2]>=120.)[0][:]
	idxCl   = [WidxCl,UidxCl,DidxCl,EidxCl]
	for ii in range(5):
		# floats indexes of regions
		Widx  = np.where(lonnn[ii,:,0]<=40)[0][:]
		Uidx  = np.where(np.logical_and(lonnn[ii,:,0] > 40,lonnn[ii,:,0] <= 68.))[0][:]
		Didx  = np.where(np.logical_and(lonnn[ii,:,0] > 68,lonnn[ii,:,0] < 120.))[0][:]
		Eidx  = np.where(lonnn[ii,:,0]>=120.)[0][:]
		idx   = [Widx,Uidx,Didx,Eidx]
	
		for pp in range(4):
			for ir,rr in enumerate(idxCl):
				ff_Cl = bin_ffCl[pp][...]
				i1,i2 = np.where(labTr2D[:,rr]==ii+1)
				ff_Cl[ff_Cl==-999.] = np.nan
				# getting rid of negative nitrate
				if pp ==3:
					ff_Cl[ff_Cl<0.] = np.nan
				mean_Cl_reg[ii,pp,ir,:] = np.nanmean(ff_Cl[:,rr,:][i1,i2,:],axis=0)
				std_Cl_reg[ii,pp,ir,:]  = np.nanstd(ff_Cl[:,rr,:][i1,i2,:],axis=0)
				
			for ir,rr in enumerate(idx):
				ff             = bin_ff[pp][ii,...]
				ff[ff==0.]     = np.nan
				mean_fl_reg[ii,pp,ir,:] = np.nanmean(ff[rr,...],axis=0)
				std_fl_reg[ii,pp,ir,:]  = np.nanstd(ff[rr,...],axis=0)
			
	# plot means:
	xmin     = [32,-2,150,-5,10]
	xmax     = [36,25,400,40,50]
	s0min    = [26.5,26.5,26.8,26.8,26.8]
	labm     = np.ma.masked_where(np.ma.getmask(dataTT),labTr2D)
	# plot the regional std mean
	for pp in range(4):
		fig = plt.figure(figsize=(15,15))
		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)
		ax4 = fig.add_subplot(224)
		axx = [ax1,ax2,ax3,ax4]
		for ir,ax in enumerate(axx):
			for ii in range(4):
				ii1,ii2 = np.where(labm[:,idxCl[ir]]==ii+1)

				ff_Cl = bin_ffCl[pp][...]
				ff_Cl[ff_Cl==-999.] = np.nan
				mean_Cl = np.nanmean(ff_Cl[:,idxCl[ir],:][ii1,ii2,:],axis=0)
				std_Cl  = np.nanstd(ff_Cl[:,idxCl[ir],:][ii1,ii2,:],axis=0)
				
				f0 = mean_Cl
				f0[f0==0.] = np.nan
				f1 = f0[~np.isnan(f0)]
				f2 = RHObin[~np.isnan(f0)]
				e1 = std_Cl
				e1 = e1[~np.isnan(f0)]
				ax.plot(f1,f2,color=col[ii],alpha=0.6,linewidth=4)
				ax.fill_betweenx(f2, f1-e1, f1+e1,alpha=0.7, edgecolor=col[ii], facecolor=col[ii], linewidth=2, linestyle='-', antialiased=True)
				ax.invert_yaxis()
		
				f0 = mean_fl_reg[ii,pp,ir,:]
				f0[f0==0.] = np.nan
				f1 = f0[~np.isnan(f0)]
				f2 = RHObin[~np.isnan(f0)]
				e1 = std_fl_reg[ii,pp,ir,:]
				e1 = e1[~np.isnan(f0)]
				ax.plot(f1,f2,color=col[ii],alpha=0.6,linewidth=4,linestyle='--')
				ax.fill_betweenx(f2, f1-e1, f1+e1,alpha=0.4, edgecolor='k', facecolor=col[ii], linewidth=2, linestyle='--', antialiased=True)
			
				ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=2)
				

			ax.set_ylim(24.5,27.7)
			ax.set_xlim(xmin[pp],xmax[pp])
			ax.invert_yaxis()
			ax.set_title('%s' %(regTit[ir]),fontsize=16)
			#ax.text(2.*xmax[pp]/3.,27.5, '%s' %(regTit[ir]),fontsize=16, color='k',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.2))
			ax.grid('on')
			if ir == 3:
				leg = ax.legend(loc=1,fancybox=True)
				leg.get_frame().set_alpha(0.5)
			ax.set_ylabel(r'$\sigma_0$ [kg m$^{-3}$]',fontsize=14)
			#if ir == 1:
			ax.set_xlabel('%s' %(xlab[pp]),fontsize=14)
				
		plt.suptitle('%s' %(axtit[pp]),fontsize=16)
		
		outfile = os.path.join(plotdir,'k_means_Indian_Ocean_mean_std_floats_GLODAP_%s_%im.png' %(prop[pp],zlev))
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)		
		
	# floats regional variability ~ HERE!!!!!!!!!!
	styles = ['-','--',':','-.']
	yax    = [[24.5,26.6],[25.5,26.8],[26.2,27.2],[26,27.2],[26.7,27.2]]
	
	deep   = True
	for pp in range(4):
		if pp == 0:
			# deep
			if deep:
				xmi = [34,33.5,33.8,33.8,32.5]
				xMa = [35,35.5,35,35,35]
			else:
				# all
				xmi = [34,33,33.5,32.5,32.5]
				xMa = [36,36,35,35,35]
		elif pp == 1:
			if deep:
				xmi = [0,0,0,-2,-4]
				xMa = [9,8,4,3,3]
			else:
				xmi = [0,0,0,-4,-4]
				xMa = [25,15,8,5,3]
		elif pp == 2:
			xmi = [150]*5
			if deep:
				xMa = [300,350,350,400,400]
			else:
				xMa = [250,300,300,350,400]
		elif pp ==3:
			if deep:
				xmi = [20,20,30,25,20]
				xMa = 5*[40]
			else:
				xmi = [-5,5,20,20,20]
				xMa = 5*[40]
			
		fig = plt.figure(figsize=(18,5))
		for ii in range(4):
			yaxis1 = np.linspace(yax[ii][0],yax[ii][1],3)
			yaxis2 = np.linspace(yax[ii][1],27.7,3)	
			Widx  = np.where(lonnn[ii,:,0]<=40)[0][:]
			Uidx  = np.where(np.logical_and(lonnn[ii,:,0] > 40,lonnn[ii,:,0] <= 68.))[0][:]
			Didx  = np.where(np.logical_and(lonnn[ii,:,0] > 68,lonnn[ii,:,0] < 120.))[0][:]
			Eidx  = np.where(lonnn[ii,:,0]>=120.)[0][:]
			idx   = [Widx,Uidx,Didx,Eidx]		
			for zz in [1]:#,5]:
				ax1 = fig.add_subplot(1,4,ii+zz)
				#ax1 = fig.add_subplot(2,4,ii+zz)
				for ir,rr in enumerate(idx):
					ff             = bin_ff[pp][ii,...]
					ff[ff==0.]     = np.nan
					f0             = np.nanmean(ff[rr,...],axis=0)
					e1             = np.nanstd(ff[rr,...],axis=0)
					
					#f0 = mean_fl_reg[ii,pp,ir,:]
					f0[f0==0.] = np.nan
					f1 = f0[~np.isnan(f0)]
					f2 = RHObin[~np.isnan(f0)]
					#e1 = std_fl_reg[ii,pp,ir,:]
					e1 = e1[~np.isnan(f0)]
					ax1.plot(f1,f2,color=col[ir],alpha=0.8,linewidth=2,label=regTit[ir])
					ax1.fill_betweenx(f2, f1-e1, f1+e1,alpha=0.2, edgecolor='k', facecolor=col[ir], linewidth=2, linestyle='-', antialiased=True)

					if zz == 1:
						ax1.set_title('%s' %labZ[ii],fontsize=18) 
						#ax1.set_ylim(yax[ii][0],yax[ii][1])
						#ax1.set_yticks(yaxis1)
						if ii > 1:
							ax1.set_ylim(27.2,27.7)#(26,27.7)
						else:
							ax1.set_ylim(26.8,27.7)#(24.5,27.7)	
					else:
						ax1.set_ylim(yax[ii][1],27.7)
						ax1.set_yticks(yaxis2)
					ax1.set_xlim(xmi[ii],xMa[ii])
					ax1.set_xticks(np.linspace(xmi[ii],xMa[ii],5))	
					#ax1.set_xticks(np.linspace(xmi[ii],xMa[ii],5))	
					ax1.invert_yaxis()
					ax1.grid('on')
					if ii == 0:
						leg = ax1.legend(loc=4,fancybox=True)
						leg.get_frame().set_alpha(0.5)
						ax1.set_ylabel(r'$\sigma_0$ [kg m$^{-3}$]',fontsize=18)
					#elif ii == 2:
					ax1.set_xlabel('%s' %(xlab[pp]),fontsize=14)
				
		#plt.suptitle('SOCCOM floats %s regional variability' %(prop[pp]),fontsize=18)
		
		outfile = os.path.join(plotdir,'k_means_Indian_Ocean_mean_floats_reg_DEEP_%s_%im.png' %(prop[pp],zlev))
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)	
		
		# GLODAP #something wrong with the nitrate...
		fig = plt.figure(figsize=(15,7))
		for ii in range(4):
			ax1 = fig.add_subplot(1,4,ii+1)
			for ir in range(4):
				f0 = mean_Cl_reg[ii,pp,ir,:]
				ii1,ii2 = np.where(labm[:,idxCl[ir]]==ii+1)
				ff_Cl = bin_ffCl[pp][...]
				ff_Cl[ff_Cl==-999.] = np.nan
				mean_Cl = np.nanmean(ff_Cl[:,idxCl[ir],:][ii1,ii2,:],axis=0)
				std_Cl  = np.nanstd(ff_Cl[:,idxCl[ir],:][ii1,ii2,:],axis=0)				
				f0      = mean_Cl
				f0[f0==0.] = np.nan
				#if ii == 1 and ir == 2:
				#	f0[np.where(RHObin<25.9)]=np.nan
				f1 = f0[~np.isnan(f0)]
				f2 = RHObin[~np.isnan(f0)]
				e1 = std_Cl
				e1 = e1[~np.isnan(f0)]
				ax1.plot(f1,f2,color=col[ir],alpha=0.8,linewidth=2,linestyle='-',label=regTit[ir])
				ax1.fill_betweenx(f2, f1-e1, f1+e1,alpha=0.2, edgecolor=col[ir], facecolor=col[ii], linewidth=2, linestyle='-', antialiased=True)
			
				ax1.set_title('%s' %labZ[ii],fontsize=14) 
				if ii >1:
					ax1.set_ylim(26,27.7)
				else:
					ax1.set_ylim(24.5,27.7)			
				ax1.set_xlim(xmi[ii],xMa[ii])
				ax1.invert_yaxis()
				ax1.grid('on')
				ax1.set_xticks(np.linspace(xmi[ii],xMa[ii],5))	
				if ii == 4:
					leg = ax1.legend(loc=1,fancybox=True)
					leg.get_frame().set_alpha(0.5)
				elif ii ==0:
					ax1.set_ylabel(r'$\sigma_0$ [kg m$^{-3}$]',fontsize=14)
				elif ii == 2:
					ax1.set_xlabel('%s' %(xlab[pp]),fontsize=14)
				
		plt.suptitle('GLODAPv2 %s regional variability' %(prop[pp]),fontsize=16)
		
		outfile = os.path.join(plotdir,'k_means_Indian_Ocean_mean_GLODAP_reg_%s_%im.png' %(prop[pp],zlev))
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)		
		
		
	#~~~~~~~ QUI ~~~~~~
	# plot color coding by properties
	colS = ['b','c','y','r']
	labS = ['SPR','SUM','AUT','WIN']
	titFig = ['OXY','NO3','SEASONS']
	
	fig3 = plt.figure(3)
	ax1  = fig3.add_subplot(111)
	im   = ax1.scatter(sann,tenn,c=oxynn,marker='.',edgecolors='face',s=100,alpha=0.2)
	im.set_clim(150,300)
	cbar = plt.colorbar(im)
	cbar.ax.set_xlabel(r'[$\mu$mol kg$^{-1}$]')
	cbar.ax.set_title('O$_2$')

	fig4 = plt.figure(4)
	ax2  = fig4.add_subplot(111)
	im   = ax2.scatter(sann,tenn,c=no3nn,marker='.',edgecolors='face',s=100,alpha=0.2)
	im.set_clim(0,40)
	cbar = plt.colorbar(im)
	cbar.ax.set_xlabel(r'[$\mu$mol kg$^{-1}$]')
	cbar.ax.set_title('NO$_3$')
	
	fig5 = plt.figure(5)
	ax3  = fig5.add_subplot(111)
	im   = ax3.scatter(sann,tenn,c=seasnn,marker='.',edgecolors='face',s=100,alpha=0.2)
	im.set_clim(1,4)
	for ii in range(4):	
		ax3.plot(np.nan,np.nan,color=colS[ii],label='%s' %labS[ii],linewidth=4)
		if ii == 3:
			ax3.legend(loc=2,fancybox=True)	
	
	ax = [ax1,ax2,ax3]
	for ii in range(3):
		plt.figure(3+ii)
		ax[ii].set_xlabel('SP', fontsize=12)
		ax[ii].set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		ax[ii].set_xlim(np.min(ss),36)
		ax[ii].set_ylim(-3,23)	
		cs  = ax[ii].contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
		plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
	
		outfile = os.path.join(plotdir,'k_means_Indian_Ocean_TS_%s_full_%im.png' %(titFig[ii],zlev))
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
		
	# means and std using density-binned profiles
	sa_std_reg    = np.nan*np.ones((5,4,nbins))	
	te_std_reg    = np.nan*np.ones((5,4,nbins))	
	oxy_std_reg   = np.nan*np.ones((5,4,nbins))	
	no3_std_reg   = np.nan*np.ones((5,4,nbins))
	dic_std_reg   = np.nan*np.ones((5,4,nbins))
	reg_std       = [sa_std_reg,te_std_reg,oxy_std_reg,no3_std_reg,dic_std_reg]
	for ii in range(5):
		# std = np.sqrt(np.nanmean(np.abs(ff - ff_Cl.mean())**2))
		Widx  = np.where(lonnn[ii,:,0]<=40)[0][:]
		Uidx  = np.where(np.logical_and(lonnn[ii,:,0] > 40,lonnn[ii,:,0] <= 68.))[0][:]
		Didx  = np.where(np.logical_and(lonnn[ii,:,0] > 68,lonnn[ii,:,0] < 120.))[0][:]
		Eidx  = np.where(lonnn[ii,:,0]>=120.)[0][:]
		idx   = [Widx,Uidx,Didx,Eidx]
	
		# GLODAP means by regions:
		mean_Cl_reg = np.nan*np.ones((4,nbins))
		WidxCl  = np.where(lonC[xC1:xC2]<=40)[0][:]
		UidxCl  = np.where(np.logical_and(lonC[xC1:xC2] > 40,lonC[xC1:xC2] <= 68.))[0][:]
		DidxCl  = np.where(np.logical_and(lonC[xC1:xC2] > 68,lonC[xC1:xC2] < 120.))[0][:]
		EidxCl  = np.where(lonC[xC1:xC2]>=120.)[0][:]
		idxCl   = [WidxCl,UidxCl,DidxCl,EidxCl]
		labTr2D = np.reshape(labTr,[len(latC[yC1:yC2]),len(lonC[xC1:xC2])])
		for pp in range(4):
			for ir,rr in enumerate(idxCl):
				ff_Cl = bin_ffCl[pp][...]
				i1,i2 = np.where(labTr2D[:,rr]==ii+1)
				mean_Cl_reg[ir,:] = np.nanmean(ff_Cl[:,rr,:][i1,i2,:],axis=0)
		
			for ir,rr in enumerate(idx):
				ff             = bin_ff[pp][:]
				ff[ff==0.]     = np.nan
				reg_std[pp][ii,ir,:] = np.sqrt(np.nanmean(np.abs(ff[ii,rr,:] - mean_Cl_reg[ir,:])**2,axis=0))
				#reg_std[pp][ii,ir,:] = np.nanstd(ff[ii,rr,:],axis=0)
	
	# plot the std(density) by zone and region
	regTit   = ['West','Upstream','Downstream','East']
	#xmax     = [0.5,2.5,100,8,50]
	xmax     = [0.8,3.5,140,10,50]
	s0min    = [26.5,26.5,26.8,26.8,26.8]
	# plot the regional std mean
	for pp in range(4):
		fig = plt.figure(figsize=(15,15))
		ax1 = fig.add_subplot(221)
		ax2 = fig.add_subplot(222)
		ax3 = fig.add_subplot(223)
		ax4 = fig.add_subplot(224)
		axx = [ax1,ax2,ax3,ax4]
		for ir,ax in enumerate(axx):
			for ii in range(5):
				f0 = reg_std[pp][ii,ir,:]
				f0[f0==0.] = np.nan
				f0[np.where(RHObin<s0min[ii])] = np.nan
				f1 = f0[~np.isnan(f0)]
				f2 = RHObin[~np.isnan(f0)]
				ax.plot(f1,f2,color=col[ii],alpha=0.6,linewidth=4)
				ax.plot(f1,f2,color='k',alpha=0.9,linewidth=1)
				ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=2)
			#ax.set_ylim(24.5,27.7)
			ax.set_ylim(26.5,27.7)	
			ax.set_xlim(0,xmax[pp])
			ax.invert_yaxis()
			ax.set_title('%s' %(regTit[ir]),fontsize=16)
			#ax.text(2.*xmax[pp]/3.,27.5, '%s' %(regTit[ir]),fontsize=16, color='k',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.2))
			ax.grid('on')
			if ir == 3:
				leg = ax.legend(loc=1,fancybox=True)
				leg.get_frame().set_alpha(0.5)
			ax.set_ylabel(r'$\sigma_0$ [kg m$^{-3}$]',fontsize=14)
			#if ir == 1:
			ax.set_xlabel('%s' %(xlab[pp]),fontsize=14)
				
		plt.suptitle('std(%s)' %(axtit[pp]),fontsize=16)
		
		outfile = os.path.join(plotdir,'k_means_Indian_Ocean_std_floats_GLODAP_upper_%s_%im.png' %(prop[pp],zlev))
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	# QUIIIIIII



	# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	# some statistics
	# compute means and var of properties
	# mean of profiles by region (5 zones x (total + 4 seasons) x 2000 levels)
	sa_zone  = np.nan*np.ones((5,5,2000))
	te_zone  = np.nan*np.ones((5,5,2000))
	oxy_zone = np.nan*np.ones((5,5,2000))
	no3_zone = np.nan*np.ones((5,5,2000))
	dic_zone = np.nan*np.ones((5,5,2000))
	
	fig,ax   = plt.subplots()
	for ii in range(5):
		# compute cluster mean of properties
		# seasonal mean
		for idxS in range(1,5):
			s1  =  np.where(seasnn[ii,:,0]==idxS)[0][:]
			sa_zone[ii,idxS,:]    = np.nanmean(sann[ii,s1,:],axis=0)
			te_zone[ii,idxS,:]    = np.nanmean(tenn[ii,s1,:],axis=0)
			oxy_zone[ii,idxS,:]   = np.nanmean(oxynn[ii,s1,:],axis=0)
			no3_zone[ii,idxS,:]   = np.nanmean(no3nn[ii,s1,:],axis=0)
			dic_zone[ii,idxS,:]   = np.nanmean(no3nn[ii,s1,:],axis=0)
		# whole mean
		sa_zone[ii,0,:]    = np.nanmean(sann[ii,...],axis=0)
		te_zone[ii,0,:]    = np.nanmean(tenn[ii,...],axis=0)
		oxy_zone[ii,0,:]   = np.nanmean(oxynn[ii,...],axis=0)
		no3_zone[ii,0,:]   = np.nanmean(no3nn[ii,...],axis=0)
		dic_zone[ii,0,:]   = np.nanmean(no3nn[ii,...],axis=0)
		# smoothing the mean because there are some annoying outliers
		#win = 200
		#dum = smooth(sa_zone[0,...],win,'hanning')
		#sa_mean[0,...] = new[win/2-1:]
		
		sa_zone[ii,:] = np.ma.masked_invalid(sa_zone[ii,...])
		te_zone[ii,:] = np.ma.masked_invalid(te_zone[ii,...])
		
		# plot the total mean
		ax.plot(sa_zone[ii,0,:],te_zone[ii,0,:],c=col[ii],alpha=0.8,linewidth=4,zorder=1000)
		ax.plot(sa_zone[ii,0,:],te_zone[ii,0,:],c='k',alpha=1,linewidth=1,zorder=100000)##,marker='.',edgecolors='face',s=200,alpha=0.4)
		for jj in range(sann.shape[1]):
			ax.plot(sann[ii,jj,:],tenn[ii,jj,:],c=col[ii],alpha=0.2,linewidth=0.2)#marker='.',edgecolors='face',s=200,alpha=0.4)
	
		ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=4)
		if ii == 4:
			ax.legend(loc=4,fancybox=True)	
		elif ii == 0:
			cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
			plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
			
	ax.set_xlim(np.min(ss),36)
	ax.set_ylim(-3,23)		
	ax.set_xlabel('SP', fontsize=12)
	ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_TS_MeanProf_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	
	# ~~~~~~~~~~~~~~~~~
	# seasons		
	labSeas = ['total','spr','sum','aut','win']
	fig     = plt.figure(1,figsize=(15,10))
	for ii in range(5):
		ax = fig.add_subplot(2,3,1)
		for jj in range(sann.shape[1]):
			ax.plot(sann[ii,jj,:],tenn[ii,jj,:],c=col[ii],alpha=0.2,linewidth=0.2)#marker='.',edgecolors='face',s=200,alpha=0.4)
		#ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=4)
		ax = fig.add_subplot(2,3,ii+2)
		for iii in range(5):
			ax.plot(sa_zone[ii,iii,:],te_zone[ii,iii,:],c=col[iii],alpha=0.6,linewidth=4,zorder=100000,label='%s' %(labSeas[iii]))
			ax.plot(sa_zone[ii,iii,:],te_zone[ii,iii,:],c='k',alpha=1,linewidth=0.5,zorder=100000)
		if ii==4:
			ax.legend(loc=2,fancybox=True)	
		ax.set_title('%s' %labZ[ii],fontsize=14)
		ax.set_xlim(np.min(ss),36)
		ax.set_ylim(-3,23)		
		ax.set_xlabel('SP', fontsize=12)
		ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
		cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
		plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
	
	ax = fig.add_subplot(2,3,1)
	#ax.legend(loc=4,fancybox=True)	
	ax.set_xlim(np.min(ss),36)
	ax.set_ylim(-3,23)		
	ax.set_xlabel('SP', fontsize=12)
	ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
	cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
	plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
		
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_TS_MeanProf_Seas_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)

	# ~~~~~~~~~~~~~~~~~
	# mean anomaly, divided by season and total
	sa_mean_anom  = np.zeros((5,3,5,2000))	
	te_mean_anom  = np.zeros((5,3,5,2000))	
	oxy_mean_anom = np.zeros((5,3,5,2000))	
	no3_mean_anom = np.zeros((5,3,5,2000))
	dic_mean_anom = np.zeros((5,3,5,2000))	
	
	# profile of std, separated by West Indian Ocean, Downstream KP and East Indian Ocean
	# divided by season and total std
	sa_std_reg    = np.nan*np.ones((5,3,5,2000))	
	te_std_reg    = np.nan*np.ones((5,3,5,2000))	
	oxy_std_reg   = np.nan*np.ones((5,3,5,2000))	
	no3_std_reg   = np.nan*np.ones((5,3,5,2000))
	dic_std_reg   = np.nan*np.ones((5,3,5,2000))
	
	for ii in range(5):
		# compute std for the 3 regions, by zone
		Widx  = np.where(lonnn[ii,:,0]<=68)[0][:]
		Didx  = np.where(np.logical_and(lonnn[ii,:,0] > 68,lonnn[ii,:,0] < 100.))[0][:]
		Eidx  = np.where(lonnn[ii,:,0]>=100.)[0][:]
		idx   = [Widx,Didx,Eidx]
		ffnn  = [sann,tenn,oxynn,no3nn,dicnn]
	
		reg_std  = [sa_std_reg,te_std_reg,oxy_std_reg,no3_std_reg,dic_std_reg]
		all_anom = [sa_mean_anom,te_mean_anom,oxy_mean_anom,no3_mean_anom,dic_mean_anom]
		ff_zone  = [sa_zone,te_zone,oxy_zone,no3_zone,dic_zone]
		for ir,rr in enumerate(idx):
			for pp in range(5):
				ff             = ffnn[pp][:]
				ff[ff==0.]     = np.nan
				reg_std[pp][ii,ir,0,:] = np.nanstd(ff[ii,rr,:],axis=0)
				plt.plot(reg_std[pp][ii,ir,0,:],label='%i' %pp)
				# compute the sum of anomaly by region
				all_anom[pp][ii,ir,0,:]  = np.nanmean(ff[ii,rr,:] - ff_zone[pp][ii,None,0,:],axis=0)
				# divided by season:
				for idxS in range(1,5):
					s1  = np.where(seasnn[ii,rr,0]==idxS)[0][:]
					reg_std[pp][ii,ir,idxS,:] = np.nanstd(ff[ii,s1,:],axis=0)
					# compute the sum of anomaly by region
					all_anom[pp][ii,ir,idxS,:]  = np.nanmean(ff[ii,s1,:] - ff_zone[pp][ii,None,idxS,:],axis=0)
				
				"""
				# extract the indexes of AAIW (for regions with ii in [0,1] and 26.9 <= sigma0 <= 27.2)
				AAIW     = np.where(np.logical_and(densnn[ii,jj,:]>=26.9,densnn[ii,jj,:]<=27.2))[0][:]
				dof      = len(AAIW)
				sa_mean_anom[ii,jj]    = np.nansum(sa_anom[AAIW])/float(dof)
				"""
	
	# regional std
	#reg_std = [sa_std_reg,te_std_reg,oxy_std_reg,no3_std_reg,dic_std_reg]
	# mean anomaly by profile
	#all_anom = [sa_mean_anom,te_mean_anom,oxy_mean_anom,no3_mean_anom,dic_mean_anom]
	#all_anom = np.ma.masked_equal(all_anom,-0.)

	regTit   = ['West','Center','East']
	xmax     = [0.3,2.5,50,7,50]
	# plot the regional std mean
	for pp in range(4):
		fig = plt.figure(figsize=(15,7))
		ax1 = fig.add_subplot(131)
		ax2 = fig.add_subplot(132)
		ax3 = fig.add_subplot(133)
		axx = [ax1,ax2,ax3]
		for ir,ax in enumerate(axx):
			for ii in range(5):
				#ax.scatter(reg_std[pp][ii,rr,:],press[:,0],c=col[ii],marker='.',edgecolors='face',s=50,alpha=0.4)
				ax.scatter(reg_std[pp][ii,ir,0,:],press[:,0],c=col[ii],marker='.',edgecolors='face',s=10,alpha=0.4)
				ax.plot(reg_std[pp][ii,ir,1,:],press[:,0],color=col[ii],linewidth=2,linestyle='-',alpha=0.5)
				ax.plot(reg_std[pp][ii,ir,2,:],press[:,0],color=col[ii],linewidth=2,linestyle='--',alpha=0.5)
				ax.plot(reg_std[pp][ii,ir,3,:],press[:,0],color=col[ii],linewidth=2,linestyle='-.',alpha=0.5)
				ax.plot(reg_std[pp][ii,ir,4,:],press[:,0],color=col[ii],linewidth=2,linestyle=':',alpha=0.5)
				ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=2)
			ax.set_ylim(0,2000)	
			ax.invert_yaxis()
			#ax.set_title('%s' %(regTit[ir]))
			ax.text(2.*xmax[pp]/3.,1000, '%s' %(regTit[ir]),fontsize=16, color='k',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.2))
			ax.grid('on')
			ax.set_xlim(0,xmax[pp])
			if ir == 0:
				leg = ax.legend(loc=4,fancybox=True)
				leg.get_frame().set_alpha(0.5)
				ax.set_ylabel('pressure [db]',fontsize=14)
			if ir == 1:
				ax.set_xlabel('%s' %(xlab[pp]),fontsize=14)
		
		im1, = plt.plot([np.nan],color='k',linestyle='-',label='SPRING',linewidth=1)
		im2, = plt.plot([np.nan],color='k',linestyle='--',label='SUMMER',linewidth=1)
		im3, = plt.plot([np.nan],color='k',linestyle='-.',label='AUTUMN',linewidth=1)
		im4, = plt.plot([np.nan],color='k',linestyle=':',label='WINTER',linewidth=1)
		leg  = plt.legend(loc=4,handles=[im1,im2,im3,im4],fancybox=True)
		leg.get_frame().set_alpha(0.5)
				
		plt.suptitle('std(%s)' %(axtit[pp]),fontsize=16)
		
		outfile = os.path.join(plotdir,'k_means_Indian_Ocean_std_seas_%s_%im.png' %(prop[pp],zlev))
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
		
	# plot the regional mean anomaly
	xmax     = [0.5,4,30,8,50]
	axtit    = ['SA',r'\theta','O_2','NO_3','DIC']
	for pp in range(4):
		fig = plt.figure(figsize=(15,7))
		ax1 = fig.add_subplot(131)
		ax2 = fig.add_subplot(132)
		ax3 = fig.add_subplot(133)
		axx = [ax1,ax2,ax3]
		for rr,ax in enumerate(axx):
			for ii in range(5):
				ffanom = all_anom[pp][ii,rr,0,:]
				f1     = ffanom[~np.isnan(ffanom)]
				f2     = press[:,0][~np.isnan(ffanom)]
				ax.plot(f1,f2,color=col[ii],linewidth=4,alpha=0.8)
				ax.plot(np.nan,np.nan,color=col[ii],label='%s' %labZ[ii],linewidth=4)
			ax.set_ylim(0,2000)	
			ax.invert_yaxis()
			#ax.set_title('\n%s' %(regTit[rr]))
			ax.text(xmax[pp]/3.,1000, '%s' %(regTit[rr]),fontsize=16, color='k',bbox=dict(facecolor='gray', edgecolor='black', boxstyle='round', alpha=0.2))
			ax.grid('on')
			ax.set_xlim(-xmax[pp],xmax[pp])
			if rr == 0:
				leg = ax.legend(loc=4,fancybox=True)
				leg.get_frame().set_alpha(0.5)
				ax.set_ylabel('pressure [db]',fontsize=14)
			if rr == 1:
				ax.set_xlabel('%s' %(xlab[pp]),fontsize=14)
		plt.suptitle(r'$\frac{\sum({%s}_i-\overline{%s})}{N_{region}}$' %(axtit[pp],axtit[pp]),fontsize=16)
		
		outfile = os.path.join(plotdir,'k_means_Indian_Ocean_anom_%s_%im.png' %(prop[pp],zlev))
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)	
		
	plt.show()


	# plot TS only for regions 1 and 2, with AAIW isopycnals
	fig= plt.figure()
	ax = fig.add_subplot(111)
	for ii in range(2):
		for jj in range(1000):
			ax.plot(sann[ii,jj,...],tenn[ii,jj,...],c=col[ii],alpha=0.2,linewidth=0.3)#marker='.',edgecolors='face',s=200,alpha=0.4)
		if ii == 0:
			cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
			plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
			ax.contour(ss,tt,s0,[26.9],colors='k',linestyles='solid',linewidths=1,alpha=0.8)
			ax.contour(ss,tt,s0,[27.2],colors='k',linestyles='solid',linewidths=1,alpha=0.8)
		ax.plot(sa_zone[ii,:],te_zone[ii,:],c=col[ii],alpha=0.8,linewidth=4,zorder=1000)
		ax.plot(sa_zone[ii,:],te_zone[ii,:],c='k',alpha=1,linewidth=1,zorder=100000)
	ax.set_xlim(np.min(ss),36)
	ax.set_ylim(-3,23)		
	ax.set_xlabel('SP', fontsize=12)
	ax.set_ylabel(r'$\theta$ [$^{\circ}$C]', fontsize=12)
	
	outfile = os.path.join(plotdir,'k_means_Indian_Ocean_TS_AAIW_full_%im.png' %(zlev))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	

	# QUI!!!!!
	############################
	
	
	
	
	
	
	
		
	# start the plot for AAIW
	depth = 'AAIW'
	for ifig in range(4):
		if ifig==0:
			clim  = [-0.1,0.02]
		elif ifig==1:
			clim  = [-2.,2]
		elif ifig==2:
			clim  = [-2,2]
		elif ifig==3:
			clim  = [-1,1]
		elif ifig==4:
			clim  = [1000,2000]
			
		fig,ax = plt.subplots(figsize=(12,7))
		im     = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
		plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
		cax    = fig.add_axes([0.15, 0.2, 0.2, 0.02])
		cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
		cbar.ax.set_xlabel('[km]')
		cbar.ax.xaxis.set_tick_params(color='k')
		plot_Orsi(fig,ax)
		plot_PF(fig,ax)
		for ii in range(2):
			im=ax.scatter(lonnn[ii,:,0],latnn[ii,:,0],c=all_anom[ifig][ii,...],marker='.',edgecolors='face',s=300,alpha=0.4,cmap=plt.cm.get_cmap('jet', 6))
			im.set_clim(clim[0],clim[1])	
		cax    = fig.add_axes([0.65, 0.2, 0.2, 0.02])
		cbar   = plt.colorbar(im,cax=cax,orientation='horizontal',extend='both')
		cbar.ax.xaxis.set_tick_params(color='k')
		cbar.set_ticks(np.linspace(clim[0],clim[1],5))
		cbar.ax.set_xlabel(r'%s' %(xlab[ifig]))
		cbar.ax.set_title(r'%s' %(axtit[ifig]))
		ax.set_xlim(XC[x1],XC[x2])
		ax.set_ylim(YC[y1],YC[y2])	

		outfile = os.path.join(plotdir,'k_means_Indian_Ocean_%s_%s_%im.png' %(prop[ifig],depth,zlev))
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
		#plt.show()
		plt.close()
					
			
	############################	
	
	fig1   = plt.figure(1,figsize=(12,7))
	ax1    = fig1.add_subplot(111)
	fig2   = plt.figure(2,figsize=(12,7))
	ax2    = fig2.add_subplot(111)
	fig3   = plt.figure(3,figsize=(12,7))
	ax3    = fig3.add_subplot(111)
	fig4   = plt.figure(4,figsize=(12,7))
	ax4    = fig4.add_subplot(111)
	ax     = [ax1,ax2,ax3,ax4]
	for ifig in range(4):
		fig    = plt.figure(ifig+1)
		im     = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
		plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
		cax    = fig.add_axes([0.15, 0.2, 0.2, 0.02])
		cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
		cbar.ax.set_xlabel('[km]')
		cbar.ax.xaxis.set_tick_params(color='k')
		plot_Orsi(fig,ax[ifig])
		plot_PF(fig,ax[ifig])
		for ii in range(2):
			im  = ax[ifig].scatter(lonnn[ii,:,0],latnn[ii,:,0],c=std_pr[ifig][ii,:],marker='.',edgecolors='face',s=200,alpha=0.4)
			if ifig==0:
				clim  = [0.05,0.1]
				axLab = ''
				axtit = 'SA'
			elif ifig==1:
				clim  = [0.8,1.1]
				axLab = '[$^{\circ}C$]'
				axtit = r'$\theta$'
			elif ifig==2:
				clim  = [5,15]
				axLab = '[$\mu$mol kg$^{-1}$]'
				axtit = 'O$_2$'
			else:
				clim  = [1,3]
				axLab = '[$\mu$mol kg$^{-1}$]'
				axtit = 'NO$_3$'
	
			clim = [np.min(std_pr[ifig,ii,:]),np.max(std_pr[ifig,ii,:])]
			im.set_clim(clim[0],clim[1])
			if ii == 0:
				cax    = fig.add_axes([0.65, 0.2, 0.2, 0.02])
				cbar   = plt.colorbar(im,cax=cax,orientation='horizontal',extend='both')
				cbar.ax.xaxis.set_tick_params(color='k')
				cbar.set_ticks(np.linspace(clim[0],clim[1],5))
				cbar.ax.set_xlabel(r'%s' %(axLab))
				cbar.ax.set_title(r'%s' %(axtit))

			ax     = [ax1,ax2,ax3,ax4]

		ax[ifig].set_xlim(XC[x1],XC[x2])
		ax[ifig].set_ylim(YC[y1],YC[y2])
		
	
			
			
	"""
	# just checking the density..
	fig = plt.figure()
	ax  = fig.add_subplot(111)
	im   = ax.scatter(sann,tenn,c=densnn,marker='.',edgecolors='face',s=100,alpha=0.2)
	im.set_clim(25,27.7)
	plt.colorbar(im)
	cs  = ax.contour(ss,tt,s0,vd,colors='k',linestyles='dashed',linewidths=0.5,alpha=0.8)
	plt.clabel(cs,inline=0,fontsize=10,alpha=0.6)
	"""

	
	
	# initialize mean of properties arrays, which will be computed by region and water mass [indexes are: regions x WM]
	# WM --> 0: surface, 1: ML, 2: AAIW, 3: SAMW, 4: UCDW, 5: LCDW
	sa_mean  = np.nan*np.ones((5,6))
	te_mean  = np.nan*np.ones((5,6))
	oxy_mean = np.nan*np.ones((5,6))
	no3_mean = np.nan*np.ones((5,6))
	
	# initialize var arrays
	sa_std   = np.nan*np.ones((5,6,1000))
	te_std   = np.nan*np.ones((5,6,1000))
	oxy_std  = np.nan*np.ones((5,6,1000))
	no3_std  = np.nan*np.ones((5,6,1000))
		
	for ii in range(5):
		if ii in [0,1]:
			# extract the indexes of AAIW (for regions with ii in [0,1] and 26.9 <= sigma0 <= 27.2)
			[AAIW_i,AAIW_j]  = np.where(np.logical_and(densnn[ii,...]>=26.9,densnn[ii,...]<=27.2))
		
			# compute cluster mean of properties
			sa_mean[ii,2]    = np.nanmean(sann[ii,AAIW_i,AAIW_j])
			te_mean[ii,2]    = np.nanmean(tenn[ii,AAIW_i,AAIW_j])
			oxy_mean[ii,2]   = np.nanmean(oxynn[ii,AAIW_i,AAIW_j])
			no3_mean[ii,2]   = np.nanmean(no3nn[ii,AAIW_i,AAIW_j])
		
			print 'AAIW in', labZ[ii]
			print 'mean SA  = ',np.nanmean(sann[ii,AAIW_i,AAIW_j])
			print 'mean TE  = ',np.nanmean(tenn[ii,AAIW_i,AAIW_j])
			print 'mean OXY = ',np.nanmean(oxynn[ii,AAIW_i,AAIW_j])
			print 'mean NO3 = ',np.nanmean(no3nn[ii,AAIW_i,AAIW_j])
			print '\n'
		
			# 2. compute std in the WM (every profile has 1 variance, which is calculated versus the total mean of the cluster)
			for kk in range(densnn.shape[1]):
				AAIW = np.where(np.logical_and(densnn[ii,kk,:]>=26.9,densnn[ii,kk,:]<=27.2))[0][:]
				dof  = len(AAIW)
				sa_std[ii,2,kk]  = np.nansum(sann[ii,kk,AAIW]-sa_mean[ii,2])/dof
				te_std[ii,2,kk]  = np.nansum(tenn[ii,kk,AAIW]-te_mean[ii,2])/dof
				oxy_std[ii,2,kk] = np.nansum(oxynn[ii,kk,AAIW]-oxy_mean[ii,2])/dof
				no3_std[ii,2,kk] = np.nansum(no3nn[ii,kk,AAIW]-no3_mean[ii,2])/dof
				"""
				# RMS of anomaly... sort of..
				sa_std[ii,2,kk]  = np.sqrt(np.nansum((sann[ii,kk,AAIW]-sa_mean[ii,2])**2)/dof**2)
				te_std[ii,2,kk]  = np.sqrt(np.nansum((tenn[ii,kk,AAIW]-te_mean[ii,2])**2)/dof**2)
				oxy_std[ii,2,kk] = np.sqrt(np.nansum((oxynn[ii,kk,AAIW]-oxy_mean[ii,2])**2)/dof**2)
				no3_std[ii,2,kk] = np.sqrt(np.nansum((no3nn[ii,kk,AAIW]-no3_mean[ii,2])**2)/dof**2)
				"""
		# SURFACE
		sa_mean[ii,0]    = np.nanmean(sann[ii,:,0])
		te_mean[ii,0]    = np.nanmean(tenn[ii,:,0])
		oxy_mean[ii,0]   = np.nanmean(oxynn[ii,:,0])
		no3_mean[ii,0]   = np.nanmean(no3nn[ii,:,0])

		print 'SURFACE in', labZ[ii]
		print 'mean SA  = ',np.nanmean(sann[ii,:,0])
		print 'mean TE  = ',np.nanmean(tenn[ii,:,0])
		print 'mean OXY = ',np.nanmean(oxynn[ii,:,0])
		print 'mean NO3 = ',np.nanmean(no3nn[ii,:,0])
		print '\n'

		# 2. compute std in the WM (every profile has 1 variance, which is calculated versus the total mean of the cluster)
		for kk in range(densnn.shape[1]):
			dof              = np.count_nonzero(~np.isnan(densnn[ii,:,0]))
			sa_std[ii,0,kk]  = np.nansum(sann[ii,kk,0]-sa_mean[ii,0])/dof
			te_std[ii,0,kk]  = np.nansum(tenn[ii,kk,0]-te_mean[ii,0])/dof
			oxy_std[ii,0,kk] = np.nansum(oxynn[ii,kk,0]-oxy_mean[ii,0])/dof
			no3_std[ii,0,kk] = np.nansum(no3nn[ii,kk,0]-no3_mean[ii,0])/dof
			"""
			# RMS of anomaly
			sa_std[ii,0,kk]  = np.sqrt(np.nansum((sann[ii,kk,0]-sa_mean[ii,0])**2)/dof**2)
			te_std[ii,0,kk]  = np.sqrt(np.nansum((tenn[ii,kk,0]-te_mean[ii,0])**2)/dof**2)
			oxy_std[ii,0,kk] = np.sqrt(np.nansum((oxynn[ii,kk,0]-oxy_mean[ii,0])**2)/dof**2)
			no3_std[ii,0,kk] = np.sqrt(np.nansum((no3nn[ii,kk,0]-no3_mean[ii,0])**2)/dof**2)
			"""
		
		std_pr = [sa_std,te_std,oxy_std,no3_std]
		std_pr = np.ma.masked_equal(std_pr,-0.)
		
		"""
		plt.subplot(211)
		plt.plot(oxynn[ii,AAIW_i,AAIW_j],'*',color=col[ii])
		plt.subplot(212)
		plt.plot(sann[ii,AAIW_i,AAIW_j],'*',color=col[ii])   
		"""
		
	for lev in [2]:
		if lev == 0:
			depth = 'SURF'
		elif lev == 2:
			depth = 'AAIW'
		# start the plot for AAIW
		fig1   = plt.figure(1,figsize=(12,7))
		ax1    = fig1.add_subplot(111)
		fig2   = plt.figure(2,figsize=(12,7))
		ax2    = fig2.add_subplot(111)
		fig3   = plt.figure(3,figsize=(12,7))
		ax3    = fig3.add_subplot(111)
		fig4   = plt.figure(4,figsize=(12,7))
		ax4    = fig4.add_subplot(111)
		ax     = [ax1,ax2,ax3,ax4]
		for ifig in range(4):
			fig    = plt.figure(ifig+1)
			im     = plt.contourf(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
			plt.contour(XC[x1:x2], YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k')
			cax    = fig.add_axes([0.15, 0.2, 0.2, 0.02])
			cbar   = plt.colorbar(im,cax=cax,orientation='horizontal')
			cbar.ax.set_xlabel('[km]')
			cbar.ax.xaxis.set_tick_params(color='k')
			plot_Orsi(fig,ax[ifig])
			plot_PF(fig,ax[ifig])
			for ii in range(5):
				im  = ax[ifig].scatter(lonnn[ii,:,0],latnn[ii,:,0],c=std_pr[ifig][ii,lev,:],marker='.',edgecolors='face',s=200,alpha=0.4)
				if ifig==0:
					clim  = [0.05,0.1]
					axLab = ''
					axtit = 'SA'
				elif ifig==1:
					clim  = [0.8,1.1]
					axLab = '[$^{\circ}C$]'
					axtit = r'$\theta$'
				elif ifig==2:
					clim  = [5,15]
					axLab = '[$\mu$mol kg$^{-1}$]'
					axtit = 'O$_2$'
				else:
					clim  = [1,3]
					axLab = '[$\mu$mol kg$^{-1}$]'
					axtit = 'NO$_3$'
		
				clim = [np.min(std_pr[ifig,ii,lev,:]),np.max(std_pr[ifig,ii,lev,:])]
				im.set_clim(clim[0],clim[1])
				if ii == 0:
					cax    = fig.add_axes([0.65, 0.2, 0.2, 0.02])
					cbar   = plt.colorbar(im,cax=cax,orientation='horizontal',extend='both')
					cbar.ax.xaxis.set_tick_params(color='k')
					cbar.set_ticks(np.linspace(clim[0],clim[1],5))
					cbar.ax.set_xlabel(r'%s' %(axLab))
					cbar.ax.set_title(r'%s' %(axtit))

				ax     = [ax1,ax2,ax3,ax4]

			prop = ['SA','TE','OXY','NO3']
			ax[ifig].set_xlim(XC[x1],XC[x2])
			ax[ifig].set_ylim(YC[y1],YC[y2])

			outfile = os.path.join(plotdir,'k_means_Indian_Ocean_%s_%s_%im.png' %(prop[ifig],depth,zlev))
			print outfile
			plt.savefig(outfile, bbox_inches='tight',dpi=200)
			
		plt.close()
		
	
	
	# 12 July 2018: END		
	# !!!!!!!!!!!!
	
