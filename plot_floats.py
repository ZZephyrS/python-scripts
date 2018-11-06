import os,sys
import numpy as np
import netCDF4 as nc
import matplotlib as mpl

from matplotlib import pyplot as plt
from scipy.io import loadmat
from scipy import interpolate
from datetime import datetime
from palettable.colorbrewer.diverging import Spectral_8 as spec
from palettable.colorbrewer.sequential import Blues_4 as blues
from palettable.colorbrewer.sequential import Greens_3 as greens
	
# import cm colorbars
sys.path.append('/home/irosso/cmocean-master/cmocean')
import cm

# import gsw library
sys.path.append('/home/irosso/gsw-3.0.3')
import gsw
    
HOMEdir     = my_path
plotdir     = os.path.join(HOMEdir,'plots')

folder      = os.path.join(HOMEdir,my_folder)
files       = [f for f in os.listdir(folder) if 'QC' in f]
files.sort()
 
ETOPO       = 'ETOPO1_Ice_g_gmt4.grd'
data        = nc.Dataset(ETOPO)
XC          = data.variables['x'][:]
YC          = data.variables['y'][:]
bathy       = data.variables['z'][:]
bathy       = np.ma.masked_greater(bathy,0.)
bathy       = -bathy 
v           = np.linspace(0,6,7)
x1          = np.min(np.where(XC>=-110))
x2          = np.min(np.where(XC>=-20))
y1          = np.min(np.where(YC>=-70))
y2          = np.min(np.where(YC>=-30))

XC_new      = 360.+XC[XC<0][x1:x2]

all_floats  = ['9750', '12573', '12575','12543', '0569', '0567', '12545']
North       = all_floats[:3]
South       = all_floats[3:]

plot_traj   = False
plot_seas   = False
plot_mld    = False 
plot_no3    = False
plot_T      = False
plot_S      = False
plot_sect   = False
plot_stat   = False
plot_corrD  = False

# make the colorbar for the trajectory discrete
cmap        = plt.get_cmap('jet', 4)
cmpB        = mpl.colors.ListedColormap(blues.mpl_colors)
cmpG        = mpl.colors.ListedColormap(greens.mpl_colors)

tit         = ['Potential Temperature','Practical Salinity','Nitrate']
units       = [r'[$^{\circ}$C]','',r'[$\mu$mol kg$^{-1}$]']
#----------------------------------------------------
def plot_Orsi(fig,ax,c):
	frontdir  = '/data/irosso/Orsi_fronts'
	fronts    = ['pf', 'stf','saf', 'saccf', 'sbdy']
	col_FF    = ['k', 'k', 'k', 'k', 'k']
	ll1       = [100,161,170,120,140]
	ll2       = [-50, -47, -57, -64, -65]
	xRoll     = 0.
	# plot Orsi's fronts
	for ii,ff in enumerate(fronts[::-1]):
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
					ax.plot(lon_FF2, lat_FF2,col,alpha=0.3,linewidth=4)
					ax.plot(lon_FF2, lat_FF2,'k',alpha=0.3,linewidth=0.5)
					lon_FF = []
					lat_FF = []
		
		data.close()
		#ax.annotate('%s' %(fronts[ii+1]),xy=(ll1[ii+1],ll2[ii+1]),fontsize=12)
	
	return fig
#----------------------------------------------------
def plot_PF(fig,ax,c):
	file = 'Polar_Front_weekly_NFreeman.nc'
	data = nc.Dataset(file)
	PF   = data.variables['PFw'][:]
	long = data.variables['longitude'][:]
	PFm  = np.ma.masked_invalid(PF)
	PFmean= np.nanmean(PFm,axis=0)
	PFvar= np.nanvar(PFm,axis=0)
	PFstd= np.sqrt(PFvar)

	x1     = np.min(np.where(long>=0))
	x2     = np.min(np.where(long>=359))
	
	PFmean = PFmean[x1:x2]
	PFstd  = PFstd[x1:x2]
	
	ax.plot(long[x1:x2],PFmean,color=c,alpha=0.5,linewidth=4)
	ax.plot(long[x1:x2],PFmean,color='k',alpha=0.5,linewidth=0.5)
	ax.plot(long[x1:x2],PFmean+PFstd,color=c,linestyle='--',alpha=0.3,linewidth=0.5)
	ax.plot(long[x1:x2],PFmean-PFstd,color=c,linestyle='--',alpha=0.3,linewidth=0.5)
	#ax.set_xlim(long[x1],long[x2])
	
	return fig
#----------------------------------------------------	
def plot_base_map(figN):
	fig = plt.figure(figN,figsize=(12,7))
	ax  = fig.add_subplot(111)
	im  = ax.contourf(XC_new, YC[y1:y2],bathy[y1:y2,x1:x2]/1000,levels=v,cmap=plt.cm.Greys,alpha=0.4)
	ax.contour(XC_new, YC[y1:y2],bathy[y1:y2,x1:x2],[0,10,20],colors='k',alpha=0.5)
	cax = fig.add_axes([0.15, 0.8, 0.2, 0.02])
	cbar= plt.colorbar(im,cax=cax,orientation='horizontal')
	cbar.ax.set_xlabel('ocean depth [km]')
	cbar.ax.xaxis.set_tick_params(color='k')
	#plot_Orsi(fig,ax,'m')
	plot_PF(fig,ax,'c')
	ax.set_xlim(XC_new[0],XC_new[-1])
	ax.set_ylim(YC[y1],YC[y2])
	
	return [fig,ax]
#----------------------------------------------------	
def plot_map_traj(fig,ax,floatN,lon,lat,col,saveFig):
	ax.plot(lon,lat,linewidth=4,color=col,alpha=0.8)
	ax.plot(lon,lat,linewidth=0.3,color='k',alpha=0.8)
	ax.scatter(lon[0],lat[0],s=100,color=col,edgecolor='k',alpha=0.8,zorder=1000)
	ax.annotate('%s' %(floatN), xy=(lon[0]+2,lat[0]),fontsize=16,zorder=2000)#,color='w',bbox=dict(boxstyle="round", alpha=0.5))
	if saveFig:
		outfile = os.path.join(plotdir,'DP_SOCCOM_floats_traj.png')
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
		
	return [fig,ax]
#----------------------------------------------------	
def plot_map_seas(fig,ax,lon,lat,mm,saveFig):
	idxW = np.where(np.logical_or(mm==2,mm==3))
	idxC = np.where(np.logical_or(mm==1,mm==4))
	ax.plot(lon,lat,linewidth=0.2,color='k',alpha=0.8)
	# when plotting all the seasons
	#im1        = ax.scatter(lon,lat,c=mm,s=100,cmap=cmap,edgecolor='face',alpha=0.5)
	#im1.set_clim(1,4)
	# plotting warm vs cold months
	ax.scatter(lon[idxW],lat[idxW],c='r',s=100,edgecolor='face',alpha=0.5)
	ax.scatter(lon[idxC],lat[idxC],c='b',s=100,edgecolor='face',alpha=0.5)
	if numFF == 0:
		cax = fig.add_axes([0.65, 0.2, 0.2, 0.02])
		# if plotting the 4 seasons:
		#cbar= fig.colorbar(im1,cax=cax,orientation='horizontal',ticks=np.arange(1,5))
		# else:
		im1 = ax.scatter([np.nan,np.nan],[np.nan,np.nan],c=[1,2],cmap=plt.get_cmap('jet',2),s=100,edgecolor='face',alpha=0.5)
		cbar= fig.colorbar(im1,cax=cax,orientation='horizontal',ticks=np.arange(1,3))
		cbar.ax.xaxis.set_tick_params(color='k')
		# if plotting the 4 seasons:
		#cbar.ax.set_xticklabels(['Spring','Summer','Autumn','Winter'])  
		#else:
		cbar.set_ticks(np.arange(1,3))
		cbar.set_ticklabels(['Cold','Warm'])
		#ax.text(315, -67, 'Cold', fontsize=14)
		#ax.text(326, -67, 'Warm', fontsize=14)
		#cbar.ax.set_xticklabels([''])
	if saveFig:
		outfile = os.path.join(plotdir,'floats_traj_seasons.png')
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
		
	return fig
#----------------------------------------------------
def plot_map_ff(figN,lon,lat,ff,saveFig,level,field):
	fig = plt.figure(figN)
	ax  = plt.subplot(111)
	if level == 'diff':
		cmap = 'coolwarm'
		tt   = '(MLD - surface)'
		if field=='no3':
			lim1 = -0.35
			lim2 = 0.35
		elif field=='PT':
			lim1 = -0.2
			lim2 = 0.06
		else:
			lim1 = -0.05
			lim2 = 0.02
	else:
		if level == 'surf':
			tt = 'Surface'
		else:
			tt = 'MLD'
		if field=='no3':	
			cmap = cmap = 'YlGnBu'
			lim1 = 5
			lim2 = 30
		elif field=='PT':	
			cmap = 'RdYlBu_r'
			lim1 = -2
			lim2 = 15
		elif field=='SP':	
			cmap = 'YlGn'
			lim1 = 33.5
			lim2 = 34.5
		
	if field=='no3':	
		lab  = 'Nitrate'
		unit = '[$\mu$mol kg$^{-1}$]'
	elif field=='PT':
		lab  = 'Potential Temperature'
		if level == 'diff':
			unit = r'$\times 10^{-2}$ [$^{\circ}$C]'
		else:
			unit = r'[$^{\circ}$C]'
	else:
		lab  = 'Practical Salinity'
		if level == 'diff':
			unit = r'$\times 10^{-2}$'
		else:
			unit = ''

	print tt,cmap,lab
	
 	ax.plot(lon,lat,linewidth=0.5,color='k',alpha=0.6)
	#idx   = np.where(~np.isnan(ff))
	#im1   = ax.scatter(lon[idx],lat[idx],c=ff[idx],s=100,edgecolor='face',alpha=0.5)
	im1   = ax.scatter(lon,lat,c=ff,s=100,cmap=cmap,edgecolor='face',alpha=0.7)
	im1.set_clim(lim1,lim2)
	if saveFig:
		cax = fig.add_axes([0.65, 0.2, 0.2, 0.02])
		cbar= fig.colorbar(im1,cax=cax,orientation='horizontal')
		cbar.ax.xaxis.set_tick_params(color='k')
		cbar.ax.set_xlabel(r'%s %s' %(lab,unit))
		cbar.set_ticks(np.linspace(lim1,lim2,6))
		ax.set_title(r'%s %s' %(tt,lab))
		outfile = os.path.join(plotdir,'traj_%s_%s.png' %(field,level))
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
		#plt.close()
		
	return [fig,ax]
#----------------------------------------------------
def plot_map_mld(figN,lon,lat,ff,saveFig):
	fig = plt.figure(figN)
	ax  = plt.subplot(111)
 	ax.plot(lon,lat,linewidth=0.5,color='k',alpha=0.6)
	im1   = ax.scatter(lon,lat,c=ff,s=100,cmap=spec.mpl_colormap,edgecolor='face',alpha=0.7)
	im1.set_clim(0,200)
	if saveFig:
		cax = fig.add_axes([0.65, 0.2, 0.2, 0.02])
		cbar= fig.colorbar(im1,cax=cax,orientation='horizontal')
		cbar.ax.xaxis.set_tick_params(color='k')
		cbar.ax.set_xlabel('MLD [m]')
		cbar.set_ticks(np.linspace(0,200,6))
		ax.set_title('Mixed Layer Depth')
	
		outfile = os.path.join(plotdir,'DP_SOCCOM_floats_traj_mld.png')
		print outfile
		plt.savefig(outfile, bbox_inches='tight',dpi=200)
		
	return [fig,ax]	
#----------------------------------------------------
def plot_section(press_pre,Nprof,PT_pre,SP_pre,NO3_pre):
	# for plotting make 2D profile array
	Nprof2D   = np.nan*np.ones(press_pre.shape)
	for nn in range(len(Nprof)):
		Nprof2D[:,nn]=[Nprof[nn]]*len(press_pre[:,0])
	vs  = np.linspace(26,27,5)
	vs  = np.append(vs,np.linspace(27,28,3))
	fig = plt.figure(figsize=(17,7))
	ax1 = plt.subplot(131)
	ax2 = plt.subplot(132)
	ax3 = plt.subplot(133)
	for nn in range(len(mld)):
		im=ax1.scatter(len(press_pre[:,nn])*[nn],press_pre[:,nn],c=SP_pre[:,nn],s=50,edgecolor='face')
		im.set_clim(np.nanmin(SP_pre),np.nanmax(SP_pre))
		if nn == 0:
			plt.colorbar(im,ax=ax1)
		im=ax2.scatter(len(press_pre[:,nn])*[nn],press_pre[:,nn],c=PT_pre[:,nn],s=50,edgecolor='face')
		im.set_clim(np.nanmin(PT_pre),np.nanmax(PT_pre))
		if nn == 0:
			plt.colorbar(im,ax=ax2)
		im=ax3.scatter(len(press_pre[:,nn])*[nn],press_pre[:,nn],c=NO3_pre[:,nn],s=50,edgecolor='face')
		im.set_clim(np.nanmin(NO3_pre),np.nanmax(NO3_pre))
		if nn == 0:
			plt.colorbar(im,ax=ax3)
		
	axx = [ax1,ax2,ax3]
	for jj,ax in enumerate(axx):
		cs = ax.contour(Nprof2D,press_pre,sigma0,v=vs,colors='k',linewidths=0.5)
		plt.clabel(cs,inline=0,fmt='%3.2f',fontsize=14,alpha=0.4)
		ax.plot(mld,color='m',linewidth=3)	
		ax.invert_yaxis()
		ax.set_ylim(2000,0)
		ax.set_xlim(0,nn)
		ax.set_title('%s' %tit[jj])
	plt.suptitle('Float #%s' %(floatN))

	outfile = os.path.join(plotdir,'%s_section.png' %floatN)
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
#----------------------------------------------------
def mld_gamma(ff,press):
    MLD    = np.zeros(ff.shape[1])
    levMLD = np.zeros(ff.shape[1])
    for ii in range(ff.shape[1]):
    	idx = np.min(np.where(~np.isnan(sigma0[:,ii])))
    	ff_ref = ff[idx,ii]
    	for kk in range(ff.shape[0]-1):
			if np.abs(ff_ref - ff[kk,ii])>=0.03:
				MLD[ii]    = press[kk,ii]
				levMLD[ii] = kk
				break
    return [MLD,levMLD]
#----------------------------------------------------
def plot_depth_stat(field,PP,month,deep,season):
	fig   = plt.figure(70,figsize=(15,7))	
	ax1   = plt.subplot(131)	
	ax2   = plt.subplot(132)	
	ax3   = plt.subplot(133)	
	ax    = [ax1,ax2,ax3]
	for ii in range(len(field)):
		ff   = field[ii]
		fN   = ff[0,...]
		fS   = ff[1,...]
		if season:
			lab2  = 'seas'
			idxWN = np.where(np.logical_or(month[0,...]==2,month[0,...]==3))[0][:]
			idxCN = np.where(np.logical_or(month[0,...]==1,month[0,...]==4))[0][:]
			idxWS = np.where(np.logical_or(month[1,...]==2,month[1,...]==3))[0][:]
			idxCS = np.where(np.logical_or(month[1,...]==1,month[1,...]==4))[0][:]
			fNW   = fN[:,idxWN]
			fNC   = fN[:,idxCN]
			fSW   = fS[:,idxWS]
			fSC   = fS[:,idxCS]
			# extract only the non nan
			warm  = [fNW,fSW]
			cold  = [fNC,fSC]
			all   = [warm,cold]
			iG    = 1
			iB    = 2
			lnst  = ['-','--']
			for s1,ss in enumerate(all):
				f1    = np.nanmedian(ss[0],axis=1)
				f2    = np.nanmedian(ss[1],axis=1)
				e1    = np.nanstd(ss[0],axis=1)
				e2    = np.nanstd(ss[1],axis=1)
				f3    = f1[~np.isnan(f1)]
				f4    = f2[~np.isnan(f2)]
				e3    = e1[~np.isnan(f1)]
				e4    = e2[~np.isnan(f2)]
				# figure
				ax[ii].plot(f3,PP[~np.isnan(f1)],color='g',linewidth=2,alpha=1,linestyle=lnst[s1])
				ax[ii].fill_betweenx(PP[~np.isnan(f1)], f3-e3, f3+e3,alpha=0.4, edgecolor=cmpG(iG), facecolor=cmpG(iG), linewidth=2, linestyle='-', antialiased=True)
				ax[ii].plot(f4,PP[~np.isnan(f2)],color='b',linewidth=2,alpha=1,linestyle=lnst[s1])
				ax[ii].fill_betweenx(PP[~np.isnan(f2)], f4-e4, f4+e4,alpha=0.4, edgecolor=cmpB(iB), facecolor=cmpB(iB), linewidth=2, linestyle='-', antialiased=True)
				iG += 1
				iB += 1
				if s1==0:
					im1, = ax[0].plot([np.nan],color='k',linestyle='-',label='Summer + Autumn',linewidth=2)
					im2, = ax[0].plot([np.nan],color='k',linestyle='--',label='Winter + Spring',linewidth=2)
					leg  = ax[0].legend(loc=3,handles=[im1,im2],fancybox=True)
					leg.get_frame().set_alpha(0.5)
		else:
			lab2  = 'all'
			f1    = np.nanmedian(fN,axis=1)
			f2    = np.nanmedian(fS,axis=1)
			e1    = np.nanstd(fN,axis=1)
			e2    = np.nanstd(fS,axis=1)
			f3    = f1[~np.isnan(f1)]
			f4    = f2[~np.isnan(f2)]
			e3    = e1[~np.isnan(f1)]
			e4    = e2[~np.isnan(f2)]
			# figure
			ax[ii].plot(f3,PP[~np.isnan(f1)],color=cmpG(1),linewidth=2,alpha=1)
			ax[ii].fill_betweenx(PP[~np.isnan(f1)], f3-e3, f3+e3,alpha=0.4, edgecolor=cmpG(1), facecolor=cmpG(1), linewidth=2, linestyle='-', antialiased=True)
			ax[ii].plot(f4,PP[~np.isnan(f2)],color=cmpB(2),linewidth=2,alpha=1)
			ax[ii].fill_betweenx(PP[~np.isnan(f2)], f4-e4, f4+e4,alpha=0.4, edgecolor=cmpB(2), facecolor=cmpB(2), linewidth=2, linestyle='-', antialiased=True)
		ax[ii].invert_yaxis()
		if deep:
			ax[ii].set_ylim(2000,0)
			lab1 = 'deep'
		else:
			ax[ii].set_ylim(500,0)
			lab1 = 'top'
		ax[ii].set_title('%s' %tit[ii],fontsize=16)
		if ii == 0:
			ax[ii].set_ylabel('pressure [db]',fontsize=14)			
		ax[ii].set_xlabel('%s' %units[ii],fontsize=14)		
		ax[ii].grid('on')
		
	outfile = os.path.join(plotdir,'depth_stat_%s_%s.png' %(lab1,lab2))
	print outfile
	plt.savefig(outfile, bbox_inches='tight',dpi=200)
	plt.close()
#----------------------------------------------------
# main program
if __name__ == "__main__":
	iN = 0
	iS = 0
	l1 = l1N = l1S = 0
	saTot   = np.nan*np.ones((2,2000,200))
	tempTot = np.nan*np.ones((2,2000,200))
	no3Tot  = np.nan*np.ones((2,2000,200))
	mmTot   = np.nan*np.ones((2,200))
	for numFF,ff in enumerate(files):	
		ntot = 1
		dateNum   = []	
		print '\n', ff
		file      = os.path.join(folder,ff)
		floatN    = ff.split('_')[1]
		print floatN
		WMOID     = loadmat(file)['WMOID'][0]
		press_pre = loadmat(file)['pr']
		lat       = loadmat(file)['lat'].transpose()
		lon       = loadmat(file)['lon'].transpose()
		date      = loadmat(file)['date'][:]
		datee     = [datetime.strptime(da,'%m/%d/%Y') for da in date]
		print date[0],date[-1]
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

		lat       = lat[:,0]
		lon       = lon[:,0]
		lat       = np.ma.masked_less(lat,-90)
		lon       = np.ma.masked_less(lon,-500)
		lon[lon>360.] = lon[lon>360.]-360.
		Nprof     = np.linspace(1,len(lat),len(lat))

		# clean coordinates
		lonm      = np.ma.masked_less(lon,-1000)
		latm      = np.ma.masked_less(lat,-1000)
		# set the spatial boundary
		iB        = np.where(np.logical_and((np.logical_and(lonm>=288,lonm<=304)),np.logical_and(latm>=-66,latm<=-53)))[0][:]
		lonB      = lonm[iB]
		latB      = latm[iB]
		monthsB   = mm[iB]
		# plot traj
		if plot_traj:
			if numFF == len(files)-1:
				saveFig = True
			elif numFF == 0:				
				[figT,axT] = plot_base_map(50)
				saveFig = False
			if floatN in North:
				col = cmpG(iN)
				iN +=1
			else:
				col = cmpB(iS)
				iS += 1
			[figT,axT] = plot_map_traj(figT,axT,floatN,lonm,latm,col,saveFig)
			
		# plot seasons
		if plot_seas:
			if numFF == len(files)-1:
				saveFig = True
			elif numFF == 0:
				[figS,axS] = plot_base_map(30)
				saveFig = False
			[figS,axS] = plot_map_seas(figS,axS,lonm,latm,mm,saveFig)
		
		# load data
		SP_pre     = loadmat(file)['sa']
		PT_pre     = loadmat(file)['th']
		NO3_pre    = loadmat(file)['NO3']
	
		# invert for pressure if float is recorded from bottom to top
		if any(press_pre[:10,0]>500.):
			press_pre = press_pre[::-1,:]
			SP_pre    = SP_pre[::-1,:]
			PT_pre    = PT_pre[::-1,:]
			NO3_pre   = NO3_pre[::-1,:]
			
		# interpolate in pressure
		press_pre = press_pre[:,iB]
		SP_pre    = SP_pre[:,iB]
		PT_pre    = PT_pre[:,iB]
		NO3_pre   = NO3_pre[:,iB]
		
		press     = np.arange(2,2002,1)
		saInt     = np.nan*np.ones((len(press),len(iB)),'>f4')
		tempInt   = np.nan*np.ones((len(press),len(iB)),'>f4')	
		no3Int    = np.nan*np.ones((len(press),len(iB)),'>f4')	
		ff_pre    = [SP_pre,PT_pre,NO3_pre]
		for ii,ffP in enumerate(ff_pre):
			for nn in range(press_pre.shape[1]):
				ffP0 = np.ma.masked_invalid(ffP[:,nn])
				# only use non-nan values, otherwise it doesn't interpolate well
				try:
					#f1 = ffP[:,nn][ffP[:,nn].mask==False]
					#f2 = press_pre[:,nn][ffP[:,nn].mask==False]
					f1 = ffP0[ffP0.mask==False]
					f2 = press_pre[:,nn][ffP0.mask==False]
				except:
					f1 = ffP0
					f2 = press_pre[:,nn]
				if len(f1)==0:
					f1 = ffP0
					f2 = press_pre[:,nn]
				try:
					sp = interpolate.interp1d(f2,f1,kind='slinear', bounds_error=False, fill_value=np.nan)
					ff_int = sp(press) 
					if ii == 0:
						saInt[:,nn]   = ff_int
					elif ii == 1:
						tempInt[:,nn] = ff_int
					elif ii == 2:
						no3Int[:,nn]  = ff_int
				except:
					print 'At profile number %i, the float %s has only 1 record valid: len(f2)=%i' %(nn,ff_SO,len(f2))
							
		saInt[saInt==0]    = np.nan
		tempInt[tempInt==0]= np.nan
		no3Int[no3Int==0]  = np.nan
		sigma0Int          = gsw.sigma0(saInt,tempInt)
		
		if floatN in North:
			idxF = 0
			l1   = l1N
		else:
			idxF = 1
			l1   = l1S
		saTot[idxF,:,l1:l1+len(iB)]   = saInt
		tempTot[idxF,:,l1:l1+len(iB)] = tempInt
		no3Tot[idxF,:,l1:l1+len(iB)]  = no3Int
		mmTot[idxF,l1:l1+len(iB)]     = monthsB
		if floatN in North:
			l1N  += len(iB)
		else:
			l1S  += len(iB)
		
		# plot median with std envelope, highlighting North vs South, Cold vs Warm
		if plot_stat:
			if numFF == len(files)-1:
				plot_depth_stat([tempTot,saTot,no3Tot],press,mmTot,deep=False,season=False)
				plot_depth_stat([tempTot,saTot,no3Tot],press,mmTot,deep=False,season=True)		
		
		###### SURFACE ######
		# plot map of surface (~8 m) no3,T,S
		no3_surf  = np.nan*np.ones(NO3_pre.shape[1])
		T_surf    = np.nan*np.ones(NO3_pre.shape[1])
		S_surf    = np.nan*np.ones(NO3_pre.shape[1])		
		for nn in range(NO3_pre.shape[1]):
			# no3
			try:
				i1  = np.max(np.where(press_pre[:,nn]<=8.))
				no3_surf[nn] = np.nanmean(NO3_pre[:i1+1,nn])
			except:
				no3_surf[nn] = np.nan
			# pt
			try:
				i1  = np.max(np.where(press_pre[:,nn]<=8.))
				T_surf[nn] = np.nanmean(PT_pre[:i1+1,nn])
			except:
				no3_surf[nn] = np.nan
			# sp
			try:
				i1  = np.max(np.where(press_pre[:,nn]<=8.))
				S_surf[nn] = np.nanmean(SP_pre[:i1+1,nn])
			except:
				S_surf[nn] = np.nan
		
		# sigma_theta
		sigma0    = gsw.sigma0(SP_pre,PT_pre) 


		##### MLD #####
		# calc MLD
		[mld,KKmld] = mld_gamma(sigma0,press_pre)
		KKmld       = [int(kk) for kk in KKmld]
		
		# plot map of MLD
		if plot_mld:
			if numFF == 0:
				saveFig = False
				[fig100,ax100] = plot_base_map(100)
			elif numFF == len(files)-1:
				saveFig = True
			[fig100,ax100] = plot_map_mld(100,lonm,latm,mld,saveFig)
		
		# save no3,T,S above MLD
		no3_mld   = np.nan*np.ones(NO3_pre.shape[1])
		T_mld     = np.nan*np.ones(NO3_pre.shape[1])
		S_mld     = np.nan*np.ones(NO3_pre.shape[1])
		for nn in range(NO3_pre.shape[1]):
			no3_mld[nn] = np.nanmean(NO3_pre[:KKmld[nn]-2,nn])
			T_mld[nn]   = np.nanmean(PT_pre[:KKmld[nn]-2,nn])
			S_mld[nn]   = np.nanmean(SP_pre[:KKmld[nn]-2,nn])
			
		# compute diff (mld-surf)
		no3_diff   = no3_mld-no3_surf
		T_diff     = T_mld-T_surf
		S_diff     = S_mld-S_surf
		
		# plot maps
		if plot_no3:
			if numFF == 0:
				[fig1,ax1] = plot_base_map(1)
				#[fig2,ax2] = plot_base_map(2)
				[fig3,ax3] = plot_base_map(3)
				saveFig = False
			elif numFF == len(files)-1:
				saveFig = True
				
			[fig1,ax1] = plot_map_ff(1,lonm,latm,no3_surf,saveFig,'surf','no3')
			#[fig2,ax2] = plot_map_ff(2,lonm,latm,no3_mld,saveFig,'mld','no3')
			[fig3,ax3] = plot_map_ff(3,lonm,latm,no3_diff,saveFig,'diff','no3')
			
		if plot_T:
			if numFF == 0:
				[fig4,ax4] = plot_base_map(4)
				#[fig5,ax5] = plot_base_map(5)
				[fig6,ax6] = plot_base_map(6)
				saveFig = False
			elif numFF == len(files)-1:
				saveFig = True
				
			[fig4,ax4] = plot_map_ff(4,lonm,latm,T_surf,saveFig,'surf','PT')
			#[fig5,ax5] = plot_map_ff(5,lonm,latm,T_mld,saveFig,'mld','PT')
			[fig6,ax6] = plot_map_ff(6,lonm,latm,T_diff,saveFig,'diff','PT')
		
		if plot_S:
			if numFF == 0:
				[fig7,ax7] = plot_base_map(7)
				#[fig8,ax8] = plot_base_map(8)
				[fig9,ax9] = plot_base_map(9)
				saveFig = False
			elif numFF == len(files)-1:
				saveFig = True
				
			[fig7,ax7] = plot_map_ff(7,lonm,latm,S_surf,saveFig,'surf','SP')
			#[fig8,ax8] = plot_map_ff(8,lonm,latm,S_mld,saveFig,'mld','SP')
			[fig9,ax9] = plot_map_ff(9,lonm,latm,S_diff,saveFig,'diff','SP')
		
		if plot_sect:
			plot_section(press_pre,Nprof,PT_pre,SP_pre,NO3_pre)

		# correlation plots
		if plot_corrD:
			# 1. surface vs all depths
			f0 = no3_surf
			plt.figure(figsize=(15,7))
			ax1 = plt.subplot(131)
			ax2 = plt.subplot(132)
			ax3 = plt.subplot(133)
			axx = [ax1,ax2,ax3]
			check = 0
			ff_pre = [NO3_pre]#[PT_pre,SP_pre,NO3_pre]
			for fidx,f_pre in enumerate(ff_pre):
				for nn in range(len(no3_surf)):
					f1 = f_pre0[:,nn]
					f3 = press_pre[:,nn]
					f2 = f1[~np.isnan(f1)]
					f3 = f3[~np.isnan(f1)]
					try:
						im = axx[fidx].scatter(ntot,f0[nn],c='white',s=100,marker='*',edgecolor='k',alpha=1,zorder=3000)
					except:
						print 'no surface value'
					try:
						im = axx[fidx].scatter(len(f2)*[ntot],f2,c=f3,s=50,cmap=spec.mpl_colormap,edgecolor='face',alpha=0.6)
						im.set_clim(0,2000)	
						if check==0:
							cbar = plt.colorbar(im,shrink=0.8)
							cbar.ax.xaxis.set_tick_params(color='k')
							cbar.ax.set_xlabel(r'pressure [db]',fontsize=14)
							#cbar.ax.set_title(r'MLD')
							cbar.set_ticks(np.linspace(0,2000,6))
							check = 1
					except:
						print 'empty array'
					ntot += 1
				axx[fidx].grid('on')
				axx[fidx].set_xlim(0,Nprof[-1]+1)
				axx[fidx].set_title('Float #%s' %(floatN),fontsize=16)
				axx[fidx].set_ylabel('%s %s' %(tit[fidx],units[fidx]),fontsize=14)
				#ax.set_xlabel('N',fontsize=14)

			outfile = os.path.join(plotdir,'corr_depths_%s.png' %(floatN))
			print outfile
			plt.savefig(outfile, bbox_inches='tight',dpi=200)
		
		# 2. surface vs mld
		plt.figure(200,figsize=(17,5))
		ax1S = plt.subplot(131)
		ax2S = plt.subplot(132)
		ax3S = plt.subplot(133)
		axS  = [ax1S,ax2S,ax3S]
		ff_surf = [T_surf,S_surf,no3_surf]
		ff_mld  = [T_mld,S_mld,no3_mld]
		for fidx,f_surf in enumerate(ff_surf):
			f1 = f_surf[~np.isnan(f_surf)]
			f2 = ff_mld[fidx][~np.isnan(f_surf)]
			f3 = mld[~np.isnan(f_surf)]
			f1 = f1[~np.isnan(f2)]
			f2 = f2[~np.isnan(f2)]
			f3 = f3[~np.isnan(f2)]

			im = axS[fidx].scatter(f1,f2,c=f3,s=100,cmap=spec.mpl_colormap,edgecolor='face',alpha=0.6)
			im.set_clim(0,200)	
			if numFF == len(files)-1:
				axS[fidx].grid('on')
				if fidx == 0:
					lms = [-3,9]
				elif fidx == 1:
					lms = [33.5,34.3]
				else:
					lms = [17,32]
				axS[fidx].set_xlim(lms[0],lms[-1])
				axS[fidx].set_ylim(lms[0],lms[-1])
				axS[fidx].plot([lms[0],lms[-1]],[lms[0],lms[-1]],'k--',linewidth=2,alpha=0.8)
				axS[fidx].set_title('%s' %(tit[fidx]),fontsize=16)
				axS[fidx].set_xlabel('ML %s %s' %(tit[fidx],units[fidx]),fontsize=14)
				axS[fidx].set_ylabel('Surface %s %s' %(tit[fidx],units[fidx]),fontsize=14)
				
				if fidx==2:
					cbar = plt.colorbar(im,shrink=0.8)
					cbar.ax.xaxis.set_tick_params(color='k')
					cbar.ax.set_xlabel(r'MLD [m]',fontsize=14)
					#cbar.ax.set_title(r'MLD')
					cbar.set_ticks(np.linspace(0,200,6))
					outfile = os.path.join(plotdir,'corr_prop.png')
					print outfile
					plt.savefig(outfile, bbox_inches='tight',dpi=200)
			

###### correlation coefficients boxplot ##### 
## original, no seasons:
fig = plt.figure(figsize=(15,10))
     ax1 = plt.subplot(131)
     ax2 = plt.subplot(132)
     ax3 = plt.subplot(133)
     ax  = [ax1,ax2,ax3]
     ffTot = [tempTot,saTot,no3Tot]
     for iT in range(3):
     	for id in range(2):
     		tmp   = ffTot[iT][id,...]
     		i1    = np.max(np.where(press<=8.))
     		tmp0  = np.nanmean(tmp[:i1+1,:],axis=0)^I
			tmpS  = tmp0[~np.isnan(tmp0)]
			rr    = []
			for nn in range(i1+1,len(press)):
				tmpB  = tmp[nn,:]
				tmpB  = tmpB[~np.isnan(tmp0)]
				tmpB  = tmpB[~np.isnan(tmpB)]
				tmpS1 = tmpS[~np.isnan(tmpB)]
				rr    = np.append(rr,np.corrcoef(tmpS1,tmpB)[0,1])

			#plt.plot(rr,press[i+1:])
			#plt.show()
		
			# make a boxplot
			data = [[rr[i2:i2+10][~np.isnan(rr[i2:i2+10])]] for i2 in range(0,510,10)]

			labels = [np.nanmean(press[i1+1+i2:i1+1+i2+10]) for i2 in range(0,510,10)]
			ax[iT].boxplot(data,vert=0,boxprops=dict(color=cmm[id]),capprops=dict(color=cmm[id]),
							whiskerprops=dict(color=cmm[id]),flierprops=dict(color=cmm[id], markeredgecolor=cmm[id]),
							medianprops=dict(color=cmm[id]))

		ax[iT].invert_yaxis()
		ax[iT].set_yticks(np.linspace(1,51,51)[::10])
		ax[iT].set_yticklabels(labels[::10])
		ax[iT].set_xlim(-0.6,1.1)
		ax[iT].grid('on')
		ax[iT].set_title('%s' %(tit[iT]),fontsize=16)
		if iT == 1:
			ax[iT].set_xlabel('correlation with surface values (8 m)')
		elif iT == 0:
			ax[iT].set_ylabel('pressure [db]',fontsize=14)

outfile = os.path.join(plotdir,'boxplot_prop.png')
print outfile
plt.savefig(outfile, bbox_inches='tight',dpi=200)^I


# with seasons (not working)
fig = plt.figure(figsize=(15,10))
ax1 = plt.subplot(131)
ax2 = plt.subplot(132)
ax3 = plt.subplot(133)
ax  = [ax1,ax2,ax3]
ffTot = [tempTot,saTot,no3Tot]
for id in range(2):
	idxC = np.where(np.logical_or(mmTot[id,...]==1,mmTot[0,...]==4))[0][:]
	idxW = np.where(np.logical_or(mmTot[id,...]==2,mmTot[1,...]==3))[0][:]
	for iT in range(3):
		cmm = [cmpG(0),cmpB(0)]
		for seas in [idxC,idxW]:
			tmp   = ffTot[iT][id,:,seas]
			i1    = np.max(np.where(press<=8.))
			tmp0  = np.nanmean(tmp[:i1+1,:],axis=0)	
			tmpS  = tmp0[~np.isnan(tmp0)]
			rr    = []
			for nn in range(i1+1,len(press)):
				tmpB  = tmp[nn,:]
				tmpB  = tmpB[~np.isnan(tmp0)]
				tmpB  = tmpB[~np.isnan(tmpB)]
				tmpS1 = tmpS[~np.isnan(tmpB)]
				rr    = np.append(rr,np.corrcoef(tmpS1,tmpB)[0,1])
	
			#plt.plot(rr,press[i+1:])
			#plt.show()

			# make a boxplot
			data = [[rr[i2:i2+10][~np.isnan(rr[i2:i2+10])]] for i2 in range(0,510,10)]

			labels = [np.nanmean(press[i1+1+i2:i1+1+i2+10]) for i2 in range(0,510,10)]
			ax[iT].boxplot(data,vert=0,boxprops=dict(color=cmm[id]),capprops=dict(color=cmm[id]),
					whiskerprops=dict(color=cmm[id]),flierprops=dict(color=cmm[id], markeredgecolor=cmm[id]),
					medianprops=dict(color=cmm[id]))
			cmm = [cmpG(2),cmpB(2)]
		if id == 1:	
			ax[iT].invert_yaxis()	
			ax[iT].set_yticks(np.linspace(1,51,51)[::10])
			ax[iT].set_yticklabels(labels[::10])
			ax[iT].set_xlim(-0.6,1.1)
			ax[iT].grid('on')
			ax[iT].set_title('%s' %(tit[iT]),fontsize=16)
			if iT == 1:
				ax[iT].set_xlabel('correlation with surface values (8 m)')
			if iT == 0:
				ax[iT].set_ylabel('pressure [db]',fontsize=14)
	
outfile = os.path.join(plotdir,'boxplot_prop.png')
print outfile
plt.savefig(outfile, bbox_inches='tight',dpi=200)	
	
	
