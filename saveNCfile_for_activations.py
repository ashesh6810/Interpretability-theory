import numpy as np
import netCDF4 as nc4

def savenc_for_activations(x,y,num_level,num_epochs,num_layers,num_samples,num_filters,lon,lat,filename):
    f = nc4.Dataset(filename,'w', format='NETCDF4')
    tempgrp = f.createGroup('Temp_data')
    tempgrp.createDimension('lon', lon)
    tempgrp.createDimension('lat', lat)
    tempgrp.createDimension('num_filters',num_filters)
    tempgrp.createDimension('samples', num_samples)
    tempgrp.createDimension('layers', num_layers)
    tempgrp.createDimension('epochs', num_epochs)
    tempgrp.createDimension('level', )

    
#    longitude = tempgrp.createVariable('Longitude', 'f4', 'lon')
#    latitude = tempgrp.createVariable('Latitude', 'f4', 'lat')  
    output = tempgrp.createVariable('Output', 'f4',('epochs','samples','level','lon','lat') )  
#    levels = tempgrp.createVariable('Levels', 'i4', 'level')
    psi = tempgrp.createVariable('PSI', 'f4', ('epochs','layers','samples','num_filters','lon','lat'))
#    time = tempgrp.createVariable('Time', 'i4', 'time')

    
    psi[:,:,:,:,:,:] = x
    output[:,:,:,:,:] = y
  
    f.close()

