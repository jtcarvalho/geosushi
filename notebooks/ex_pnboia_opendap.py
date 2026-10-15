from pydap.client import open_url
import matplotlib.pyplot as pl
import numpy as np

buoy = open_url('http://goosbrasil.org:8080/pnboia/Bsantos.nc')
lon = np.array(buoy['longitude'][:].data, dtype=float)
lat = np.array(buoy['latitude'][:].data, dtype=float)

pl.plot(lon, lat)
pl.plot(lon[-1], lat[-1], 'r.', markersize=10)
pl.axis('equal')
pl.show()
