import ngEHTforecast.fisher as fp
import matplotlib.pyplot as plt
import numpy as np
import ehtim as eh


## Create a splined-raster FisherForecast object (themage)
ff = fp.FF_splined_raster(20,60.0)


## Initialize from a FisherForecast object
# Create and plot a smoothed delta ring model
ffinit = fp.FF_smoothed_delta_ring()
pinit = [1.0, 40.0, 10.0]
ffinit.display_image(pinit,limits=50)
plt.text(0.05,0.95,r'FF_smoothed_delta_ring',transform=plt.gca().transAxes,color='w',va='top')
plt.savefig('tutorial_smdr_exp.png',dpi=300)

# Initialize and plot the splined raster
p = ff.generate_parameter_list(ffinit,p=pinit,limits=100)
ff.display_image(p,limits=50)
plt.text(0.05,0.95,r'FF_splined_raster',transform=plt.gca().transAxes,color='w',va='top')
plt.savefig('tutorial_smdr_themage.png',dpi=300)


## Initialize from a FisherForecast child class name
# Create and plot an asymmetric Gaussian for comparison
ffinit = fp.FF_asymmetric_gaussian()
pinit = [1,20,0.5,1.0]
ffinit.display_image(pinit,limits=50)
plt.text(0.05,0.95,r'FF_asymmetric_gaussian',transform=plt.gca().transAxes,color='w',va='top')
plt.savefig('tutorial_asg_exp.png',dpi=300)

# Initialize and plot the splined raster
p = ff.generate_parameter_list(fp.FF_asymmetric_gaussian,p=[1,20,0.5,1.0])
ff.display_image(p,limits=50)
plt.text(0.05,0.95,r'FF_splined_raster',transform=plt.gca().transAxes,color='w',va='top')
plt.savefig('tutorial_asg_themage.png',dpi=300)



## Initialize from a FITS file
# Create an eht.image.Image object and plot for comparison
img = eh.image.load_fits('M87_230GHz.fits')
img = img.blur_circ(np.sqrt(ff.dxcp*ff.dycp))
img.display(cbar_unit=('mJy','$\mu$as$^2$'),has_title=False)
plt.text(0.05,0.95,r'ehtim',transform=plt.gca().transAxes,color='w',va='top')
plt.savefig('tutorial_img_exp.png',dpi=300)

# Initialize and plot the splined raster
p = ff.generate_parameter_list(img)
ff.display_image(p,limits=80)
plt.text(0.05,0.95,r'FF_splined_raster',transform=plt.gca().transAxes,color='w',va='top')
plt.savefig('tutorial_img_themage.png',dpi=300)

# Initialize from the FITS file
p = ff.generate_parameter_list('M87_230GHz.fits')
ff.display_image(p,limits=80)
plt.text(0.05,0.95,r'FF_splined_raster',transform=plt.gca().transAxes,color='w',va='top')
plt.savefig('tutorial_fits_themage.png',dpi=300)

