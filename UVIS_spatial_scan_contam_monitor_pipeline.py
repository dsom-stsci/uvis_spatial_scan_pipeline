""" 

	Runs all steps to generate photometry catalogs from UVIS spatial scan data.

	Author
	------
	Clare Shanahan, 2018

"""

import os
import glob
import shutil

from astropy.io import ascii, fits
from astropy.table import Table
from multiprocessing import Pool

from WFC3_phot_tools.spatial_scan.cr_reject import *
from WFC3_phot_tools.spatial_scan.phot_tools import *
from WFC3_phot_tools.data_tools.sort_data import *
from WFC3_phot_tools.utils.UVIS_PAM import *
from WFC3_phot_tools.utils.daophot_err import *


import warnings
warnings.filterwarnings("ignore")

###################################################################################### 
######  Set parameters here before running any of the functions in this module. ######
###################################################################################### 
DATA_DIR = '/grp/hst/wfc3p/cshanahan/phot_group_work/data/scan_demo_data/' # directory that data should be sorted into 
PHOT_TABLE_DIR = '/grp/hst/wfc3p/cshanahan/phot_group_work/data/scan_demo_data/output/' # directory where catalogs should be output
PROP_IDS = [15398, 15583, 14878, 16021] # list of proposal ids, used when downloading data.
PAM_DIR = '/grp/hst/wfc3p/cshanahan/phot_group_work/pixel_area_maps/' # directory containing pixel area maps
AP_DIMENSIONS = [(36, 240)] # list of desired apertures (x dimension, y dimension) to use for aperture photometry 
SKY_AP_DIMENSION = (75, 350) 
BACK_METHOD = 'mean' # 'mean' or 'median'
FILE_TYPE = 'flt' # file type (flt, flc, raw, etc..)
###################################################################################### 
###################################################################################### 

def _setup_dirs():
	""" Creates output directories in DATA_DIR, if they don't exist."""

	subdirs = ['new', 'bad', 'data']
	for s in subdirs:
		if not os.path.isdir(os.path.join(DATA_DIR, s)):
			os.makedirs(os.path.join(DATA_DIR, s))
			print(f'Making {os.path.join(DATA_DIR, s)}')

def _get_existing_filenames(data_dir, fits_file_type):

    """Returns a list of roontames of already retrieved file that are in the
    sorted directory files, the 'bad' data directory and the 'new' data
    directory."""
 
    new_data_dir = data_dir + 'new/'
    new_data_files = glob.glob(new_data_dir+'*{}.fits'.format(fits_file_type))
    new_data_filenames = [os.path.basename(f) for f in new_data_files]

    bad_data_dir = data_dir + 'bad/'
    bad_data_files = glob.glob(bad_data_dir+'*{}.fits'.format(fits_file_type))
    bad_data_filenames = [os.path.basename(f) for f in bad_data_files]

    existing_data_dir = data_dir + 'data/'
    existing_data_files = glob.glob(existing_data_dir + \
                         '*/*/*{}.fits'.format(fits_file_type))
    existing_data_filenames = [os.path.basename(f) for f in existing_data_files]

    return(bad_data_filenames + existing_data_filenames + new_data_filenames)

def _retrieve_scan_data_ql(prop_id, fits_file_type, data_dir,
                     ql_dir = '/grp/hst/wfc3a/*/'):

	""" Copies spatial scan files from quicklook directories to `data_dir`/new.
		Only copies files that aren't sorted into a subdirectory in `data_dir` 
		already. """

	print('Retrieving data from proposal {}'.format(str(prop_id)))

	dest_dir = data_dir + 'new/'
	dirr = ql_dir + str(prop_id)
	#glob for files in all visit directories
	files = glob.glob(dirr+'/*/*{}.fits'.format(fits_file_type))

	scan_files = []
	for f in files:
	    scan_typ = fits.getval(f, 'SCAN_TYP')
	    if scan_typ != 'N':
	        scan_files.append(f)

	#now compare list against files already retrieved/sorted
	existing_filenames = _get_existing_filenames(data_dir, fits_file_type)
	scan_file_basenames = [os.path.basename(x) for x in scan_files]

	new_files = [scan_files[i] for i, x in enumerate(scan_file_basenames) if x not in existing_filenames]

	print(f'{len(new_files)} new files to retrieve. Copying files to {dest_dir}.\n')

	for f in new_files:
	  shutil.copy(f, dest_dir+os.path.basename(f))

def get_header_info(hdr, keywords=['rootname', 'proposid', 'date-obs', 
								   'expstart', 'exptime', 'ccdamp', 'aperture', 
								   'flashlvl']):
	header_info = []
	for keyword in keywords:
		header_info.append(hdr[keyword])
	return header_info

def _wrapper_make_phot_table(input_files, show_ap_plot, data_ext):
	""" Runs full photometry process (PAM image, source detection, etc...)
		on input_files, returns an astropy table with photometry infoself.

		All parameters to control source detection, sky subtraction etc
		should be set in this function. For this monitor, they shouldn't change
		much so this is easier than allowing them to be tuned when calling this
		function."""

	phot_tab_colnames = []
	all_phot_info = []

	for f in input_files:
		print('\nRunning photometry on {}'.format(f))
		hdu = fits.open(f)
		data = fits.open(f)[data_ext].data
		hdr = fits.open(f)[data_ext].header
		if data_ext != 0:
			pri_hdr = fits.open(f)[0].header
			hdr = hdr + pri_hdr

		phot_info_cols = ['rootname', 'proposid', 'date-obs', 'expstart', 
						  'exptime', 'ccdamp', 'aperture', 'flashlvl']

		phot_info = get_header_info(hdr, keywords=phot_info_cols)

		source_tbl = detect_sources_scan(copy.deepcopy(data),
								snr_threshold=3.0, n_pixels=1000,
								show=False)

		x_pos = source_tbl['xcentroid'][0].value
		y_pos = source_tbl['ycentroid'][0].value
		theta = -(90 - source_tbl['orientation'][0].value) * (np.pi / 180)
		print(theta)


		print('Detected {} sources at {}, {}.'.format(len(source_tbl), x_pos, 
			                                          y_pos))

		print('Making PAM corrected image in memory.')
		data = make_PAMcorr_image_UVIS(copy.deepcopy(data), hdr, hdr,
			   PAM_DIR)

		# divide by countrate 
		data = data / hdr['EXPTIME']

		back, back_rms = calc_sky(copy.deepcopy(data), x_pos, y_pos, 
													SKY_AP_DIMENSION[1], 
													SKY_AP_DIMENSION[0], 50,
													method = BACK_METHOD)

		sky_ap_area = SKY_AP_DIMENSION[0] * SKY_AP_DIMENSION[1]
		phot_info += [back, back_rms]
		phot_info_cols += ['back', 'back_rms']
		#now iterate through aperture sizes

		# background subtract data -  units are now e-/s, sky subtracted
		data = data - back

		for ap_size in AP_DIMENSIONS:
			w, l = ap_size
			phot_ap_area = w * l

			phot_table = aperture_photometry_scan(data, x_pos, y_pos,
												ap_width=w, ap_length=l,
												theta=theta, show=show_ap_plot,
												plt_title=os.path.basename(f))
			ap_sum = phot_table['aperture_sum'][0]
			print(f'*** Background subtracted countrate in {ap_size} is {ap_sum} ***')

			# need to convert source sum to countrate for error calculation 
			flux_err = compute_phot_err_daophot(ap_sum * hdr['EXPTIME'], 
											    back * hdr['EXPTIME'],
												back_rms * hdr['EXPTIME'], 
												phot_ap_area, sky_ap_area)

			flux_err = flux_err / hdr['EXPTIME']
			phot_info += [ap_sum, flux_err]
			phot_info_cols += ['countrate_{}_{}'.format(w,l), \
							   'err_{}_{}'.format(w,l)]

		phot_tab_colnames = phot_info_cols
		all_phot_info.append(phot_info)

	phot_tab = Table()

	for i, val in enumerate(phot_tab_colnames):
		phot_tab[val] = [item[i] for item in all_phot_info]

	return phot_tab

def main_process_scan_UVIS(get_new_data=False, sort_new_data=False,
						   run_cr_reject=False, cr_reprocess=False,
						   run_ap_phot=True, ap_phot_file_type = 'fcr',
						   n_cores=20, process_objs='all',
						   process_filts='all', show_ap_plot=False):
	
	_setup_dirs()

	objss = process_objs if type(process_objs) in [list, tuple] else [process_objs]
	filtss = process_filts if type(process_filts) in [list, tuple] else [process_filts]

	if process_objs == 'all':
		objss=[os.path.basename(x) for x in glob.glob(DATA_DIR + '/data/*')]
	if process_filts == 'all':
		filtss=list(set([os.path.basename(x) for x in 
					glob.glob(DATA_DIR + '/data/*/*')]))
	print(filtss, objss)

	if get_new_data:
		for id in PROP_IDS: # data now is in two directories...ugh 
			_retrieve_scan_data_ql(id, FILE_TYPE, DATA_DIR)

	if sort_new_data:
		print(os.path.join(DATA_DIR, 'new'))
		sort_data_targname_filt(os.path.join(DATA_DIR, 'new'), 
								os.path.join(DATA_DIR, 'data'),
								file_type=FILE_TYPE,
								targname_mappings={'GD153' : 
												  ['GD153', 'GD-153'],
					 							  'GRW70' : ['GRW+70D5824', 
					 							  'GRW+70D']})
	for objj in objss:
		for filtt in filtss:
			dirr = DATA_DIR + f'data/{objj}/{filtt}/'

			if run_cr_reject:
				print(f'CR rejection {filtt}, {objj}')
				cr_input = glob.glob(dirr+f'*{FILE_TYPE}.fits')
				cr_input = glob.glob(DATA_DIR + f'data/{objj}/{filtt}/*{FILE_TYPE}.fits')

				if cr_reprocess is False:
					existing_fcr = glob.glob(dirr+'/*fcr.fits')
					cr_input = [f for f in cr_input if f.replace(FILE_TYPE, 'fcr') not in \
								existing_fcr]

				if len(cr_input) > 0:
					print('{} new files to cr reject.'.format(len(cr_input)))
					p = Pool(n_cores)
					p.map(make_crcorr_file_scan_wfc3, cr_input)
				else:
					print('All files already CR rejected.')

			if run_ap_phot:

				if ap_phot_file_type == 'fcr':
					data_ext = 0
				elif (ap_phot_file_type == 'flt') or (ap_phot_file_type == 'flc'):
					data_ext = 1
				else:
					return ValueError('ap_phot_file_type must be fcr, flc, or flt.')

				#make a photometry file for each object / filter
				print(f'\n\n *** Photometry {filtt}, {objj} *** \n\n')

				fcrs = glob.glob(dirr + f'/*{ap_phot_file_type}.fits')
				print(f'{len(fcrs)} files for obj = {objj} filter = {filtt}.')

				phot_table = _wrapper_make_phot_table(fcrs, 
													  show_ap_plot=show_ap_plot,
													  data_ext=data_ext)

				output_path = PHOT_TABLE_DIR + '{}_{}_phot.dat'.format(objj, filtt)
				if os.path.isfile(output_path):
					os.remove(output_path)
				print(f'Writing {output_path}\n\n')
				ascii.write(phot_table, output_path, format = 'csv')

if __name__ == '__main__':
	main_process_scan_UVIS(process_objs=['GD153'],
						   process_filts='F218W')
