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
from WFC3_phot_tools.data_tools.get_wfc3_data_astroquery import *
from WFC3_phot_tools.utils.UVIS_PAM import *
from WFC3_phot_tools.utils.daophot_err import *

from pyql.database.ql_database_interface import session
from pyql.database.ql_database_interface import Master
from pyql.database.ql_database_interface import UVIS_flt_0, UVIS_spt_0


import warnings
warnings.filterwarnings("ignore")

######################################################################################
######  Set parameters here before running any of the functions in this module. ######
######################################################################################
######################################################################################
### ------------------------------ Set Paths -----------------------------------------
DATA_DIR = '/Users/dsom/workarea/WFC3_projects/UVIS_spatscan/trials/trial_data/' # directory that data should be sorted into
PHOT_TABLE_DIR = '/Users/dsom/workarea/WFC3_projects/UVIS_spatscan/trials/trial_data/output/' # directory where catalogs should be output
PAM_DIR = '/grp/hst/wfc3p/cshanahan/phot_group_work/pixel_area_maps/' # directory containing pixel area maps
### ----------------------------------------------------------------------------------
### ---------------- Specify datasets and data-handling options ----------------------
### ----------------------------------------------------------------------------------
PROP_IDS = [14878, 15398, 15583, 16021] # list of proposal ids, used when downloading data.
NEWDAT = True # if True, look for new data from specified proposals and download.
SORTDAT = True # if True, sort downloaded data else data remain in the 'new' directory.
PROC_OBJS = 'all' # object name -or- list of object names in the downloaded data to be processed.
PROC_FILTS = 'all' # filter name -or- list of filter names in the downloaded data to be processed.
FILE_TYPE = 'flt' # input file type (flt, flc, raw, etc..) to be processed.
### ----------------------------------------------------------------------------------
### ----------------------- Specify analysis parameters ------------------------------
### ----------------------------------------------------------------------------------
CRREJ = True # if True, run CR rejection on input files.
CRREPROC = False # if True, run all data through CR rejection process including already processed data.
RUN_APPHOT = True # if True, run aperture photometry on data
AP_DIMENSIONS = [(36, 240)] # list of desired aperture dimensions (x dimension, y dimension) to use for aperture photometry.
SKY_AP_DIMENSION = (75, 350) # list of desired sky aperture dimensions (x dimension, y dimension) to use for aperture photometry.
BACK_METHOD = 'mean' # background determination method: 'mean' or 'median'
NCORES = 20 # specify number of processor cores to use
SHOW_AP = False # if True, plot each aperture
######################################################################################
######################################################################################

if CRREJ:
	FTYPE_APPHOT = 'fcr'
else:
	FTYPE_APPHOT = FILE_TYPE

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

def _retrieve_scan_data_astroquery(prop_id, fits_file_type, data_dir):

    """ Copies spatial scan files from quicklook directories to `data_dir`/new.
        Only copies files that aren't sorted into a subdirectory in `data_dir`
        already. """

    print('Retrieving data from proposal {}'.format(str(prop_id)))

    results = session.query(Master.rootname).join(UVIS_flt_0).join(UVIS_spt_0).\
              filter(UVIS_flt_0.proposid == prop_id).filter(UVIS_spt_0.scan_typ != 'N').all()
    all_scan_rootnames = [item[0] for item in results]

    # now compare list against files already retrieved/sorted
    existing_filenames = [os.path.basename(x)[0:9] for x in _get_existing_filenames(data_dir, fits_file_type)]

    # sometimes files have a 'j' or 's'. replace this with a q. i don't know why this is - failed obs?
    new_file_rootnames =  [item[0:8] + 'q' for item in list(set(all_scan_rootnames) - set(existing_filenames))]
    print(f'Found {len(new_file_rootnames)} un-ingested files in QL database.')

    # query astroquery
    query_results = query_by_data_id(new_file_rootnames, file_type=fits_file_type)

    print(f'Found {len(query_results)} results in Astroquery. Downloading.')

    # download data
    download_products(query_results, output_dir=os.path.join(data_dir, 'new'))



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

def main_process_scan_UVIS(get_new_data=False, sort_new_data=True,
						   run_cr_reject=True, cr_reprocess=False,
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

	if get_new_data:
		for id in PROP_IDS: # data now is in two directories...ugh
			_retrieve_scan_data_astroquery(id, FILE_TYPE, DATA_DIR)

	if sort_new_data:
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
					# p = Pool(n_cores)
					# p.map(make_crcorr_file_scan_wfc3, cr_input)
					for f in cr_input:
						print(f)
						make_crcorr_file_scan_wfc3(f)
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
	main_process_scan_UVIS(get_new_data=NEWDAT, sort_new_data=SORTDAT,
						   run_cr_reject=CRREJ, cr_reprocess=CRREPROC,
						   run_ap_phot=RUN_APPHOT, ap_phot_file_type=FTYPE_APPHOT,
						   n_cores=NCORES, process_objs=PROC_OBJS,
						   process_filts=PROC_FILTS, show_ap_plot=SHOW_AP)
