#!/usr/bin/env python3

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pylab as plt
from scipy.signal import detrend
import numpy as np
import pandas as pd
import h5py
from collections import OrderedDict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
import logging
import tqdm
from psrqpy import QueryATNF

def qpsr(sample_ra =6.02363,sample_dec =-72.08128, params = ['BNAME','JNAME', 'RAJ', 'DECJ','DM', 'S1400', 'P0'], condition = None, rad=1):
    if isinstance(sample_ra, str): #allow preformatted ra/dec
        (ra,dec) = (sample_ra, sample_dec)
    else:
        (ra,dec)=deg2HMS(ra=sample_ra, dec=sample_dec)
    c=[ra,dec,rad]
    #c=['00h24m05.67s','-72d04m52.62s',1]
    logging.info(f'Querying with  {c}')
    query = QueryATNF(params=params, condition=condition,
                      circular_boundary=c,include_errs=False)
    return query, query.table[:5]

def deg2HMS(ra='', dec='', round=False):
    (RA, DEC, rs, ds) = ('', '', '', '')
    if str(dec)[0] == '-':
        (ds, dec) = ('-', abs(dec))
    deg = int(dec)
    decM = abs(int((dec - deg) * 60))
    if round:
        decS = int((abs((dec - deg) * 60) - decM) * 60)
    else:
        decS = (abs((dec - deg) * 60) - decM) * 60
    DEC = '{0}{1}d{2}m{3:.2f}s'.format(ds, deg, decM, decS)
    
    if str(ra)[0] == '-':
        (rs, ra) = ('-', abs(ra))
    raH = int(ra / 15)
    raM = int((ra / 15 - raH) * 60)
    if round:
        raS = int(((ra / 15 - raH) * 60 - raM) * 60)
    else:
        raS = ((ra / 15 - raH) * 60 - raM) * 60
    RA = '{0}{1}h{2}m{3:.2f}s'.format(rs, raH, raM, raS)
    return (RA, DEC)

def h5_loction_2_stuff(h5_file, csv_path = None, skip_pointing = False):
    """
    Reads an h5 file into two dictionaries with various fields I'll document later.

    Positional arguments:
    h5_file (String) -- filename and path of h5 file.

    Keyword arguments:
    csv_path (String) -- file path of CSV associated with the h5 file. For manual processing. (Default None)
    skip_pointing (bool)  -- skip pointing data for test purposes (default False)

    Returns (dict, dict) whose fields remain a mystery to me at this moment.
    """

    param_dict=OrderedDict()
    stuff_dict={}
    with h5py.File(h5_file,'r') as file:
        stuff_dict['ft']=detrend(np.array(file['data_freq_time'])[:, ::-1].T)
        stuff_dict['dt']=np.array(file['data_dm_time'])
        stuff_dict['dm'] = float(file.attrs['dm'])
        stuff_dict['snr'] = float(file.attrs['snr'])
        stuff_dict['tcand'] = float(file.attrs['tcand'])
        
    folder = h5_file.split('/')[3]
    if csv_path is None:
        if "gal" in h5_file:
            df = pd.read_csv(f'/ldata/trunk/{folder}/{folder}_gal.csv')
        else:
            df = pd.read_csv(f'/ldata/trunk/{folder}/{folder}.csv')
    else:
        df = pd.read_csv(csv_path)
    df_mask_dm = df['dm'] == stuff_dict['dm']
    df_mask_snr = df['snr'] == stuff_dict['snr']
    df_mask_tcand = df['tcand'] == stuff_dict['tcand']
    # Fix for when the above does not find the value in the df
    for name, mask in zip(("dm", "snr", "tcand"), (df_mask_dm, df_mask_snr, df_mask_tcand)):
        if mask.sum() == 0:
            logging.warning(f"{name} mask didn't find any candidate! Looking for closest match.")
            index = df[name].sub(stuff_dict[name]).abs().idxmin()
            mask[index] = True 
    total_mask = df_mask_dm & df_mask_snr & df_mask_tcand
    row = df[total_mask]
    if not skip_pointing:
        ra, dec = deg2HMS(ra=row['RA_deg'].values[0], dec=row['DEC_deg'].values[0])
        if len(row) > 0:
            param_dict['S/N'] = float(row['snr'].values[0])
            param_dict['Width (ms)'] = float(0.256* 2**row['width'].values[0])
            param_dict['Width (samples)'] = int(2**row['width'].values[0])
            param_dict['DM (pc/cc)'] = float(row['dm'].values[0])
            param_dict['NE2001 DM (pc/cc)'] = float(row['cand_ne2001'].values[0])
            param_dict['YMW16 DM (pc/cc)'] = float(row['cand_ymw16'].values[0])
            param_dict['MJD'] = str(row['cand_mjd'].values[0])
            param_dict['RA (J2000)'] = str(ra)
            param_dict['DEC (J2000)'] = str(dec)
            param_dict['GL (degree)'] = float(row['cand_gl'].values[0])
            param_dict['GB (degree)'] = float(row['cand_gb'].values[0])
            param_dict['Receiver'] = str(row['IFV1TNCI'].values[0])
            param_dict['Project ID'] = str(row['SCPROJID'].values[0])
            param_dict['Observation ID'] = str(folder)
            param_dict['Turret Angle (degree)'] = float(row['ATRXOCTA'].values[0])

            #psrcat search section
            #first pass search; 1 degree radius with no DM or width filtering
            DM = param_dict['DM (pc/cc)']
            filterString = f"(DM > {DM*(0.8)} && DM < {DM*(1.2)})"
            query, psr_table = qpsr(float(row['RA_deg'].values[0]), float(row['DEC_deg'].values[0]))
            if len(psr_table) > 0:
                param_dict['Known Nearby Sources'] = [query, psr_table]
                logging.debug(psr_table)
                #determine what to title the plot as with a filtered search
                query, psr_table = qpsr(float(row['RA_deg'].values[0]), float(row['DEC_deg'].values[0]), condition=filterString)
                if len(psr_table) > 0:
                    param_dict['Title'] = psr_table[0]['BNAME'] if isinstance(psr_table[0]['BNAME'],str) else psr_table[0]['JNAME']
                else: #pulsars in the field, but none at this DM
                    param_dict['Title'] = "Unknown"
            else:
                param_dict['Known Nearby Sources'] = str(None)
                param_dict['Title'] = "Unknown"
                logging.info('No known pulsars in the field')
    else: #if skip_pointing
        if len(row) > 0:
            param_dict['S/N'] = float(row['snr'].values[0])
            param_dict['Width (ms)'] = float(0.256* 2**row['width'].values[0])
            param_dict['Width (samples)'] = int(2**row['width'].values[0])
            param_dict['DM (pc/cc)'] = float(row['dm'].values[0])
            param_dict['NE2001 DM (pc/cc)'] = 0
            param_dict['YMW16 DM (pc/cc)'] = 0
            param_dict['MJD'] = str(row['cand_mjd'].values[0])
            param_dict['RA (J2000)'] = 0
            param_dict['DEC (J2000)'] = 0
            param_dict['GL (degree)'] = 0
            param_dict['GB (degree)'] = 0
            param_dict['Receiver'] = 0
            param_dict['Project ID'] = 0
            param_dict['Observation ID'] = str(folder)
            param_dict['Turret Angle (degree)'] = 0
            param_dict['Known Nearby Sources'] = str(None)

    return stuff_dict, param_dict

def plotem(h5_file,fout=None,nrby=False,skip_pointing=False):
    """
    Generates a plot from an h5 filename.

    Positional arguments:
    h5_file (string) -- filename of h5 file

    Keyword arguments:
    fout (string)         -- filename of output file (default None)
    nrby (bool)           -- change return type (default False)
    skip_pointing (bool)  -- skip pointing data for test purposes (default False)
    
    Returns a string containing the file name of the generated plot if nrby is False
    Else, returns (string, bool) tuple with the same string and a boolean which is True if there is a known nearby source.
    """

    stuff_dict, param_dict = h5_loction_2_stuff(h5_file, skip_pointing=skip_pointing)
    #if skip_pointing, it's a test run so we should write out some of the metadata for checking
    if skip_pointing:
        with open("/ldata/dev_tests/repro/plotOut.txt", "a") as file:
            file.write(f"{str(round(param_dict['DM (pc/cc)'])).ljust(6)} pc/cc {str(round(param_dict['Width (ms)']/(1000),3))} s {h5_file.split('/')[3]}\n")
    #this writes DM, pulse width, and filename to plotOut.txt in a format similar to injection script

    ts = np.linspace(-128,128,num=256)*.256*param_dict['Width (samples)']

    if ts[-1]//1000 > 0:
        ts /= 1000
        ts_label='Time (s)'
    else:
        ts_label='Time (ms)'
    plt.rc('font', family='monospace')
    plt.rc('xtick', labelsize='x-small')
    plt.rc('ytick', labelsize='x-small')

    fig3 = plt.figure(constrained_layout=True,figsize=(9,7))
    gs = fig3.add_gridspec(5, 4)
    
    # DMT plot
    f3_ax3 = fig3.add_subplot(gs[3:5, :2])
    f3_ax3.imshow(stuff_dict['dt'],aspect='auto',
                  interpolation=None,extent=[ts[0],ts[-1],2*param_dict['DM (pc/cc)'],0])
    f3_ax3.set_ylabel('DM (pc/cc)')
    f3_ax3.set_xlabel(ts_label)
    
    # time profile
    f3_ax1 = fig3.add_subplot(gs[0, :2],sharex=f3_ax3)
    f3_ax1.plot(ts,stuff_dict['ft'].sum(0),'k',linewidth=.7)
    f3_ax1.set_ylabel('Arb. Flux')
    plt.setp(f3_ax1.get_xticklabels(), visible=False)

    # FT plot
    f3_ax2 = fig3.add_subplot(gs[1:3, :2],sharex=f3_ax3)
    f3_ax2.imshow(stuff_dict['ft'],aspect='auto',
                  interpolation=None,
                  extent=[ts[0],ts[-1],1919.8828125,959.8828125])
    f3_ax2.set_ylabel('Frequency (Mhz)')
    plt.setp(f3_ax2.get_xticklabels(), visible=False)

    # metadata
    size = 13
    f3_ax4 = fig3.add_subplot(gs[:, 3])
    for idx, pair in enumerate(reversed(param_dict.items())):
        if isinstance(pair[1], str) or  isinstance(pair[1], int):
            f3_ax4.text(-0.75,0.15+ idx/20,f'{pair[0]}: {pair[1]}',size=size, transform=f3_ax4.transAxes)
        elif isinstance(pair[1],list):
            f3_ax4.text(-0.75,0.15+idx/20,f'{pair[0]}:', size=size, transform=f3_ax4.transAxes)
            for psr_idx, psr in enumerate(pair[1][1]):
                if isinstance(psr['BNAME'],str):
                    string=f"{ psr['BNAME']}--{psr['DM']:.2f} pc/cc--{psr['P0']*1000:3.3f} ms--{psr['S1400']:.2f} mJy"
                else:
                    string=f"{ psr['JNAME']}--{psr['DM']:.2f} pc/cc--{psr['P0']*1000:3.3f} ms--{psr['S1400']:.2f} mJy"
                f3_ax4.text(-0.75,0.1-psr_idx/30,string,size=8, transform=f3_ax4.transAxes)
        else:
            f3_ax4.text(-0.75,0.15+idx/20,f'{pair[0]}: {pair[1]:.2f}', size=size, transform=f3_ax4.transAxes)
    f3_ax4.set_axis_off()
    plt.tight_layout()
    plt.title(param_dict['Title'])
    plt.subplots_adjust(hspace=.0)
    plt.subplots_adjust(wspace=.0)
    if fout is None:
        fout = h5_file[:-3]+'.png'
    plt.savefig(fout, bbox_inches='tight')
    try:
        plt.close() #to clean up memory usage
    except:
        pass
    if nrby: #return True as second part of tuple if known source nearby
        return (fout, param_dict['Title'] != "Unknown")
    else: #old function output
        return fout

if __name__ == '__main__':
    parser=ArgumentParser(description='Plot cands', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument('-f', '--files', nargs='+', help='Filterbank file')
    parser.set_defaults(verbose=False)
    values = parser.parse_args()

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=format)
    else:
        logging.basicConfig(level=logging.INFO, format=format)
    for file in tqdm.tqdm(values.files):
        logging.debug(f'plotting {file}')
        plotem(file)
