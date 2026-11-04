#!/usr/bin/env python3

from influx_2df import mjd2influx
import pysigproc
import numpy as np
import glob
import sys
import logging
import json
from astropy.time import Time
from elasticsearch import Elasticsearch
import pylab as plt

def tel_df_to_es(es,influx_df,filterbank, frac_flagged):
    data_validity_mask = influx_df['DATA_VALID']==1
    time_mask = 3600*24*(influx_df['MJD'] - influx_df['MJD'][0]) < 503.31
    total_influx_mask = data_validity_mask & time_mask
    df = influx_df[total_influx_mask]
    gl = np.array(np.round(df['GL_deg']), dtype=np.int32)
    gb = np.array(np.round(df['GB_deg']), dtype=np.int32)
    for _gl in set(gl):
        for _gb in set(gb):
            gl_mask = (gl == _gl)
            gb_mask = (gb == _gb)
            df_mask = gl_mask & gb_mask
            if df_mask.sum() > 0:
                _df = df[df_mask]
                time = Time(_df['MJD'][0], format='mjd')
                dump_dict={}
                time_spent = len(_df) #24*3600*(np.sum(np.diff(np.sort(_df['MJD'])))) #24*3600*(np.max(_df['MJD']) - np.min((_df['MJD'][0])))
                logging.info(f'Spent {time_spent}s at {_gl} gl and {_gb} gb')
                dump_dict['type'] = "Obs"
                dump_dict['Filename'] = filterbank
                dump_dict['Eq'] = {"lon" : _df['RA_deg'].mean()-180, "lat": _df['DEC_deg'].mean()}
                dump_dict['Gal'] = {"lon" :_df['GL_deg'].mean()-180, "lat": _df ['GB_deg'].mean()}
                dump_dict['MJD'] = _df['MJD'][0]
                dump_dict['UTC'] = time.isot
                dump_dict['Receiver'] = _df['IFV1TNCI'][0]
                dump_dict['Project_ID'] = _df['SCPROJID'][0]
                dump_dict['Time_spent'] = time_spent
                dump_dict['Fraction_flagged'] = frac_flagged
                result = es.index(index='greenburst',doc_type='_doc',id=time.isot,body=json.dumps(dump_dict))
                logging.info(result)



if __name__ == "__main__":
    logging_format = '%(asctime)s - %(funcName)s -%(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=logging_format)
    es=Elasticsearch([{'host':'localhost','port':9200}])

    for filterbank in glob.glob('/ldata/trunk/data_2019-04*/*.fil'):
        fil_obj = pysigproc.SigprocFile(filterbank)
        mjd = fil_obj.tstart
        influx_df = mjd2influx(mjd)
        try:
            tel_df_to_es(es, influx_df, filterbank)
        except TypeError:
            logging.error(filterbank)

