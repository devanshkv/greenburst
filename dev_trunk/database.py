#!/usr/bin/env python3.6
import sqlite3 as sql
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from astropy.time import Time
from os import path, curdir
import logging
from psrqpy import QueryATNF
from plot_cand import qpsr, h5_loction_2_stuff

def psrRowSort(psr):
    """
    Helper function which sorts astropy table row 'psr' according to the database schema order and returns an ordered list.
    """

    return [psr['NAME'], psr['RAJ'], psr['DECJ'], psr['P0'], psr['DM'],psr['W50'],psr['W10'],psr['S1400'],psr['ASSOC']]

def tsRowSort(ts):
    """
    Helper function which sorts telescope dictionary 'ts' according to the database schema order and returns an ordered list.
    """

    return [ts['MJD'], Time(ts['MJD'], format='mjd').iso, ts['Turret Angle (degree)'], ts['Receiver']]


def connect(file):
    """
    Connects to a database at given path.

    Returns a sql connect object, or None if some issue prevented access to file.
    """
    conn = None
    try:
        conn = sql.connect(file)
        conn.execute('pragma journal_mode=wal') #write-ahead log, in case of simultaneous access
    except:
        logging.critical(f"Unable to access file {file}")       
    return conn

def schema(file):
    """
    Verifies correct schema in database at given path.

    No return value.
    """

    conn = connect(file)
    schemaString = """ CREATE TABLE IF NOT EXISTS detections (
    row integer PRIMARY KEY,
    Name text NOT NULL,
    RAJ text NOT NULL,
    DecJ text NOT NULL,
    P0 real,
    DM real,
    W50 real,
    W10 real,
    S1400 real,
    Assoc text,
    MJD real,
    UTC text,
    Turret real,
    Receiver text
    ); """

    if conn is not None:
        c = conn.cursor()
        c.execute(schemaString)
        conn.commit()
        conn.close()
        
def filteredSearch(ra, dec, epsilon=0.1, width=None, DM=None):
    """
    Searches ATNF database for pulsars within 1 degree of coordinates with pulse width and/or DM conditions.

    Positional arguments:
    ra (string) -- right ascention formatted as XXhXXmXXs
    dec (string) -- declination formatted as XXdXXmXXs

    Keyword arguments:
    epsilon (float) -- factor of value to use as conditional window (default 0.05)
    width (float) -- width of pulse in milliseconds (default None)
    DM (float) -- Dispersion measure in pc/cm3 (default None)

    Returns a single astropy table row representing a found source if at least one is found, and False otherwise.
    """ 

    #start by constructing search filter
    if width is not None:
        filterString = f"(W50 > {width*(1-epsilon)} && W50 < {width*(1+epsilon)})"
        if DM is not None: #if both, need &&
            filterString += f" && (DM > {DM*(1-epsilon)} && DM < {DM*(1+epsilon)})"
    elif DM is not None:
        filterString = f"(DM > {DM*(1-epsilon)} && DM < {DM*(1+epsilon)})"
    else:
        logging.warning("Filtered search run with no filters.")
        filterString = None

    query, qTable = qpsr(ra, dec, params = ['NAME','RAJ', 'DECJ', 'P0', 'DM', 'W50', 'W10', 'S1400', 'ASSOC'], condition=filterString)
    qTable = query.table #reassign due to a hardcoded slice in original function
    logging.info(f"Sources found: {len(qTable)}")
    if len(qTable) > 0:
        print(qTable)
        return qTable[0]
    return False
    
def addRow(psr, ts=None, db=path.join(curdir,'test.db')):
    """
    Given a astropy table row 'psr' and h5 parameter dictionary 'ts', adds a database entry.

    Positional arguments:
    psr (dict) -- Unsorted astropy table row containing pulsar information

    Keyword arguments:
    ts (dict) -- Dictionary of parameters from h5 file (default None)
    db (string) -- database file to add to (default 'test.db')

    Returns True if row sucessfully added, False otherwise.
    """

    conn = connect(db)
    if conn is None:
        logging.critical('Connect object failed')
        return False
    c = conn.cursor()
    storeString = """ INSERT INTO detections(Name, RAJ, DecJ, P0, DM, W50, W10, S1400, Assoc, MJD, UTC, Turret, Receiver)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """

    if ts is not None:
        info = psrRowSort(psr) + tsRowSort(ts)
    else:
        info = psrRowSort(psr) + ([None]*4) #make sure they're the same length
    if db == path.join(curdir,'test.db'):
        logging.warning('Using test database file!')
    c.execute(storeString, info)
    conn.commit()
    conn.close()
    return True
    
def processh5File(h5File, csvPath, db=path.join(curdir,'test.db')):
    """
    Given an h5 file, searches ATNF for a match and adds a row to database if one is found :3

    Positional arguments:
    h5File (string) -- Filename of an h5 file
    csvPath (string or None) -- Filename of associated csv file 

    Returns True if source is found and added to database, False if no source or database error.
    """

    stuff, params = h5_loction_2_stuff(h5File, csv_path=csvPath)
    ra = params['RA (J2000)']
    dec = params['DEC (J2000)']
    dm = params['DM (pc/cc)']
    result = filteredSearch(ra, dec, DM=dm) #search
    if not result: #nothing found
        logging.info('No known source in this region with this DM!')
        return False
    return addRow(result, ts=params, db=db)
    

if __name__ == '__main__':
    parser=ArgumentParser(description='Access database', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    #TODO: Add arguments for different queries and possibly additions
    #Example argument add: parser.add_argument('-f', '--files', nargs='+', help='Filterbank file')
    parser.add_argument('--db', dest='dbfile', help='Database file')
    parser.add_argument('-s', '--search', dest='searchString', help='Test search ra/dec/width')
    parser.add_argument('-f', '--file', dest='h5File', help='h5 file to process and add to database')
    parser.add_argument('-c', '--csv', dest='csvFile', help='csv file containing info for h5 file')
    parser.set_defaults(dbfile=path.join(curdir,'test.db'))
    parser.set_defaults(searchString = None)
    parser.set_defaults(h5File = None)
    parser.set_defaults(csvFile=None)
    parser.set_defaults(verbose=False)
    values = parser.parse_args()

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=format)
    else:
        logging.basicConfig(level=logging.INFO, format=format)

    logging.debug(f"Accessing database at {values.dbfile}")
    schema(values.dbfile)
    if values.h5File is not None:
        processh5File(values.h5File, values.csvFile)
    elif values.searchString is not None:
        ra, dec, w, dm = values.searchString.split(" ")
        filteredSearch(values.dbfile, float(ra), float(dec), epsilon=0.001, width=float(w), DM=float(dm))