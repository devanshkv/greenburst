#!/usr/bin/env python3.6

__author__='Devansh Agarwal'
__email__='da0017@mix.wvu.edu'

import collections
import threading
import socket
import numpy as np
import numba
import struct
#import filutils as fu
import time
import pysigproc
from astropy.time import Time
from datetime import datetime as dt
from itertools import cycle
import logging
import pika

logger = logging.getLogger()
logger=logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(threadName)s - %(levelname)s - %(message)s')
logging.getLogger('pika').setLevel(logging.INFO)

# Buffer size
N = 16
# Buffer init
buf = [0] * N

chunks=list(range(16))

pop_counter=cycle(chunks)

# spectra buffers
'''
We first buffer each spectra in producer.
It is decoded by consumer and fed to the spectra dumping buffer
which holds 2^17 samples (~33 sec, delay for 1e4 DM).


We make a FIFO queue with each element having 2^17 spectra in it.
Each filterbank will have 33 sec overlap and total time ~9 min.
This is from 2^21 samples in the file
'''

fill_count = threading.Semaphore(0)
empty_count = threading.Semaphore(N)


def init_stage1(Qinit):
    while True:
        try:
            filename=Qinit.pop()
            connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
            channel = connection.channel()
            channel.queue_declare(queue='stage01_queue', durable=True)
            channel.basic_publish(exchange='',
                                  routing_key='stage01_queue',
                                  body=filename,
                                  properties=pika.BasicProperties(
                                     delivery_mode = 2, # make message persistent
                                  ))
            logging.info(f'Sent {filename}')
            connection.close()
        except IndexError:
            logging.info(f'sleeping for 33 s')
            time.sleep(33)

@numba.jit(cache=True)
def get_data():
    '''
    Get 16 udp packets in an bytearrymemoryview obj
    '''
    data=bytearray(768*16)
    for ii in range(16):
        sock.recv_into(memoryview(data)[ii*768:(ii+1)*768])
    return data

def producer(wait):
    '''
    Wait till you are at the start of the spectra, then take data. Forever.
    '''
    front = 0
    while wait:
        data= sock.recvfrom(768)[0]
        header_tmp=struct.unpack('<768B',data)[0::3]
        header = header_tmp[0]+(header_tmp[1]<<8)+(header_tmp[2]<<16)+(header_tmp[3]<<24)+(header_tmp[4]<<32)+(header_tmp[5]<<40)+(header_tmp[6]<<48)+(header_tmp[7]<<56)
        CHANNEL = (header) & 0x3fff
        logging.info(f'waiting at {CHANNEL} Channel')
        if CHANNEL == 15360:
            wait=False
    while True:
        empty_count.acquire()
        buf[front] = get_data()
        fill_count.release()
        front = (front + 1) % N

@numba.jit(numba.types.Tuple((numba.uint64[:],numba.uint16[:]))(numba.uint8[:]),cache=True)
def get_head(all_head):
    '''
    Get sequence and channel numbers from the data
    '''
    pattern=np.zeros(4096,dtype=np.uint16)
    seq=np.zeros(4096,dtype=np.uint64)
    for i in range(16):
        header=all_head[i*256:i*256 + 8]
        chan=0
        for ii in range(8):
            chan += np.left_shift(header[ii],ii*8)
        cstart=np.right_shift(((chan)&0x3fff),2)
        seq[i*256:(i+1)*256]=np.right_shift(chan,10)
        pattern[i*256:(i+1)*256]=list(range(cstart,cstart+256))
        #print(numba.typeof(seq),numba.typeof(pattern),numba.typeof(all_head))
    return seq,pattern

def consume(data):
    '''
    Consume the memoryview object! Give meaning full data, add (xx/2 + yy/2)
    '''
    data=np.frombuffer(data,dtype=np.uint8)
    seq,cstart=get_head(data[0::3])
    spectra=memoryview(np.right_shift(data[1::3],1) + np.right_shift(data[2::3],1))
    #print(len(spectra))
    return seq,cstart,spectra

def consumer(wait,Q,Q2):
    '''
    Take data, get sequence number, then start filling 2^17 time smaples.
    Also, take care of the overflow, record time MJD, bake a cake and what not.
    Too much stuff in one function. Also put stuff in FIFO and log it, and tell when
    the program drops udp packets.
    '''
    rear = 0
    overflow_spec=None
    jj=-1
    fill_count.acquire()
    y = buf[rear]
    empty_count.release()
    rear = (rear + 1) % N
    FIRST_SEQ,cstart,spec=consume(y)
    while True:
        t = dt.utcnow()
        last_increment=0
        spectra=np.zeros((2**17,2**12),dtype=np.uint8)
        if overflow_spec is None:
            jj=-1
        else:
            jj=np.max(overflow_count)
            spectra[overflow_count,overflow_cstart]=overflow_spec
        while jj < spectra.shape[0]-1:
            #start_time=time.time()
            fill_count.acquire()
            y = buf[rear]
            empty_count.release()
            rear = (rear + 1) % N
            seq,cstart,spec=consume(y)
            #start_time=time.time()
            fill_index=(seq-FIRST_SEQ)//16
            increment=fill_index.max()
            jj=increment
            if increment-last_increment > 1:
                logging.info(f'Lost {increment-last_increment} spectra(s)')
            if jj >= spectra.shape[0]:
                overflow_count=((seq-FIRST_SEQ)//16) - spectra.shape[0]
                overflow_mask=overflow_count<0
                overflow_count[overflow_mask]=0
                overflow_spec=np.zeros(4096,dtype=np.uint8)
                overflow_spec=spec[~overflow_mask]
                overflow_cstart=cstart[~overflow_mask]
            else:
                overflow_spec=None
                spectra[fill_index,cstart]=spec
            last_increment=increment
            #print('took:', (time.time()-start_time)/256e-6)
        FIRST_SEQ+=(2**21)
        logging.info(f'Dumping to fifo')
        Q.appendleft(spectra)
        Q2.appendleft(t)

def filwriter(Q,Q2,Qinit):
    '''
    Innocent filterbank writer.
    '''
    while True:
        global f
        global filename
        try:
            spectra_dump=Q.pop()
            spectra_time=Q2.pop()
            counter_val=next(pop_counter)
            if counter_val == 0:
                mjd=Time(spectra_time).mjd
                # create the filterbank
                filobj=pysigproc.SigprocFile()
                filename='/sdata/filterbank/'+str(spectra_time.strftime("%Y_%m_%d_%H_%M_%S"))+'.fil'
                logging.info(f'writing in {filename}')
                filobj=fu.make_header(filobj)
                filobj.tstart=mjd
                fu.write_header(filobj,filename)
                f=open(filename,'r+b')
                fu.append_spectra(f,spectra_dump)
            elif counter_val == 15:
                Q.appendleft(spectra_dump)
                Q2.appendleft(spectra_time)
                fu.append_spectra(f,spectra_dump)
                f.close()
                Qinit.appendleft(filename)
            else:
                fu.append_spectra(f,spectra_dump)
        except IndexError:
            time.sleep(11)


if __name__ == "__main__":
    
    # Bind to the socket to get the udp packets
    HOST, PORT = "10.0.1.38", 10000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((HOST, PORT))
    '''
    The seemingly omnipotent and immortal species who inhabited the Q Continuum
    or not, a simple thread safe queue to keep spectrum buffers.
    In case you didn't get the above comment you really need to watch Star Trek TNG.
    '''
    Q = collections.deque(maxlen=3)

    Q2 = collections.deque(maxlen=3)

    '''
    Next we need another deque for init'ing the first stage
    '''

    Qinit = collections.deque(maxlen=3)

    wait=True
    producer_thread = threading.Thread(name='udp2buf',target=producer,args=(wait,))
    consumer_thread = threading.Thread(name='buf2spec',target=consumer,args=(wait,Q,Q2,))
    #filwrite_therad = threading.Thread(name='spec2fil',target=filwriter,args=(Q,Q2,Qinit,))
    #heiminit_thread = threading.Thread(name='heiminit',target=init_stage1,args=(Qinit,))
    
    producer_thread.start()
    consumer_thread.start()
    time.sleep(24)
    filwrite_therad.start()
    time.sleep(8*60)
    heiminit_thread.start()
