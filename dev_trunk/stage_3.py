#!/usr/bin/env python3.6

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import logging
import glob
import pika
import pandas as pd
from pika_send import send2Q
from plot_cand import plotem
import dropbox
import yaml
import re
from os import path
from slack_send import send_img_2_slack
from database import processh5File, schema

logger = logging.getLogger()
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.getLogger('pika').setLevel(logging.INFO)

__author__='Devansh Agarwal'
__email__ = 'da0017@mix.wvu.edu'

DB_FILE = path.join("/","ldata","known_srcs.db")

def stage_initer(values):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    
    channel.queue_declare(queue='stage03_queue', durable=True)
    
    def callback(ch, method, properties, body):
        values.file=body.decode()
        logging.info(f'got it {values.file}')
        begin_main(values)
        ch.basic_ack(delivery_tag = method.delivery_tag)
        logging.info("Ack'ed")
    
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(on_message_callback=callback, queue='stage03_queue')
    
    channel.start_consuming()

def process_file(file_name, skip_pointing, dbx):
    try:
        if not skip_pointing:
            if processh5File(file_name, None, DB_FILE): #source with matching DM found
                logging.info(f"Existing source found for file {file_name}")
        fout, known_src = plotem(file_name, nrby=True, skip_pointing=skip_pointing) #modified: get nearby source info
        file_name=fout.split('/')[-1][:-3]
        if not skip_pointing:
            with open(fout, 'rb') as f:
                data = f.read()
                response=dbx.files_upload(data,f'/plots/{file_name}png',mode=dropbox.files.WriteMode.overwrite)
                logging.info(response)
                link=dbx.sharing_create_shared_link(f'/plots/{file_name}png')
                url=link.url
                slack_send_url=re.sub(r"\&dl\=0", "&dl=1", url) # Replaced ? with & b/c change in dropbox url
            logging.info(f'Create png at {fout}')
            logging.info(f'Dropbox URL {slack_send_url}')
            logging.info(send_img_2_slack(slack_send_url, nrby=known_src))
        else: #if test mode
            logging.info(f'Created png at {fout}')
    except Exception as error:
        logging.warning(f"Encountered {error} while processing file {file_name}")

def begin_main(values):
    with open("config/conf.yaml", 'r') as stream:
        data_loaded = yaml.load(stream)
    TOKEN = data_loaded['dropbox']['token']
    dbx = dropbox.Dropbox(TOKEN)
    if values.file is not None:
        process_file(values.file, values.test, dbx)
    else:
        with open(values.filelist, "r") as file:
            for fname in file.readlines():
                process_file(fname[:-1], values.test, dbx) #strip the newline
    return None


if __name__ == '__main__':
    parser=ArgumentParser(description='Stage 3: Get Real Cands', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_true', help='Be verbose')
    parser.add_argument('-d', '--daemon', dest='daemon', action='store_false', help='Run with AMQP')
    parser.add_argument('-f', '--file', nargs='+')
    parser.add_argument('-F', '--filelist', type=str, help="file containing list of stage 2 outputs")
    parser.add_argument('-t', '--test', action="store_true", help="Skip pointing data and database, and just plot cands")
    parser.set_defaults(file=None)
    parser.set_defaults(filelist=None)
    parser.set_defaults(verbose=False)
    parser.set_defaults(daemon=True)
    parser.set_defaults(test=False)
    values = parser.parse_args()

    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=format)
    else:
        logging.basicConfig(level=logging.INFO, format=format)

    if values.daemon:
        logging.info('Running in daemon mode')
        schema(DB_FILE)
        stage_initer(values)
    else:
        if (values.file is None) and values.filelist is None:
            logging.critical("No file or list of files given. Closing.")
        else:
            begin_main(values)
