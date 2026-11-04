#!/usr/bin/env python3.6
import matplotlib

matplotlib.use("Agg")
import glob
import logging
import sys
import threading
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from multiprocessing import Process
from subprocess import PIPE, Popen

import numpy as np
import pika
import pysigproc
from elasticsearch import Elasticsearch
from pika.exceptions import *  # StreamLostError, ConnectionResetError
from scipy.signal import savgol_filter

from dump_all import tel_df_to_es
from gpu_client import send2gpuQ
from influx_2df import mjd2influx
from pika_send import send2Q
from slack_send import *

logger = logging.getLogger()
format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=format)
logging.getLogger("pika").setLevel(logging.INFO)

__author__ = "Devansh Agarwal"
__email__ = "da0017@mix.wvu.edu"


def stage_initer(values):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host="localhost"))
    channel = connection.channel()

    channel.queue_declare(queue="stage01_queue", durable=True)

    def callback(ch, method, properties, body):
        values.file = body.decode()
        logging.info(f"got it {values.file}")
        ch.basic_ack(delivery_tag=method.delivery_tag)
        begin_main(values)
        logging.info("Done")
        # ch.basic_ack(delivery_tag = method.delivery_tag)
        logging.info("Ack'ed")

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(on_message_callback=callback, queue="stage01_queue")
    try:
        channel.start_consuming()
    except (StreamLostError, ConnectionResetError) as e:
        connection = pika.BlockingConnection(
            pika.ConnectionParameters(host="localhost")
        )
        channel = connection.channel()


def mask_finder(data, sigma):
    y = savgol_filter(data, 61, 2)
    mask = (data - y > sigma) | (data - y < -sigma)
    mask[:10] = True
    mask[980:1445] = True
    mask[3518:3560] = True
    return mask


def _cmdline(command):
    """
    Function that captures output from screen
    """
    process = Popen(args=command, stdout=PIPE, shell=True)
    output = process.communicate()[0]
    logging.info(f"Processed {command}")
    return output


def write_and_plot(chan_nos, freqs, bandpass, outdir, mask=None):
    """
    Writes and plots the bandpass
    """
    bad_chans = outdir + "bad_chans.flag"
    bp_plot = outdir + "bandpass.png"
    if mask is not None:
        logging.info("Flagged %d channels", mask.sum())
        with open(bad_chans, "w") as f:
            np.savetxt(f, chan_nos[mask], fmt="%d", delimiter=" ", newline=" ")
    else:
        with open(bad_chans, "w") as f:
            pass

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax11 = fig.add_subplot(111)
    ax11.plot(chan_nos, bandpass, "k-", label="Bandpass")
    ax11.set_xlabel("Chan. no.")
    ax11.set_ylabel("Arb. Units")
    ax21 = ax11.twiny()
    if mask is not None:
        ax11.plot(chan_nos[mask], bandpass[mask], "ro", label="Flaged Channels")
        ax21.plot(freqs[mask], bandpass[mask], "r.")
    ax21.invert_xaxis()
    ax21.set_xlabel("Frequency (MHz)")
    ax11.legend()
    plt.savefig(bp_plot, bbox_inches="tight")


def begin_main(values):
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=format)
    else:
        logging.basicConfig(level=logging.INFO, format=format)

    try:
        filterbank = glob.glob(values.file)[0]

        fil_obj = pysigproc.SigprocFile(filterbank)
        freqs = fil_obj.chan_freqs
        if values.influx:
            #This entire if-else block checks metadata presence from InfluxDB.
            #It then indexes the metadata into elasticsearch.
            #At time of writing, InfluxDB is down so we're going to
            #ignore it and hope for the best.
            df = mjd2influx(fil_obj.tstart)
            if df is not None:
                if len(df) < 33:
                    #send_msg_2_slack(f"Pointing info is missing!")
                    logging.info("Less than 33s of data is valid, skipping this file")
                    pass
                all_data_valid = df["DATA_VALID"].sum()
                if all_data_valid < 33:
                    logging.info("Less than 33s of data is valid, skipping this file")
                    _cmdline(f"rm {filterbank}")
                    return None
                else:
                    es=Elasticsearch([{'host':'localhost','port':9200}])
                    tel_df_to_es(es,df,filterbank)
            else:
                logging.warning("No response from InfluxDB.")
                send_msg_2_slack(f"No info from InfluxDB")
                _cmdline(f"rm {filterbank}")
                return None
            logging.info(f"{100*all_data_valid/len(df)}% data valid")
        bandpass = fil_obj.bandpass
        chan_nos = np.arange(0, bandpass.shape[0])
        # mask = mask_finder(
        #     bandpass, values.sigma
        # )  # chan_nos,values.nchans,values.sigma)
        # bad_chans = chan_nos[mask]

        # frac_flagged = mask.sum() / 4096
        frac_flagged = 0  # Get the real value from the output of jess
        if values.influx:
            es = Elasticsearch([{"host": "localhost", "port": 9200}])
            tel_df_to_es(es, df, filterbank, frac_flagged)

        # out_chans = []
        # for chans in bad_chans:
        #     out_chans.append("-zap_chans")
        #     out_chans.append(chans)
        #     out_chans.append(chans)

        #This block moves the filterbank file to a standardized location.
        #To get this working on other machines, this is probably
        #the main thing to worry about.
        filterbank_name = filterbank.split("/")[-1].split(".")[0]
        out_dir = "/ldata/trunk/{}/".format(filterbank_name)
        _cmdline("mkdir -p {}".format(out_dir))
        _cmdline(f"mv {filterbank} {out_dir}/")
        new_fil_path = f"{out_dir}{filterbank_name}.fil"
        clean_fil_path = f"{out_dir}{filterbank_name}_jb_4096.fil"
        
        jess_command = (
            "time /sdata/miniconda/envs/py38/bin/python /opt/soft/jess/bin/jess_gauss.py"
            + " -test jarque_bera -spb 4096 -mtz 1.5"
            + f" -f {new_fil_path} -o {clean_fil_path}"
        )
        logging.info(f"Running {jess_command}")
        send2gpuQ(jess_command)
        #logging.warn("Skipping jess filtering!")
        #_cmdline(f"cp {new_fil_path} {clean_fil_path}")

        heimdall_command = (
            f"heimdall -nsamps_gulp 524288 -dm 10 10000 -boxcar_max 4096 -cand_sep_dm_trial 200 -cand_sep_time 128 -cand_sep_filter 3"
            + " -rfi_no_broad"
            + " -rfi_no_narrow"
            + " -output_dir {}".format(out_dir)
            + " -f {}".format(clean_fil_path)
        )
        logging.info(f"Running {heimdall_command}")

        p1 = Process(target=send2gpuQ, args=[heimdall_command])
        p1.start()
        p2 = Process(target=write_and_plot, args=[chan_nos, freqs, bandpass, out_dir])
        p2.start()

        p1.join()
        p2.join()
        if values.ret:
            logging.info(f"Stage 1 output to /ldata/trunk/{filterbank_name}")
            return f"/ldata/trunk/{filterbank_name}"
        send2Q("stage02_queue", f"/ldata/trunk/{filterbank_name}")
    except IndexError:
        logging.debug("Index error occurred.")
        pass
    return None


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Stage 1: Get RFI Flags, run heimdall",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="Be verbose"
    )
    parser.add_argument(
        "-d", "--daemon", dest="daemon", action="store_false", help="Run with AMQP"
    )
    parser.add_argument(
        "-m", "--skipmetadata", dest="influx", action="store_false", help="Skip metadata indexing."
    )
    parser.add_argument(
        "-n", "--nchans", type=int, help="no. of chans to calc. median over", default=64
    )
    parser.add_argument(
        "-s",
        "--sigma",
        type=int,
        help="sigma over which values are tagged as RFI",
        default=5,
    )
    parser.add_argument("-r", "--return", dest="ret", action="store_true", help="Return filename instead of adding to queue")
    parser.add_argument("-f", "--file", type=str, help="Filterbank file")
    parser.set_defaults(verbose=False)
    parser.set_defaults(daemon=True)
    parser.set_defaults(influx=True)
    parser.set_defaults(ret=False)
    values = parser.parse_args()

    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    if values.verbose:
        logging.basicConfig(level=logging.DEBUG, format=format)
    else:
        logging.basicConfig(level=logging.INFO, format=format)

    if values.daemon:
        logging.info("Running in daemon mode")
        stage_initer(values)
    else:
        logging.info("Running in normal mode")
        begin_main(values)
