#!/usr/bin/env python3

#from slackclient import SlackClient
from slack import WebClient as SlackClient

import yaml
import logging

def send_msg_2_slack(msg):
    with open("config/conf.yaml", 'r') as stream:
        data_loaded = yaml.load(stream)
    TOKEN = data_loaded['slack']['bot_oauth']
    
    client = SlackClient(TOKEN)
    response = client.chat_postMessage(
      channel="CPAK5A4G2",
      text=msg
    )
    return response

def send_img_2_slack(img, nrby=True):
    """
    Posts an image from url string to hardcoded slack channels.

    Positional arguments:
    img (string) -- publically accessible url of image to send

    Keyword arguments:
    nrby (bool)  -- whether a known source is near the candidate (default True)

    Returns string containing slack responses.
    """

    gb_images = 'C013W4P08MB'
    gb_alerts = 'CPAK5A4G2'
    gb_unknown = 'C08JCLTMNJD'
    gb_known = 'C0A9QNBMWK0'
    with open("config/conf.yaml", 'r') as stream:
        data_loaded = yaml.load(stream)
    TOKEN = data_loaded['slack']['bot_oauth']
    
    client = SlackClient(TOKEN)
    attachments = [{"title": "", "image_url": img}]
    response = client.chat_postMessage(channel=gb_alerts, text='',
                attachments=attachments)
    response2 = client.chat_postMessage(channel=gb_images, text='',
                attachments=attachments)
    if nrby: #known sources
        response3 = client.chat_postMessage(channel=gb_known, text='',
                attachments=attachments)
        return f"{response}\n{response2}\n{response3}"
    else: #no nearby sources
        response3 = client.chat_postMessage(channel=gb_unknown, text='',
                attachments=attachments)
        return f"{response}\n{response2}\n{response3}"



if __name__ == "__main__":
    logger = logging.getLogger()
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=format)
    #respose = send_msg_2_slack("Hello from Python! :tada:")
    #logging.info(f'{respose}')
    response = send_img_2_slack('https://www.dropbox.com/scl/fi/rx5c6kn2jbaxdz2mopfqe/cand_tstart_60586.508384786939_tcand_426.2090000_dm_5454.92000_snr_10.30260.png?rlkey=3bvcyn05xwk5p52o4rhobi8uk&dl=1')
    logging.info(f"{response}")
