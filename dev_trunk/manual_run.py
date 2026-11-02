#!/usr/bin/env python3.6

from stage_1 import begin_main as stage1
from stage_2 import begin_main as stage2
from stage_3 import begin_main as stage3
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser, Namespace

def parseS1(s1String):
    s1parser = ArgumentParser()
    s1parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="Be verbose"
    )
    s1parser.add_argument(
        "-d", "--daemon", dest="daemon", action="store_false", help="Run with AMQP"
    )
    s1parser.add_argument(
        "-n", "--nchans", type=int, help="no. of chans to calc. median over", default=64
    )
    s1parser.add_argument(
        "-s",
        "--sigma",
        type=int,
        help="sigma over which values are tagged as RFI",
        default=5,
    )
    s1parser.add_argument("-f", "--file", type=str, help="Filterbank file")
    s1parser.set_defaults(verbose=False)
    s1parser.set_defaults(daemon=False)
    return s1parser.parse_args()

if __name__ == "__main__":
    parser = ArgumentParser(
        description="Manual Processing: Full stack on filterbank.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="Be verbose"
    )
    parser.add_argument(
        "-f", "--file", type=str, help="Filterbank file"
    )
    values = parser.parse_args()
    #build fake arguments for stage 1 and ask for a return value
    #this value is normally passed to stage 2 via a queue system but 
    #let's keep it simple
    s1args = Namespace(file=values.file, verbose=values.verbose)
    s1Ret = stage1(s1Args, ret=True) 
    print(s1Ret)
    