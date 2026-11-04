"""
Tests for stage_2.begin_main: the candidate thresholds, the pointing check (the real influx_2df.extend_df on
fake Influx rows), the files it writes and what it sends on. RabbitMQ, the GPU queue, Slack and sigproc reading
are stubbed, and /ldata paths point into the test directory.

Run from dev_trunk:  python -m pytest tests
"""

import builtins
import os
import sys
import types
from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from test_influx_2df import DAY, MJD, influx  # also stubs influx_2df's imports

DEV_TRUNK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OBS = "data_2026-01-01_00-00-00"

# heimdall .cand columns: snr sample tcand width dm_trial dm members begin end
PASS = (9.0, 390625, 100.0, 3, 50, 100.0, 10, 390620, 390630)


def cand(**change):
    row = dict(zip(("snr", "sample", "tcand", "width", "dm_trial", "dm", "members", "begin", "end"), PASS))
    row.update(change)
    return tuple(row.values())


class Recorder:
    def __init__(self):
        self.gpu, self.queue, self.slack = [], [], []


@pytest.fixture
def stage2(monkeypatch, tmp_path):
    """stage_2 from this tree, with /ldata/trunk -> tmp_path/trunk and /ldata/dev_tests -> tmp_path/dev_tests."""
    rec = Recorder()
    state = {"influx": influx(), "fetch_labels": None}
    real_open = builtins.open

    def open_redirected(file, *args, **kwargs):
        if isinstance(file, str) and file.startswith("/ldata/dev_tests/"):
            file = str(tmp_path / "dev_tests" / os.path.basename(file))
            os.makedirs(os.path.dirname(file), exist_ok=True)
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_redirected)

    def module(name, **attrs):
        mod = types.ModuleType(name)
        mod.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, mod)
        return mod

    exceptions = module("pika.exceptions", StreamLostError=Exception, ConnectionResetError=Exception)
    module("pika", exceptions=exceptions, BlockingConnection=None, ConnectionParameters=None)
    module("pysigproc", SigprocFile=lambda filename=None, fp=None: types.SimpleNamespace(tstart=MJD))

    def send2gpuQ(command):
        rec.gpu.append(command)
        if command.startswith("predict.py"):
            # FETCH labels the candidates candmaker would have made from the csv
            cands_dir = command.split(" -c ")[1].split()[0]
            rows = pd.read_csv(os.path.join(os.path.dirname(cands_dir.rstrip("/")), OBS + ".csv"))
            labels = state["fetch_labels"] or [1] * len(rows)
            pd.DataFrame({"candidate": ["%s/cand_%d.h5" % (cands_dir.rstrip("/"), i) for i in range(len(rows))],
                          "label": labels}).to_csv(os.path.join(cands_dir, "results_a.csv"), index=False)

    module("gpu_client", send2gpuQ=send2gpuQ)
    module("pika_send", send2Q=lambda queue, body: rec.queue.append((queue, body)))
    module("slack_send", send_msg_2_slack=lambda msg: rec.slack.append(msg))

    import influx_2df
    monkeypatch.setattr(influx_2df, "mjd2influx", lambda mjd: state["influx"])
    monkeypatch.setattr(influx_2df, "get_both_dms", lambda gl, gb: (30.0, 30.0))

    trunk = tmp_path / "trunk"
    source = open(os.path.join(DEV_TRUNK, "stage_2.py")).read().replace("/ldata/trunk", str(trunk))
    stage = types.ModuleType("stage_2_under_test")
    exec(compile(source, "stage_2.py", "exec"), stage.__dict__)

    def run(rows, outfile=False):
        obs = trunk / OBS
        obs.mkdir(parents=True, exist_ok=True)
        with real_open(str(obs / "2026-01-01-00:00:34_01.cand"), "w") as f:
            for row in rows:
                f.write(" ".join(str(x) for x in row) + "\n")
        (obs / (OBS + "_jb_4096.fil")).write_text("fake")
        (obs / "bad_chans.flag").write_text("")
        stage.begin_main(Namespace(files=str(obs / "*cand"), snr=8.0, width=12, dm=20, members=5, outfile=outfile))
        csv = obs / (OBS + ".csv")
        return pd.read_csv(str(csv)) if csv.exists() else None, obs

    return run, rec, state


def test_thresholds(stage2):
    run, rec, _ = stage2
    rows = [cand(tcand=10.0), cand(snr=7.9), cand(members=4), cand(tcand=504.5), cand(dm=20.0), cand(width=13),
            cand(tcand=200.0, snr=8.0, members=5, dm=20.01, width=12)]
    csv, obs = run(rows)
    assert list(csv["tcand"]) == [10.0, 200.0]
    assert csv["cand_mjd"].values == pytest.approx(MJD + np.array([10.0, 200.0]) / DAY, abs=1e-9)
    assert list(csv["cand_valid"]) == [1, 1]
    assert csv["RA_deg"].values == pytest.approx([150.01, 150.2], abs=1e-6)
    assert rec.gpu[0] == "candmaker.py -n 11 -c {0}/{1}.csv -o {0}/cands/".format(obs, OBS)
    assert rec.gpu[1].startswith("predict.py ")
    assert [q for q, _ in rec.queue] == ["stage03_queue", "stage03_queue"]
    assert not (obs / (OBS + "_jb_4096.fil")).exists()   # stage 2 deletes the cleaned file


def test_partly_invalid_pointing_keeps_valid_candidates(stage2):
    run, rec, state = stage2
    state["influx"] = influx(invalid=range(300, 400))
    csv, _ = run([cand(tcand=100.0), cand(tcand=350.0), cand(tcand=450.0)])
    assert list(csv["tcand"]) == [100.0, 450.0]
    assert len(rec.queue) == 2


def test_invalid_after_the_file_keeps_candidates(stage2):
    # data_2026-11-01_08-38-16: invalid only from 509 s, past stage 2's 504 s limit
    run, _, state = stage2
    state["influx"] = influx(invalid=range(509, 541))
    csv, _ = run([cand(tcand=t) for t in (12.0, 250.0, 503.0)])
    assert list(csv["tcand"]) == [12.0, 250.0, 503.0]


def test_all_invalid_writes_nothing(stage2):
    run, rec, state = stage2
    state["influx"] = influx(invalid=range(540))
    csv, obs = run([cand(tcand=100.0)])
    assert csv is None
    assert rec.gpu == [] and rec.queue == []
    assert not (obs / (OBS + "_jb_4096.fil")).exists()


def test_nothing_above_threshold(stage2):
    run, rec, _ = stage2
    csv, obs = run([cand(snr=6.0), cand(dm=5.0)])
    assert csv is None
    assert rec.gpu == [] and rec.queue == []
    assert (obs / (OBS + ".png")).exists()


def test_fetch_negatives_not_sent(stage2):
    run, rec, state = stage2
    state["fetch_labels"] = [0, 1, 0]
    run([cand(tcand=t) for t in (10.0, 20.0, 30.0)])
    assert rec.queue == [("stage03_queue", rec.queue[0][1])]
    assert rec.queue[0][1].endswith("cand_1.h5")


def test_outfile(stage2):
    run, rec, _ = stage2
    _, obs = run([cand(tcand=10.0)], outfile=True)
    assert rec.queue == []
    assert (obs / "stage2_results.out").read_text().splitlines() == [str(obs / "cands" / "cand_0.h5")]
