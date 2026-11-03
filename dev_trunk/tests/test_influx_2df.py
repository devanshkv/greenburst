"""
Tests for influx_2df.extend_df's pointing check: a candidate is valid when the Influx rows within 3 s of it
are valid, not only when every row of the file's 9-minute window is.

Run from dev_trunk:  python -m pytest tests
"""

import os
import subprocess
import sys
import types

import numpy as np
import pandas as pd
import pytest

DEV_TRUNK = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, DEV_TRUNK)

# extend_df needs none of the Influx client, the config or astropy
sys.modules.setdefault("influxdb", types.SimpleNamespace(DataFrameClient=None))
sys.modules.setdefault("yaml", types.ModuleType("yaml"))
try:
    import astropy.time  # noqa: F401
except ImportError:
    sys.modules["astropy"] = types.ModuleType("astropy")
    sys.modules["astropy.time"] = types.SimpleNamespace(Time=None)

import influx_2df  # noqa: E402

MJD = 61000.5
DAY = 60 * 60 * 24


def influx(seconds=range(540), invalid=(), project=lambda t: "AGBT26A_001"):
    """One Influx row a second from MJD, tracking a source that drifts 0.001 deg/s in RA."""
    t = np.array(list(seconds), dtype=float)
    return pd.DataFrame({
        "MJD": MJD + t / DAY,
        "DATA_VALID": [0 if s in invalid else 1 for s in t],
        "RA_deg": 150 + 0.001 * t, "DEC_deg": 10 + 0 * t,
        "GL_deg": 230 + 0.001 * t, "GB_deg": 50 + 0 * t,
        "ATAZIND": 100 + 0 * t, "ATELIND": 40 + 0 * t, "AZCORR": 100 + 0 * t, "ELCORR": 40 + 0 * t,
        "RA_drift": 150 + 0 * t, "DEC_drift": 10 + 0 * t,
        "SCPROJID": [project(s) for s in t], "IFV1TNCI": "1400", "ATRXOCTA": 0.0,
    })


def cands(*tcand):
    df = pd.DataFrame({"tcand": tcand, "dm": 100.0})
    df["cand_mjd"] = MJD + df["tcand"] / DAY
    return df


@pytest.fixture(autouse=True)
def no_dm_models(monkeypatch):
    # NE2001 and ymw16 are external programs
    monkeypatch.setattr(influx_2df, "get_both_dms", lambda gl, gb: (30.0, 30.0))


def unpatched_extend_df():
    """extend_df as it was before this fix, from git."""
    src = subprocess.run(["git", "show", "origin/greenburst:dev_trunk/influx_2df.py"], cwd=DEV_TRUNK,
                         stdout=subprocess.PIPE, check=True).stdout.decode()
    mod = types.ModuleType("influx_2df_unpatched")
    exec(compile(src, "influx_2df_unpatched", "exec"), mod.__dict__)
    mod.get_both_dms = lambda gl, gb: (30.0, 30.0)
    return mod.extend_df


def test_all_valid_unchanged():
    old = unpatched_extend_df()(influx(), cands(10.3, 200.7, 503.9))
    new = influx_2df.extend_df(influx(), cands(10.3, 200.7, 503.9))
    pd.testing.assert_frame_equal(old, new)
    assert list(new["cand_valid"]) == [1, 1, 1]


def test_invalid_seconds_elsewhere_keep_candidates():
    # Invalid only after stage 2's 504 s limit, as in data_2026-11-01_08-38-16
    df = influx_2df.extend_df(influx(invalid=range(509, 541)), cands(10.3, 200.7, 503.9))
    assert list(df["cand_valid"]) == [1, 1, 1]
    assert df.loc[1, "RA_deg"] == pytest.approx(150 + 0.001 * 200.7, abs=1e-6)


def test_candidate_near_invalid_seconds_dropped():
    df = influx_2df.extend_df(influx(invalid=range(100, 200)), cands(50, 98.5, 150, 201.5, 300))
    assert list(df["cand_valid"]) == [1, 0, 0, 0, 1]
    assert df.loc[1, "RA_deg"] is None


def test_pointing_from_own_valid_run():
    # A slew (invalid) between two tracks: each candidate is interpolated over its own track only
    t = influx(invalid=range(200, 260))
    t.loc[t["MJD"] >= MJD + 260 / DAY, "RA_deg"] += 20
    df = influx_2df.extend_df(t, cands(150, 400))
    assert list(df["cand_valid"]) == [1, 1]
    assert df.loc[0, "RA_deg"] == pytest.approx(150 + 0.001 * 150, abs=1e-6)
    assert df.loc[1, "RA_deg"] == pytest.approx(170 + 0.001 * 400, abs=1e-6)


def test_candidate_in_influx_gap():
    # No rows within 3 s: valid when the rows either side are, interpolated across the gap
    rows = [s for s in range(540) if not 100 <= s < 160]
    df = influx_2df.extend_df(influx(rows), cands(130))
    assert df.loc[0, "cand_valid"] == 1
    assert df.loc[0, "RA_deg"] == pytest.approx(150 + 0.001 * 130, abs=1e-6)
    assert df.loc[0, "SCPROJID"] == "AGBT26A_001"


def test_candidate_in_gap_next_to_invalid_rows():
    rows = [s for s in range(540) if not 100 <= s < 160]
    df = influx_2df.extend_df(influx(rows, invalid=range(160, 170)), cands(130))
    assert df.loc[0, "cand_valid"] == 0


def test_project_change_gives_one_name():
    t = influx(project=lambda s: "AGBT26A_001" if s < 300 else "AGBT26A_002")
    df = influx_2df.extend_df(t, cands(100, 400))
    assert list(df["SCPROJID"]) == ["AGBT26A_001", "AGBT26A_002"]


def test_no_influx_data():
    df = influx_2df.extend_df(None, cands(100))
    assert df.loc[0, "cand_valid"] == 1
