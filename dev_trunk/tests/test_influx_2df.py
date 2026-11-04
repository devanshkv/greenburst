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
GET_BOTH_DMS = influx_2df.get_both_dms
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
    """extend_df as it was before this fix, from git (4c312b8, the code deployed until the fix)."""
    src = subprocess.run(["git", "show", "4c312b8:dev_trunk/influx_2df.py"], cwd=DEV_TRUNK,
                         stdout=subprocess.PIPE, check=True).stdout.decode()
    mod = types.ModuleType("influx_2df_unpatched")
    exec(compile(src, "influx_2df_unpatched", "exec"), mod.__dict__)
    mod.get_both_dms = lambda gl, gb: (30.0, 30.0)
    return mod.extend_df


def test_all_valid_unchanged():
    old = unpatched_extend_df()(influx(), cands(10.3, 200.7, 503.9))
    new = influx_2df.extend_df(influx(), cands(10.3, 200.7, 503.9))
    # The same, apart from rounding in the RA and GL splines, which now fit unwrapped angles
    assert list(old.columns) == list(new.columns)
    for column in old.columns:
        if column in ("SCPROJID", "IFV1TNCI"):
            assert list(old[column]) == list(new[column])
        else:
            assert new[column].astype(float).values == pytest.approx(old[column].astype(float).values, abs=1e-9)
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


# ---------------------------------------------------------------- interpolate_value

@pytest.mark.parametrize("key", ["RA_deg", "GL_deg"])
@pytest.mark.parametrize("rate", [0.001, 0.4, -0.4])
def test_interpolation_across_zero(key, rate):
    # The pointing crosses 0/360 between two rows, for a drift and a slew either way (data_2026-11-02_22-31-35
    # crossed RA 0 at 0.4 deg/s); the old spline of RA - 180 was up to 179 deg off near the crossing
    t = np.arange(540.0)
    start = 360 - rate * 250.5 if rate > 0 else -rate * 250.5
    df = pd.DataFrame({"MJD": MJD + t / DAY, key: (start + rate * t) % 360})
    for tc in (240.3, 249.5, 250.2, 250.5, 250.8, 251.5, 260.7):
        got = float(influx_2df.interpolate_value(MJD + tc / DAY, df, key))
        assert 0 <= got < 360
        assert (got - (start + rate * tc) + 180) % 360 - 180 == pytest.approx(0, abs=1e-6)


def test_interpolation_ra_unchanged_away_from_zero():
    t = np.arange(540.0)
    df = pd.DataFrame({"MJD": MJD + t / DAY, "RA_deg": 150 + 0.004 * t + 0.01 * np.sin(t / 30)})
    for tc in (0.5, 100.25, 503.9):
        old = float(UnivariateSpline_of(df, "RA_deg", -180)(MJD + tc / DAY)) + 180
        assert float(influx_2df.interpolate_value(MJD + tc / DAY, df, "RA_deg")) == pytest.approx(old, abs=1e-9)


def UnivariateSpline_of(df, key, offset):
    return influx_2df.UnivariateSpline(df["MJD"], y=df[key] + offset)


def test_extend_df_across_ra_zero():
    t = influx()
    t["RA_deg"] = (359.9 + 0.001 * np.arange(540)) % 360
    df = influx_2df.extend_df(t, cands(99.6, 100.4))
    assert [float(x) for x in df["RA_deg"]] == pytest.approx([359.9996, 0.0004], abs=1e-6)


# ---------------------------------------------------------------- get_both_dms

class Run:
    """subprocess.run stand-in giving each program's output, from the formats in NE2001.f and ymw16's dmdtau.c."""

    def __init__(self, outputs):
        self.outputs, self.commands = outputs, []

    def __call__(self, cmd, stdout=None):
        self.commands.append(cmd)
        return types.SimpleNamespace(stdout=self.outputs[cmd[0]])


NE2001_OUT = b"   27.7551250    \n"   # write(*,*) dm, for ndir -1
YMW16_OUT = b"Gal: gl= 103.473 gb=  69.195 Dist=  25000.0 DM:   21.17 log(tau_sc): -9.876 \n"


def test_get_both_dms(monkeypatch):
    run = Run({"NE2001": NE2001_OUT, "ymw16": YMW16_OUT})
    monkeypatch.setattr(influx_2df.subprocess, "run", run)
    assert GET_BOTH_DMS(103.47, 69.19) == (pytest.approx(27.755125), pytest.approx(21.17))
    assert run.commands == [["NE2001", "103.47", "69.19", "25", "-1"], ["ymw16", "Gal", "103.47", "69.19", "25000", "2"]]


def test_get_both_dms_ymw16_warning_line(monkeypatch):
    # ymw16 prints a warning first when YMW16_DIR isn't set
    out = b"Warning: YMW16_DIR set to local directory\n" + YMW16_OUT
    monkeypatch.setattr(influx_2df.subprocess, "run", Run({"NE2001": NE2001_OUT, "ymw16": out}))
    assert GET_BOTH_DMS(103.47, 69.19)[1] == pytest.approx(21.17)


@pytest.mark.parametrize("ne2001, ymw16", [(b"", YMW16_OUT), (NE2001_OUT, b""),
                                           (NE2001_OUT, b"please input correct model\n"),
                                           (b" Usage: NE2001 l b DM/D ndir\n", YMW16_OUT)])
def test_get_both_dms_bad_output(monkeypatch, ne2001, ymw16):
    monkeypatch.setattr(influx_2df.subprocess, "run", Run({"NE2001": ne2001, "ymw16": ymw16}))
    with pytest.raises(ValueError, match="No DM in the output"):
        GET_BOTH_DMS(103.47, 69.19)
