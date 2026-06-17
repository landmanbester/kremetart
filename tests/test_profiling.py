"""Tests for the smoovie per-stage profiling helpers (:mod:`kremetart.utils.profiling`)."""

from kremetart.utils.profiling import print_profile, stage_timer


def test_print_profile_outputs(capsys):
    print_profile([("imaging", 2.0), ("render", 1.0)], nframes=4)
    out = capsys.readouterr().out
    assert "smoovie profile" in out
    assert "imaging" in out and "render" in out and "TOTAL" in out


def test_stage_timer_records():
    timings = []
    with stage_timer("stage_a", timings):
        pass
    assert len(timings) == 1 and timings[0][0] == "stage_a" and timings[0][1] >= 0.0
