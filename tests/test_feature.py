from collections import namedtuple

from pytest import approx
from sykepic.compute import feature

Args = namedtuple("Args", "raw samples out matlab parallel force")


def test_call(tmp_path, matlab):
    """Pass `--matlab /path/to/matlab` to test matlab version"""
    out_dir = tmp_path / "out"
    arguments = Args(
        raw="tests/data/raw/valid/",
        samples=None,
        out=out_dir,
        matlab=matlab,
        parallel=False,
        force=False,
    )
    feature.call(arguments)
    out_csvs = list(out_dir.glob("**/*.csv"))
    assert len(out_csvs) == 1
    with open(out_csvs[0]) as fh:
        lines = fh.readlines()
        assert len(lines) == 5
        volume = float(lines[1].rpartition("=")[-1])
        assert volume == approx(0.985, rel=1e-3)
        header = lines[2].split(",")
        assert len(header) == 7
        assert header[0] == "roi"
        roi_2 = list(filter(None, lines[3].split(",")))
        roi_3 = list(filter(None, lines[4].split(",")))
        assert len(roi_2) == len(header)
        assert len(roi_3) == len(header)
        assert int(roi_2[0]) == 2
        assert int(roi_3[0]) == 3
