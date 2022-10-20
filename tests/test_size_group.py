from collections import namedtuple

from pytest import approx
from sykepic.compute import size_group

Args = namedtuple(
    "Args",
    (
        "features groups size_column value_column out "
        "append force pixels_to_um3 volume quiet"
    ),
)


def test_main(tmp_path):
    out_file = tmp_path / "out.csv"
    arguments = Args(
        features="tests/data/feat/",
        groups="tests/model/size-groups.txt",
        size_column="biovolume_um3",
        value_column="biomass_ugl",
        out=out_file,
        append=False,
        force=False,
        pixels_to_um3=False,
        volume=True,
        quiet=True,
    )
    size_group.call(arguments)
    assert out_file.is_file()
    with open(out_file) as fh:
        lines = fh.readlines()
        assert len(lines) == 2
        header = lines[0].split(",")
        assert len(header) == 5
        assert header[0] == "time"
        assert header[1] == "small"
        assert header[2] == "large"
        assert header[3] == "total"
        assert header[-1].strip() == "volume_ml"
        first_result = list(filter(None, lines[1].split(",")))
        assert len(first_result) == len(header)
        small = float(first_result[1])
        large = float(first_result[2])
        total = float(first_result[3])
        volume = float(first_result[4])
        assert total == approx(1.748 + 0.034, rel=1e-3)
        assert small == approx(0.0342, rel=1e-3)
        assert large == approx(1.748, rel=1e-3)
        assert volume == approx(0.985, rel=1e-3)


def test_main_no_value_column(tmp_path):
    out_file = tmp_path / "out.csv"
    arguments = Args(
        features="tests/data/feat/",
        groups="tests/model/size-groups.txt",
        size_column="biovolume_um3",
        value_column=None,
        out=out_file,
        append=False,
        force=False,
        pixels_to_um3=False,
        volume=False,
        quiet=True,
    )
    size_group.call(arguments)
    assert out_file.is_file()
    with open(out_file) as fh:
        lines = fh.readlines()
        assert len(lines) == 2
        header = lines[0].split(",")
        assert len(header) == 4
        assert header[0] == "time"
        assert header[1] == "small"
        assert header[2] == "large"
        assert header[-1].strip() == "total"
        first_result = list(filter(None, lines[1].split(",")))
        assert len(first_result) == len(header)
        small = float(first_result[1])
        large = float(first_result[2])
        total = float(first_result[3])
        assert total == approx(1722.738 + 33.716, rel=1e-3)
        assert small == approx(33.716, rel=1e-3)
        assert large == approx(1722.738, rel=1e-3)
