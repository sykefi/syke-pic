from collections import namedtuple

from pytest import approx
from sykepic.compute import classification

Args = namedtuple(
    "Args",
    "probabilities features thresholds divisions out summarize append force",
)


def test_main(tmp_path):
    out_file = tmp_path / "out.csv"
    arguments = Args(
        probabilities="tests/data/prob/",
        features="tests/data/feat/",
        thresholds="tests/model/thresholds-2021.txt",
        divisions=None,
        out=out_file,
        summarize="biomass_ugl",
        append=False,
        force=False,
    )
    classification.main(arguments)
    assert out_file.is_file()
    with open(out_file) as fh:
        lines = fh.readlines()
        assert len(lines) == 2
        header = lines[0].split(",")
        # 49 classes remain, since
        # Dolichospermum-Anabaenopsis(-coiled) is merged into one class
        assert len(header) == 52
        assert header[0] == "Time"
        first_result = list(filter(None, lines[1].split(",")))
        assert len(first_result) == len(header)
        assert float(first_result[-1]) == approx(1.782, rel=1e-3)
