from collections import namedtuple

from sykepic.compute import probability

Args = namedtuple(
    "Args", "raw samples image_dir images model out batch_size num_workers force"
)


def test_call(tmp_path):
    out_dir = tmp_path / "out"
    arguments = Args(
        raw="./data/raw/valid/",
        samples=None,
        image_dir=None,
        images=None,
        model="./model/resnet18_20201022/",
        out=out_dir,
        batch_size=64,
        num_workers=2,
        force=False,
    )
    probability.call(arguments)
    out_csvs = list(out_dir.glob("**/*.csv"))
    assert len(out_csvs) == 1
    with open(out_csvs[0]) as fh:
        lines = fh.readlines()
        assert len(lines) == 3
        header = lines[0].split(",")
        assert len(header) == 51
        assert header[0] == "roi"
        roi_2 = list(filter(None, lines[1].split(",")))
        roi_3 = list(filter(None, lines[2].split(",")))
        assert len(roi_2) == len(header)
        assert len(roi_3) == len(header)
        assert int(roi_2[0]) == 2
        assert int(roi_3[0]) == 3
