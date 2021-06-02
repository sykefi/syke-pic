from pathlib import Path

from sykepic.compute import feature


def test_main(tmp_path):
    sample_paths = [
        Path("tests/data/raw/valid/D20180712T065600_IFCB114"),
    ]
    out_dir = tmp_path / "prob"
    samples_processed = feature.main(
        sample_paths,
        out_dir,
        parallel=False,
    )
    assert isinstance(samples_processed, set)
    assert len(samples_processed) == 1
    assert len(list(out_dir.glob("**/*.csv"))) == 1


def test_sample_features():
    sample_path = Path("tests/data/raw/valid/D20180712T065600_IFCB114")
    volume_ml, roi_features = feature.sample_features(sample_path)
    assert abs(volume_ml - 0.985) < 0.001
    assert len(roi_features) == 2
    first_roi = roi_features[0][0]
    assert first_roi == 2
    first_biovolume_px = roi_features[0][1]
    assert abs(first_biovolume_px - 6000.0) < 1000.0
    first_biovolume_um3 = roi_features[0][2]
    assert first_biovolume_um3 == feature.pixels_to_um3(first_biovolume_px)
