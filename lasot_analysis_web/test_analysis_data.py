from pathlib import Path

import numpy as np

import analysis_data
from analysis_data import (
    SEARCH_MODE_INDEPENDENT,
    SEARCH_MODE_SYNC_GROUND_TRUTH,
    SEARCH_MODE_SYNC_PREDICTION,
    build_analyses,
    calculate_metrics,
    read_boxes,
    read_flags,
    scan_dataset,
    scan_predictions,
    search_reference_boxes,
)


def test_parsers(tmp_path: Path) -> None:
    boxes = tmp_path / "boxes.txt"
    boxes.write_text("0, 1, 10, 20\n2 3 4 5\n", encoding="utf-8")
    flags = tmp_path / "flags.txt"
    flags.write_text("0,1,\n0\n1,\n", encoding="utf-8")
    np.testing.assert_allclose(read_boxes(boxes), [[0, 1, 10, 20], [2, 3, 4, 5]])
    np.testing.assert_array_equal(read_flags(flags), [0, 1, 0, 1])


def test_metrics() -> None:
    gt = np.asarray([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=float)
    pred = np.asarray([[0, 0, 10, 10], [10, 0, 10, 10]], dtype=float)
    iou, distance = calculate_metrics(gt, pred)
    np.testing.assert_allclose(iou, [1, 0])
    np.testing.assert_allclose(distance, [0, 10])


def test_search_region_modes_use_previous_boxes() -> None:
    ground_truth = np.asarray([[1, 1, 10, 10], [2, 2, 20, 20]], dtype=float)
    prediction = np.asarray([[3, 3, 30, 30], [4, 4, 40, 40]], dtype=float)

    truth, predicted = search_reference_boxes(ground_truth, prediction, 1, SEARCH_MODE_INDEPENDENT)
    np.testing.assert_array_equal(truth, ground_truth[0])
    np.testing.assert_array_equal(predicted, prediction[0])
    truth, predicted = search_reference_boxes(ground_truth, prediction, 1, SEARCH_MODE_SYNC_GROUND_TRUTH)
    np.testing.assert_array_equal(truth, ground_truth[0])
    np.testing.assert_array_equal(predicted, ground_truth[0])
    truth, predicted = search_reference_boxes(ground_truth, prediction, 1, SEARCH_MODE_SYNC_PREDICTION)
    np.testing.assert_array_equal(truth, prediction[0])
    np.testing.assert_array_equal(predicted, prediction[0])

    first_truth, first_predicted = search_reference_boxes(ground_truth, prediction, 0, SEARCH_MODE_INDEPENDENT)
    np.testing.assert_array_equal(first_truth, ground_truth[0])
    np.testing.assert_array_equal(first_predicted, prediction[0])


def test_scan_and_match_nested_dataset(tmp_path: Path, monkeypatch) -> None:
    sequence = tmp_path / "dataset" / "unrelated-category" / "0007"
    images = sequence / "img"
    images.mkdir(parents=True)
    for name in ("1.jpg", "2.jpg"):
        (images / name).write_bytes(b"image placeholder")
    (sequence / "groundtruth.txt").write_text("0,0,10,10\n0,0,10,10\n", encoding="utf-8")
    (sequence / "full_occlusion.txt").write_text("0,1,\n", encoding="utf-8")
    (sequence / "out_of_view.txt").write_text("0\n0\n", encoding="utf-8")
    prediction_dir = tmp_path / "results"
    prediction_dir.mkdir()
    (prediction_dir / "0007.txt").write_text("0 0 10 10\n1 0 10 10\n", encoding="utf-8")

    scanned_directories: list[Path] = []
    real_scandir = analysis_data.os.scandir

    def recording_scandir(path):
        scanned_directories.append(Path(path))
        return real_scandir(path)

    monkeypatch.setattr(analysis_data.os, "scandir", recording_scandir)
    dataset, dataset_warnings = scan_dataset(tmp_path / "dataset")
    assert images not in scanned_directories
    predictions, prediction_warnings = scan_predictions(prediction_dir)
    analyses, missing, extra, build_warnings = build_analyses(dataset, predictions)
    assert not dataset_warnings + prediction_warnings + build_warnings
    assert not missing and not extra
    assert len(analyses) == 1 and analyses[0].name == "0007"
    assert analyses[0].frame_count == 2
    assert dataset["0007"]._images is None
    assert dataset["0007"].load_images() == [images / "1.jpg", images / "2.jpg"]
    scans_after_first_load = len(scanned_directories)
    dataset["0007"].load_images()
    assert len(scanned_directories) == scans_after_first_load
