from pathlib import Path

from download_modelscope_subset import unmatched_patterns


def test_unmatched_patterns_supports_exact_paths_and_globs(tmp_path: Path) -> None:
    archive = tmp_path / "Images" / "Diffusion_based" / "generator.zip"
    label = tmp_path / "label_csv_files" / "generator.csv"
    archive.parent.mkdir(parents=True)
    label.parent.mkdir(parents=True)
    archive.write_bytes(b"zip")
    label.write_text("header\n", encoding="utf-8")

    missing = unmatched_patterns(
        tmp_path,
        [
            "Images/Diffusion_based/generator.zip",
            "label_csv_files/*.csv",
            "Images/Real/missing.zip",
        ],
    )

    assert missing == ["Images/Real/missing.zip"]
