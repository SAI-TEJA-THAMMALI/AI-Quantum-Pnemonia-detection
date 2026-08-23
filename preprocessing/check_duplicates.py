from pathlib import Path
from collections import defaultdict

import imagehash
from PIL import Image


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


def find_images(dataset_dir: Path) -> list[Path]:
    """Find all supported image files."""

    if not dataset_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory not found: {dataset_dir}"
        )

    return sorted(
        path
        for path in dataset_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def calculate_hash(
    image_path: Path,
):
    """Calculate perceptual hash for an image."""

    try:
        with Image.open(image_path) as image:
            return imagehash.phash(image)

    except Exception:
        return None


def find_duplicates(
    dataset_dir: Path,
    max_distance: int = 0,
) -> dict:
    """
    Find duplicate or near-duplicate images.

    max_distance:
        0 = exact perceptual hash match
        1-5 = increasingly tolerant similarity

    Returns a dictionary containing duplicate groups.
    """

    image_files = find_images(dataset_dir)

    hash_groups = defaultdict(list)

    failed_images = []

    for image_path in image_files:

        image_hash = calculate_hash(image_path)

        if image_hash is None:
            failed_images.append(image_path)
            continue

        hash_groups[str(image_hash)].append(
            image_path
        )

    duplicate_groups = [
        paths
        for paths in hash_groups.values()
        if len(paths) > 1
    ]

    return {
        "total_images": len(image_files),
        "hashed_images": (
            len(image_files)
            - len(failed_images)
        ),
        "failed_images": failed_images,
        "duplicate_groups": duplicate_groups,
    }


def save_duplicate_report(
    result: dict,
    dataset_dir: Path,
    report_path: Path,
) -> None:
    """Save duplicate information to a report."""

    report_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with report_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write("DUPLICATE IMAGE REPORT\n")
        file.write("=" * 60 + "\n\n")

        file.write(
            f"Dataset: {dataset_dir}\n"
        )

        file.write(
            f"Total images: "
            f"{result['total_images']}\n"
        )

        file.write(
            f"Duplicate groups: "
            f"{len(result['duplicate_groups'])}\n\n"
        )

        for index, group in enumerate(
            result["duplicate_groups"],
            start=1,
        ):

            file.write(
                f"Duplicate Group {index}\n"
            )

            for image_path in group:

                try:
                    relative_path = (
                        image_path.relative_to(
                            dataset_dir
                        )
                    )
                except ValueError:
                    relative_path = image_path

                file.write(
                    f"  {relative_path}\n"
                )

            file.write("\n")

        if result["failed_images"]:

            file.write(
                "Images that could not be hashed:\n"
            )

            for image_path in result[
                "failed_images"
            ]:

                file.write(
                    f"  {image_path}\n"
                )


def check_duplicates(
    dataset_dir: Path,
    report_path: Path | None = None,
) -> dict:
    """
    Run duplicate detection.

    No files are deleted or modified.
    """

    result = find_duplicates(
        dataset_dir
    )

    if report_path is not None:
        save_duplicate_report(
            result,
            dataset_dir,
            report_path,
        )

    return result