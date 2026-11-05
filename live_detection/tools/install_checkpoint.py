"""Install the user's known checkpoint locally without publishing its weights."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def install(source, directory=ROOT / "models"):
    source, directory = Path(source).expanduser(), Path(directory)
    profile = json.loads((directory / "legacy_detector.json").read_text())
    expected = profile["source_sha256"]
    target = directory / "legacy_detector.pt"
    if target.exists():
        with target.open("rb") as handle:
            current = hashlib.file_digest(handle, "sha256").hexdigest()
        if current == expected:
            return target
        raise ValueError("An unrelated checkpoint already exists at the destination; move it before installing.")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=directory, suffix=".pending", delete=False) as output:
            temporary = Path(output.name)
            if zipfile.is_zipfile(source):
                with zipfile.ZipFile(source) as archive:
                    candidates = [i for i in archive.infolist() if
                                  i.filename.endswith("models/legacy_detector.pt") and not i.is_dir()]
                    if len(candidates) != 1:
                        raise ValueError("ZIP must contain exactly one models/legacy_detector.pt checkpoint.")
                    if candidates[0].file_size > 512 * 1024 * 1024:
                        raise ValueError("Checkpoint exceeds the expected size limit.")
                    with archive.open(candidates[0]) as handle:
                        shutil.copyfileobj(handle, output)
            else:
                if source.stat().st_size > 512 * 1024 * 1024:
                    raise ValueError("Checkpoint exceeds the expected size limit.")
                with source.open("rb") as handle:
                    shutil.copyfileobj(handle, output)
        with temporary.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual != expected:
            raise ValueError("SHA-256 mismatch: this is not the checkpoint verified for the supplied profile.")
        # Hard-link publication is atomic and will not overwrite another install.
        os.link(temporary, target)
        return target
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", help="Path to the saved project ZIP or original .pt checkpoint")
    args = parser.parse_args()
    try:
        target = install(args.source)
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        parser.exit(1, str(error) + "\n")
    print(f"Checkpoint verified and installed: {target}")
    print("Restart the application to load it. Predictions remain provisional.")


if __name__ == "__main__":
    main()
