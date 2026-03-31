"""Copy example notebooks into the documentation virtual filesystem.

This script is run by mkdocs-gen-files during the build process.
It copies all Jupyter notebooks from the examples/ directory into the
docs virtual filesystem so that mkdocs-jupyter can render them.
It also copies any data files that the notebooks reference.
"""

from pathlib import Path

import mkdocs_gen_files

EXAMPLES_DIR = Path("examples")

# Copy all notebooks and supporting data files
COPY_EXTENSIONS = {".ipynb", ".xlsx", ".csv", ".png", ".jpg", ".svg", ".json", ".nc"}


for source_path in sorted(EXAMPLES_DIR.rglob("*")):
    if source_path.is_file() and source_path.suffix in COPY_EXTENSIONS:
        dest_path = source_path  # keep the same relative path

        with mkdocs_gen_files.open(dest_path, "wb") as dest:
            dest.write(source_path.read_bytes())
