"""MkDocs hooks for ETHOS.FINE documentation.

Hooks run before plugins, so files copied here are visible to
mkdocs-jupyter for proper notebook → HTML conversion.
"""

import shutil
from pathlib import Path

EXAMPLES_SRC = Path("examples")
EXAMPLES_DST = Path("docs/examples")
COPY_EXTENSIONS = {".ipynb", ".xlsx", ".csv", ".png", ".jpg", ".svg", ".json", ".nc"}


def on_page_markdown(markdown, page, config, files) -> str:
    """Rewrite image paths in docs/index.md that originate from the README snippet.

    README.md uses ``./docs/<file>`` paths so images render on GitHub.
    MkDocs resolves paths relative to docs/, so ``./docs/`` must become ``./``.
    """
    if page.file.src_path == "index.md":
        markdown = markdown.replace("./docs/", "./")
    return markdown


def on_pre_build(config) -> None:
    """Copy example notebooks and data files into docs/examples/ before the build.

    This must happen as a hook (not via gen-files) so that mkdocs-jupyter can
    see the .ipynb files during its on_files event and render them to HTML.
    """
    for src in sorted(EXAMPLES_SRC.rglob("*")):
        if src.is_file() and src.suffix in COPY_EXTENSIONS:
            dst = EXAMPLES_DST / src.relative_to(EXAMPLES_SRC)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
