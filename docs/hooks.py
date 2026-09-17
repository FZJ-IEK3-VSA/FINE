"""MkDocs hooks for ETHOS.FINE documentation.

Hooks run before plugins, so files copied here are visible to
mkdocs-jupyter for proper notebook → HTML conversion.
"""

import shutil
import urllib.request
from pathlib import Path

CODE_OF_CONDUCT_URL = (
    "https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/main/CODE_CONDUCT.md"
)
CODE_OF_CONDUCT_DST = Path("CODE_OF_CONDUCT.md")

EXAMPLES_SRC = Path("examples")
EXAMPLES_DST = Path("docs/examples")
# Compared against `Path.suffix.lower()` — the match must be case-insensitive.
# Several example figures are committed with an uppercase ".PNG" suffix, and a
# case-sensitive check silently skipped them, so the images 404'd on the built
# site while still resolving on a case-insensitive local filesystem.
COPY_EXTENSIONS = {
    ".ipynb",
    ".xlsx",
    ".csv",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".json",
    ".nc",
}


def on_pre_build(config) -> None:
    """Copy example notebooks and data files into docs/examples/ before the build.

    This must happen as a hook (not via gen-files) so that mkdocs-jupyter can
    see the .ipynb files during its on_files event and render them to HTML.
    """
    try:
        with urllib.request.urlopen(CODE_OF_CONDUCT_URL) as response:
            CODE_OF_CONDUCT_DST.write_bytes(response.read())
    except Exception as exc:
        print(f"Warning: could not fetch remote Code of Conduct: {exc}")

    for src in sorted(EXAMPLES_SRC.rglob("*")):
        if src.is_file() and src.suffix.lower() in COPY_EXTENSIONS:
            dst = EXAMPLES_DST / src.relative_to(EXAMPLES_SRC)
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def on_page_context(context, page, config, nav):
    """Point "Edit this page" at where the example notebooks really live.

    on_pre_build stages ``examples/`` into ``docs/examples/``, and that copy is
    gitignored, so MkDocs derives an edit URL under ``docs/examples/`` — a path
    that does not exist in the repository and 404s on GitHub. The notebooks
    themselves live in ``examples/`` at the repository root.

    ``docs/examples/index.md`` is a real tracked file, so it is left alone; only
    the staged ``.ipynb`` pages are rewritten. Spaces are percent-encoded while
    we are here: two of the notebook filenames contain them, which otherwise
    yields a malformed URL.
    """
    src_path = page.file.src_path.replace("\\", "/")
    if (
        page.edit_url
        and src_path.startswith("examples/")
        and src_path.endswith(".ipynb")
    ):
        page.edit_url = page.edit_url.replace("/docs/examples/", "/examples/").replace(
            " ", "%20"
        )
    return context
