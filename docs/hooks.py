"""MkDocs hooks for ETHOS.FINE documentation.

Hooks run before plugins, so files copied here are visible to
mkdocs-jupyter for proper notebook → HTML conversion.
"""

import fnmatch
import os
import posixpath
import re
import shutil
import urllib.parse
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

# For faster local previews, rendering can be limited to some notebooks, e.g.
#   FINE_DOCS_NOTEBOOKS="Examples" mkdocs serve
# The value is one or more folders or glob patterns relative to examples/,
# separated by ";", e.g. "Examples;09_PerfectForesight" or "Examples/4*".
# Unset (the default, e.g. on Read the Docs), all notebooks are rendered.
NOTEBOOK_FILTER_ENV = "FINE_DOCS_NOTEBOOKS"


def _matches(path: str, pattern: str) -> bool:
    """Return whether a path matches a glob pattern or lies in the folder `pattern`."""
    pattern = pattern.strip("/")
    return fnmatch.fnmatch(path, pattern) or path.startswith(pattern + "/")


def _isWanted(relPath: Path, patterns: list[str]) -> bool:
    """Return whether a file below examples/ should be staged for the build."""
    if relPath.suffix.lower() != ".ipynb" or not patterns:
        return True
    return any(_matches(relPath.as_posix(), pattern) for pattern in patterns)


def on_pre_build(config) -> None:
    """Copy example notebooks and data files into docs/examples/ before the build.

    This must happen as a hook (not via gen-files) so that mkdocs-jupyter can
    see the .ipynb files during its on_files event and render them to HTML.

    Copies whose source no longer exists (or that are filtered out via
    FINE_DOCS_NOTEBOOKS) are removed, so renamed or deleted examples are not
    rendered from stale copies. Unchanged files are not copied again, which keeps
    ``mkdocs serve`` from reacting to its own staging.
    """
    try:
        with urllib.request.urlopen(CODE_OF_CONDUCT_URL) as response:
            CODE_OF_CONDUCT_DST.write_bytes(response.read())
    except Exception as exc:
        print(f"Warning: could not fetch remote Code of Conduct: {exc}")

    patterns = [
        p.strip()
        for p in os.environ.get(NOTEBOOK_FILTER_ENV, "").split(";")
        if p.strip()
    ]
    if patterns:
        print(
            f"{NOTEBOOK_FILTER_ENV} is set: only rendering notebooks matching {patterns}"
        )

    wanted = {
        src.relative_to(EXAMPLES_SRC)
        for src in EXAMPLES_SRC.rglob("*")
        if src.is_file() and src.suffix.lower() in COPY_EXTENSIONS
    }
    wanted = {relPath for relPath in wanted if _isWanted(relPath, patterns)}

    # Remove stale staged copies. Only file types this hook creates are touched,
    # so tracked pages such as docs/examples/index.md are never removed.
    for dst in sorted(EXAMPLES_DST.rglob("*")):
        if dst.is_file() and dst.suffix.lower() in COPY_EXTENSIONS:
            if dst.relative_to(EXAMPLES_DST) not in wanted:
                dst.unlink()
    for directory in sorted(EXAMPLES_DST.rglob("*"), reverse=True):
        if directory.is_dir() and not any(directory.iterdir()):
            directory.rmdir()

    for relPath in sorted(wanted):
        src, dst = EXAMPLES_SRC / relPath, EXAMPLES_DST / relPath
        if dst.exists():
            srcStat, dstStat = src.stat(), dst.stat()
            if (
                srcStat.st_size == dstStat.st_size
                and srcStat.st_mtime == dstStat.st_mtime
            ):
                continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


# href attributes with a relative target, i.e. no scheme ("https:", "mailto:"),
# no absolute path and no pure anchor.
_RELATIVE_HREF = re.compile(r'href="(?![a-zA-Z][a-zA-Z0-9+.-]*:|/|#)([^"#?]+)([^"]*)"')


def _isExampleNotebook(page) -> bool:
    """Return whether a page is an example notebook staged by on_pre_build."""
    src_path = page.file.src_path.replace("\\", "/")
    return src_path.startswith("examples/") and src_path.endswith(".ipynb")


def on_page_content(html, page, config, files):
    """Make relative links in example notebooks work on the documentation site.

    Notebooks link to each other by their file names (e.g. ``11_sources.ipynb``),
    which works in Jupyter and on GitHub. MkDocs rewrites such links in Markdown
    pages, but not in notebooks rendered by mkdocs-jupyter, so on the site they
    would point to the copied ``.ipynb`` file, which the browser downloads.

    A link to a notebook that is a page of the site is pointed at that page; all
    other links are left as they are.
    """
    if not _isExampleNotebook(page):
        return html

    pageDir = posixpath.dirname(page.file.src_path.replace("\\", "/"))
    pageUrlDir = posixpath.dirname(page.url)

    def rewrite(match):
        target, suffix = match.group(1), match.group(2)
        srcPath = posixpath.normpath(
            posixpath.join(pageDir, urllib.parse.unquote(target))
        )
        targetFile = files.get_file_from_path(srcPath)
        if targetFile is not None and srcPath.endswith(".ipynb"):
            relativeUrl = posixpath.relpath(targetFile.url, pageUrlDir or ".")
            return f'href="{relativeUrl}{suffix}"'
        return match.group(0)

    return _RELATIVE_HREF.sub(rewrite, html)


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
