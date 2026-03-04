"""Generate the API reference pages automatically.

This script is run by mkdocs-gen-files during the build process.
It discovers all Python modules in the fine package and generates
corresponding Markdown files with mkdocstrings directives.
"""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()

# Directories to skip
SKIP_DIRS = {"__pycache__", "aggregations", "expansionModules", "IOManagement", "subclasses"}

# Modules already documented in the User Guide > Python Package Description section
SKIP_MODULES = {
    "energySystemModel",
    "component",
    "sourceSink",
    "conversion",
    "transmission",
    "storage",
    "utils",
    "utilsPWLCF",
}

for path in sorted(Path("fine").rglob("*.py")):
    module_path = path.with_suffix("")
    doc_path = path.relative_to("fine").with_suffix(".md")
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts)

    # Skip __pycache__ and other non-module directories
    if any(skip in parts for skip in SKIP_DIRS):
        continue

    # Skip modules already documented in User Guide > Python Package Description
    if parts[-1] in SKIP_MODULES:
        continue

    if parts[-1] in ("__init__", "__pycache__"):
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")

    if not parts:
        continue

    nav[parts[1:] if len(parts) > 1 else parts] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:
        ident = ".".join(parts)
        fd.write(f"::: {ident}\n")

    mkdocs_gen_files.set_edit_path(full_doc_path, path)

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:
    nav_file.writelines(nav.build_literate_nav())
