"""Generate the documentation home page (``index.md``) from the repository
``README.md`` at build time, so the README is the single source of truth.

This replaces the previous ``--8<-- "README.md"`` snippet include. The snippet
was expanded by pymdownx.snippets *during* markdown conversion, i.e. after
``on_page_markdown``, so the ``docs/`` path prefixes could only be patched up
afterwards in the rendered HTML (an ``on_page_content`` hook that string-replaced
``src="./docs/``). Generating the page up front removes that workaround and, more
importantly, lets the docs render different markup from GitHub — which is what
the logo handling below needs.

The README lives at the repository root and refers to in-repo assets with a
``docs/`` prefix (e.g. ``./docs/fine_logo_v19_dark.svg``) so that they resolve on
GitHub. Inside the built site those same files live at the docs root, so the
prefix is stripped here. External URLs are untouched (they contain no ``docs/``
path segment).
"""

import re
from pathlib import Path

import mkdocs_gen_files

README = Path("README.md")

# Content wrapped in these HTML comments is shown on GitHub (where the comments
# are invisible) but removed from the generated docs home page.
README_ONLY = re.compile(
    r"<!-- readme-only:start -->.*?<!-- readme-only:end -->\n{0,2}", re.DOTALL
)

# --- Theme-aware logos -------------------------------------------------------
# On GitHub the README switches logos between light and dark with the HTML
# <picture> element (GitHub's supported mechanism, which follows the GitHub
# theme). Material for MkDocs does not use <picture> for this -- it switches
# images via the "#only-light" / "#only-dark" URL-fragment convention, which
# follows the docs palette toggle. So on the docs home page we swap the
# <picture> markup for its Material-native two-<img> equivalent.
#
# The logo region is wrapped in the README with
#     <!-- logo:NAME:start --> ... <!-- logo:NAME:end -->
# (invisible HTML comments on GitHub). LOGO_BLOCKS maps NAME to the version that
# replaces everything between those markers.
#
# The README and the docs both lay the row out as <p align="center">. The README
# used to use a borderless <table>, but GitHub strips the inline border styles and
# draws its own table chrome, so the logos sat in a visible grid; a centred
# paragraph avoids that on GitHub and avoids Material's data-table styling here.
#
# Sizing: the `width` attributes below are what GitHub honours, but on the docs
# site Material's ".md-typeset img" rule (specificity 0,1,1) sets `height: auto`
# and the images also carry a "hero-logo--NAME" class that extra.css sizes at a
# higher specificity. Keep the two in agreement.
_JSA_LOGOS = (
    "https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/ICE2_Logos"
)
_HELMHOLTZ_LOGOS = "https://raw.githubusercontent.com/FZJ-IEK3-VSA/README_assets/v.1.0.0/Helmholtz_Logos"

LOGO_BLOCKS = {
    "header": f"""\
<p align="center">
  <img src="./docs/fine_logo_v19_no_overlap.svg#only-light" alt="ETHOS.FINE logo" class="hero-logo hero-logo--fine" width="180">
  <img src="./docs/fine_logo_v19_dark.svg#only-dark" alt="ETHOS.FINE logo" class="hero-logo hero-logo--fine" width="180">
  &nbsp;&nbsp;
  <a href="https://www.fz-juelich.de/en/ice/ice-2">
    <img src="{_JSA_LOGOS}/JSA-Header.svg#only-light" alt="Jülich Systems Analysis" class="hero-logo hero-logo--jsa" width="300">
    <img src="{_JSA_LOGOS}/JSA-Header-dark.svg#only-dark" alt="Jülich Systems Analysis" class="hero-logo hero-logo--jsa" width="300">
  </a>
</p>""",
    "helmholtz": f"""\
<a href="https://www.helmholtz.de/en/">
    <img src="{_HELMHOLTZ_LOGOS}/Helmholtz-Logo-Dark-Blue-RGB.svg#only-light" alt="Helmholtz Association" class="hero-logo hero-logo--helmholtz" width="200">
    <img src="{_HELMHOLTZ_LOGOS}/Helmholtz-Logo-White-RGB.svg#only-dark" alt="Helmholtz Association" class="hero-logo hero-logo--helmholtz" width="200">
  </a>""",
}


def _swap_logo_blocks(text: str) -> str:
    """Replace each ``<!-- logo:NAME:start --> ... <!-- logo:NAME:end -->``
    region (markers included) with its Material-native version from
    ``LOGO_BLOCKS``. Plain string slicing, no regex; a missing marker pair
    leaves the text untouched.
    """
    for name, replacement in LOGO_BLOCKS.items():
        start = f"<!-- logo:{name}:start -->"
        end = f"<!-- logo:{name}:end -->"
        while start in text and end in text:
            before, _, rest = text.partition(start)
            _, _, after = rest.partition(end)
            text = before + replacement + after
    return text


def _readme_to_index(text: str) -> str:
    # Drop a leading UTF-8 BOM if present.
    text = text.lstrip("﻿")
    # Remove README-only blocks (redundant on the docs site).
    text = README_ONLY.sub("", text)
    # Swap the <picture> logo table for its Material-native equivalent.
    text = _swap_logo_blocks(text)
    # Rewrite the repo-root 'docs/' prefixes to be relative to the docs root.
    text = text.replace("./docs/", "./")  # HTML/Markdown asset paths
    text = text.replace("](docs/", "](")  # Markdown links
    return text.replace("`docs/", "`")  # inline-code folder mentions


with mkdocs_gen_files.open("index.md", "w") as fd:
    fd.write(_readme_to_index(README.read_text(encoding="utf-8")))

# "Edit this page" should point at the source of truth.
mkdocs_gen_files.set_edit_path("index.md", "README.md")
