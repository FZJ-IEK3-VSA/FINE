

import inspect
import re

import pandas as pd

# Matches any Sphinx field header, e.g. ":param name: text", ":type name: text",
# ":returns: text". Group 1 is the field name (param/type/returns/...), group 2 is
# the optional argument (the parameter name for :param/:type), group 3 is any text
# that follows on the same line.
_FIELD_RE = re.compile(r"^:([A-Za-z]+)(?:\s+([\w]+))?:\s*(.*)$")
# Matches the "|br| * the default value is X" convention used throughout fine's docstrings.
_DEFAULT_RE = re.compile(r"the default value is\s+([^,.\n]+)", re.IGNORECASE)
_BULLET_RE = re.compile(r"^\*\s*")
# Matches standalone bold section headers used as dividers, e.g. "**Default arguments:**",
# "**Required arguments:**", "**Additional keyword arguments that can be passed via kwargs:**".
_SECTION_HEADER_RE = re.compile(r"^\*\*.+\*\*:?$")


def _clean_line(line):
    """Strip docutils/rst markup (|br|, leading bullet '*') from a single line."""
    return _BULLET_RE.sub("", line.replace("|br|", "").strip()).strip()


def parse_docstring_params(doc):
    """Parse Sphinx-style ':param:'/':type:' fields from a single docstring.

    Accumulates continuation lines (descriptions/types that span multiple
    lines, or that are left empty on the ':type name:' line itself with the
    real content following as a bullet list below it) until the next Sphinx
    field is encountered. Bold section-header lines (e.g. "**Default
    arguments:**") are dropped so they don't pollute the preceding field.

    :return: dict mapping parameter name -> {"description": str, "type": str, "default": str}
    """
    param_data = {}
    current_field, current_name, buffer = None, None, []

    def flush():
        if current_field not in ("param", "type") or not current_name:
            return
        entry = param_data.setdefault(current_name, {})
        default, cleaned = None, []
        for raw in buffer:
            match = _DEFAULT_RE.search(raw)
            if match and default is None:
                default = match.group(1).strip()
                continue
            line = _clean_line(raw)
            if line:
                cleaned.append(line)
        text = " ".join(cleaned).strip()
        entry["description" if current_field == "param" else "type"] = text
        if default is not None:
            entry["default"] = default

    for raw_line in doc.split("\n"):
        line = raw_line.strip()
        match = _FIELD_RE.match(line)
        if match:
            flush()
            current_field, current_name, rest = match.groups()
            buffer = [rest] if rest else []
            continue
        if not line or _SECTION_HEADER_RE.match(line):
            continue
        buffer.append(line)
    flush()

    return param_data


def collect_param_data(cls):
    """Merge parameter documentation across the whole class hierarchy.

    Subclasses (e.g. Storage) often only document the parameters specific to
    them in their own __init__ docstring; parameters inherited from a base
    class (e.g. esM, name, hasCapacityVariable, investPerCapacity... defined
    on Component) are documented once on that base class instead. Since
    inspect.getdoc(cls.__init__) only returns the docstring of the exact
    __init__ that was found (no automatic merging across the MRO), we walk
    cls.__mro__ ourselves and fill in gaps from parent classes, letting the
    most specific (sub)class win whenever a parameter is documented more than
    once.
    """
    param_data = {}
    for base in cls.__mro__:
        init = base.__dict__.get("__init__")
        if init is None:
            continue
        base_data = parse_docstring_params(inspect.getdoc(init) or "")
        for name, info in base_data.items():
            entry = param_data.setdefault(name, {})
            for key, value in info.items():
                entry.setdefault(key, value)
    return param_data


def build_param_table(cls):
    """Build a parameter reference table (argument/description/type/default)
    for a class constructor, combining its signature (for names and defaults)
    with the docstrings of the whole class hierarchy (for descriptions and
    types, including parameters only documented on a base class)."""
    param_data = collect_param_data(cls)
    sig = inspect.signature(cls.__init__)

    rows = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue

        info = param_data.get(name, {})
        type_value = info.get("type")
        if type_value == "":
            # a ":type name:" field existed but had no text at all
            type_value = "see below"

        rows.append(
            {
                "Argument": name,
                "Description": info.get("description", ""),
                "Type": type_value or "",
                "Default": info.get(
                    "default",
                    param.default if param.default is not inspect._empty else "/",
                ),
            }
        )

    return pd.DataFrame(rows)


def style_param_table(df):
    """Return a pandas Styler for a param table built by build_param_table,
    formatted for readability: 'Argument' as a left-aligned index, wrapped
    text columns, light row banding, and required arguments (Default == "/")
    shown in bold so they stand out from optional ones."""
    df = df.set_index("Argument")
    required = df["Default"] == "/"

    def bold_required(row):
        weight = "bold" if required.loc[row.name] else "normal"
        return [f"font-weight: {weight}"] * len(row)

    return (
        df.style.apply(bold_required, axis=1)
        .set_properties(
            **{
                "text-align": "left",
                "vertical-align": "top",
                "white-space": "normal",
                "max-width": "420px",
                "color": "black",
                "background-color": "white",
            }
        )
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [("border-collapse", "collapse")],
                },
                {
                    "selector": "th",
                    "props": [
                        ("text-align", "left"),
                        ("background-color", "white"),
                        ("color", "black"),
                        ("vertical-align", "top"),
                        ("border", "2px solid black"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("border", "2px solid black"),
                        ("padding", "6px 10px"),
                        ("color", "black"),
                        ("background-color", "white"),
                    ],
                },
                {
                    # the "Argument" column, used as the table's index
                    "selector": "th.row_heading",
                    "props": [
                        ("font-weight", "bold"),
                        ("font-size", "1.15em"),
                    ],
                },
            ]
        )
    )


def display_param_table(cls):
    """Convenience wrapper: build and immediately style a parameter table for cls."""
    return style_param_table(build_param_table(cls))
