"""Rendering the schema for a reader: `rsatb doc` and `rsatb schema`.

Both are generated from `spec.py` rather than written alongside it, so the
documentation cannot drift from what the validator accepts. If a field is
missing here, it is missing from the schema.
"""

from __future__ import annotations

import json
import textwrap
from typing import List, Optional

from . import spec
from .nodes import Array, Choice, Mapping, Node, Obj, Scalar, Union

WIDTH = 86


def pretty_json(value, indent: int = 2, width: int = 76) -> str:
    """Serializes an input, keeping short numeric rows on one line.

    `json.dumps(indent=2)` puts every component of every position on its own
    line, which turns a two-atom cell into thirty. A coordinate triple reads as
    a triple, so it stays one.
    """
    return _dump(value, indent, width, 0)


def _dump(value, indent: int, width: int, level: int) -> str:
    pad = " " * (indent * level)
    inner = " " * (indent * (level + 1))
    flat = json.dumps(value)
    if len(flat) + len(pad) <= width and _depth(value) <= 2:
        return flat
    if isinstance(value, dict):
        if not value:
            return "{}"
        rows = ['{}{}: {}'.format(inner, json.dumps(key),
                                  _dump(entry, indent, width, level + 1))
                for key, entry in value.items()]
        return "{\n" + ",\n".join(rows) + "\n" + pad + "}"
    if isinstance(value, list):
        if not value:
            return "[]"
        rows = [inner + _dump(entry, indent, width, level + 1) for entry in value]
        return "[\n" + ",\n".join(rows) + "\n" + pad + "]"
    return flat


def _depth(value) -> int:
    """How deeply containers nest: a scalar is 0, [1, 2] is 1, [[1], [2]] is 2."""
    if isinstance(value, dict):
        return 1 + max([_depth(entry) for entry in value.values()] or [0])
    if isinstance(value, list):
        return 1 + max([_depth(entry) for entry in value] or [0])
    return 0


def _wrap(text: str, indent: str) -> str:
    paragraphs = text.split("\n\n")
    out = []
    for paragraph in paragraphs:
        out.append(textwrap.fill(" ".join(paragraph.split()), width=WIDTH,
                                 initial_indent=indent, subsequent_indent=indent))
    return "\n\n".join(out)


def _status(field) -> str:
    if field.required:
        return "required"
    if field.has_default:
        return "default: {}".format(json.dumps(field.default))
    return "optional"


def _topic_for(node: Node) -> Optional[str]:
    """Returns the `rsatb doc` topic that covers this node, if there is one."""
    for name, candidate in spec.topics().items():
        if candidate is node:
            return name
    return None


def _render_obj(node: Obj, lines: List[str], depth: int, max_depth: int) -> None:
    indent = "  " * (depth + 1)
    for name, field in node.fields.items():
        head = "{}{:<24} {:<22} {}".format(indent, name, field.node.describe(), _status(field))
        lines.append(head.rstrip())
        doc = field.node.doc
        if field.note:
            doc = doc + " " + field.note if doc else field.note
        if doc:
            lines.append(_wrap(doc, indent + "    "))
        before = len(lines)
        _render_child(field.node, lines, depth + 1, max_depth)
        if len(lines) == before:
            # Not expanded here: point at the topic that does expand it, so a
            # reader is never left at a dead end.
            topic = _topic_for(_inner(field.node))
            if topic:
                lines.append("{}    see: rsatb doc {}".format(indent, topic))
        lines.append("")


def _inner(node: Node) -> Node:
    """Unwraps the containers a documented node may sit inside."""
    while True:
        if isinstance(node, Mapping):
            node = node.value
        elif isinstance(node, Array):
            node = node.item
        else:
            return node


def _render_child(node: Node, lines: List[str], depth: int, max_depth: int) -> None:
    if depth >= max_depth:
        return
    indent = "  " * (depth + 1)
    if isinstance(node, Obj):
        _render_obj(node, lines, depth, max_depth)
    elif isinstance(node, Union):
        for name, variant in node.variants.items():
            lines.append('{}{} = "{}"'.format(indent, node.tag, name))
            if variant.doc:
                lines.append(_wrap(variant.doc, indent + "    "))
            _render_obj(variant, lines, depth + 1, max_depth)
    elif isinstance(node, (Mapping, Array, Choice)):
        # A container does not add a level of its own, but what it holds does.
        children = ([node.value] if isinstance(node, Mapping)
                    else [node.item] if isinstance(node, Array)
                    else list(node.options.values()))
        for child in children:
            _render_child(child, lines, depth, max_depth)


def render(name: str, node: Node, max_depth: int = 1) -> str:
    """Renders one documentation topic."""
    lines = ["=" * WIDTH, name, "=" * WIDTH, ""]
    if node.doc:
        lines.append(_wrap(node.doc, ""))
        lines.append("")

    if isinstance(node, Obj):
        _render_obj(node, lines, -1, max_depth)
        if node.require_exactly_one:
            for group in node.require_exactly_one:
                lines.append("  exactly one of: {}".format(", ".join(group)))
            lines.append("")
    elif isinstance(node, Union):
        lines.append("  {!r} selects the variant. Supported values:".format(node.tag))
        lines.append("")
        for variant_name, variant in node.variants.items():
            lines.append('  {} = "{}"'.format(node.tag, variant_name))
            if variant.doc:
                lines.append(_wrap(variant.doc.split("\n\n")[0], "      "))
            lines.append("      full detail: rsatb doc {}.{}".format(name, variant_name))
            lines.append("")
    elif isinstance(node, Choice):
        for shape, option in node.options.items():
            lines.append("  written as a JSON {}: {}".format(shape, option.describe()))
            if option.doc:
                lines.append(_wrap(option.doc, "      "))
            _render_child(option, lines, 0, max(max_depth, 1))
            lines.append("")
    else:
        _render_child(node, lines, -1, max_depth)

    example = _find_example(node)
    if example is not None:
        lines.append("example")
        lines.append("")
        for line in pretty_json(example).splitlines():
            lines.append("  " + line)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _find_example(node: Node):
    from .nodes import MISSING
    if node.example is not MISSING:
        return node.example
    if isinstance(node, Union):
        for variant in node.variants.values():
            if variant.example is not MISSING:
                return variant.example
    if isinstance(node, Choice):
        for option in node.options.values():
            if option.example is not MISSING:
                return option.example
    return None


OVERVIEW = """\
RSATB reads one JSON file.

The file has four sections, and they answer four separate questions:

  structure   where the atoms are
  model       what the Hamiltonian is
  run         how the calculation is carried out
  tasks       which observables to compute

Nothing is written twice. The model reports the basis it works in, so orbital
counts and graph cutoffs are never input -- they are read from the parameters
or the basis files. Unknown keys are errors, not defaults, so a typo cannot
quietly become a different calculation.

The smallest complete input:

  {
    "structure": {"file": "graphene.vasp"},
    "model": {
      "type": "sktb",
      "basis": {"C": ["pz"]},
      "hamiltonian": {
        "onsite": {"C": {"pz": 0.0}},
        "pairs": {"C-C": {"pp_pi": {"kind": "constant", "value": -2.7, "r_cut": 1.6}}}
      }
    },
    "tasks": [{"type": "dos", "nt": 200}]
  }

The workflow:

  rsatb template sktb -o run.json   start from a working input
  rsatb doc model.sktb              read what each key means
  rsatb check run.json              validate it: schema, files, and consistency
  rsatb show run.json               see it with every default filled in
  rsatb run run.json                compute; mpirun -n N rsatb run run.json to scale

`check` needs neither MPI nor the compiled solver, so an input can be validated
anywhere it is written. It reports three kinds of problem: a malformed field, a
file that is not there, and two sections that disagree -- a task asking for the
velocity operator from a model that cannot build one, say.

Paths inside the file are resolved against the file's own directory, so an
input and its data can be moved together.

Results are written by rank 0 into `output.dir`: one .npz per task, plus a
summary.json holding the resolved input, the derived basis, the spectral
window, every scalar result, and the shape of every array.
"""


def overview() -> str:
    """Returns the top-level 'how do I use this' text plus the topic index."""
    lines = [OVERVIEW, "topics", ""]
    for name in sorted(spec.topics()):
        node = spec.topics()[name]
        first = " ".join(node.doc.split("\n\n")[0].split())
        if len(first) > 58:
            first = first[:57] + "..."
        lines.append("  {:<26} {}".format(name, first))
    lines.append("")
    lines.append("  rsatb doc <topic>     one topic in full")
    lines.append("  rsatb doc --all       every topic")
    lines.append("  rsatb schema          the same thing as JSON Schema")
    return "\n".join(lines) + "\n"


def render_all(max_depth: int = 1) -> str:
    """Renders every topic, for a reader who wants the whole surface at once."""
    parts = [overview()]
    for name in sorted(spec.topics()):
        parts.append(render(name, spec.topics()[name], max_depth))
    return "\n".join(parts)


def json_schema() -> dict:
    """Exports the input schema as JSON Schema (draft 2020-12)."""
    schema = spec.ROOT.json_schema()
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://github.com/rsatb/rsatb/input.schema.json"
    schema["title"] = "RSATB input"
    return schema
