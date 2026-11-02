"""MkDocs hook: everything on the site that is derived from the code.

The input reference is not written by hand. It is rendered here from
`rsatb/inputs/spec.py` -- the same declaration `rsatb check` validates
against, `rsatb doc` prints, and `rsatb schema` exports -- so a key cannot be
documented differently from how it is accepted.

The hook also

* embeds real input files with `<!-- rsatb-input: examples/x.json -->`,
  with the explanatory `_comment` keys stripped, so a tutorial shows exactly
  the file a reader will run;
* validates every complete input in the docs (a JSON block with both
  `structure` and `model`) against the schema, failing the build on an error,
  so the examples cannot drift from what the program accepts;
* publishes the JSON Schema at `/rsatb.schema.json` and the landing-page
  figures from `doc/fig/`.

Nothing here imports the compiled extension: `rsatb.inputs` is pure Python.
"""

from __future__ import annotations

import html
import json
import logging
import os
import posixpath
import re
import sys
from typing import Dict, List, Optional

from mkdocs.exceptions import PluginError
from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.hooks.rsatb")

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from rsatb.inputs import check as _check  # noqa: E402
from rsatb.inputs import nodes as _nodes  # noqa: E402
from rsatb.inputs import report as _report  # noqa: E402
from rsatb.inputs import spec as _spec  # noqa: E402
from rsatb.inputs import templates as _templates  # noqa: E402

REF = "reference/input"
SPEC_SOURCE = "rsatb/inputs/spec.py"

# Figures shared with the README; served from the site, not copied into docs/.
FIGURES = ("RSATB_repo.png", "fig_validation.png", "fig_scaling.png", "fig_application.png")


# --------------------------------------------------------------------------
# Pages: which schema node is documented where.
# --------------------------------------------------------------------------

class Page:
    def __init__(self, uri: str, title: str, node, kicker: str, cli: str,
                 selector: str = "", prefix: str = ""):
        self.uri = uri            # e.g. reference/input/tasks/dos.md
        self.title = title
        self.node = node
        self.kicker = kicker      # small label above the title
        self.cli = cli            # the `rsatb doc` topic with the same content
        self.selector = selector  # JSON that selects this variant
        self.prefix = prefix      # key path of this object inside the input


def _pages() -> List[Page]:
    pages = [
        Page(REF + "/structure.md", "structure", _spec.STRUCTURE, "Input section",
             "structure", prefix="structure"),
        Page(REF + "/run.md", "run", _spec.RUN, "Input section", "run", prefix="run"),
        Page(REF + "/output.md", "output", _spec.OUTPUT, "Input section", "output",
             prefix="output"),
        Page(REF + "/model/sk-parameters.md", "Slater-Koster parameters",
             _spec.SK_PARAMETERS, "model.sktb", "model.sktb.parameters",
             prefix="model.hamiltonian"),
    ]
    for name, variant in _spec.MODEL.variants.items():
        pages.append(Page(REF + "/model/{}.md".format(name), name, variant, "Model family",
                          "model." + name, selector='"type": "{}"'.format(name),
                          prefix="model"))
    for name, variant in _spec.TASK.variants.items():
        pages.append(Page(REF + "/tasks/{}.md".format(name), name, variant, "Task",
                          "tasks." + name, selector='{{"type": "{}"}}'.format(name),
                          prefix="tasks[]"))
    return pages


def _link_targets() -> Dict[int, str]:
    """Nodes that have a page of their own, so a field holding one links there."""
    targets = {
        id(_spec.STRUCTURE): REF + "/structure.md",
        id(_spec.RUN): REF + "/run.md",
        id(_spec.OUTPUT): REF + "/output.md",
        id(_spec.MODEL): REF + "/model/index.md",
        id(_spec.TASK): REF + "/tasks/index.md",
        id(_spec.SK_PARAMETERS): REF + "/model/sk-parameters.md",
        id(_spec.RADIAL): REF + "/model/radial.md",
    }
    return targets


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

_QUOTED = re.compile(r"(?<![A-Za-z0-9])'([^'\s][^']{0,48}?)'(?![A-Za-z0-9])")
_DOC_PATH = re.compile(r"\(?doc/([A-Za-z0-9_]+\.md)\)?")


def _prose(text: str) -> str:
    """Schema doc text -> safe HTML paragraphs, with 'values' set as code."""
    out = []
    for paragraph in text.split("\n\n"):
        paragraph = " ".join(paragraph.split())
        if not paragraph:
            continue
        escaped = html.escape(paragraph, quote=False)
        escaped = _QUOTED.sub(lambda m: "<code>{}</code>".format(m.group(1)), escaped)
        escaped = _DOC_PATH.sub(
            lambda m: '(<a href="https://github.com/DeepElectron/rsatb/blob/main/doc/{0}">'
                      'design note</a>)'.format(m.group(1)), escaped)
        out.append("<p>{}</p>".format(escaped))
    return "\n".join(out)


def _first_paragraph(text: str) -> str:
    return " ".join(text.split("\n\n")[0].split())


def _rel(from_uri: str, to_uri: str) -> str:
    return posixpath.relpath(to_uri, posixpath.dirname(from_uri))


def _inner(node):
    while True:
        if isinstance(node, _nodes.Mapping):
            node = node.value
        elif isinstance(node, _nodes.Array):
            node = node.item
        else:
            return node


def _type_text(node) -> str:
    if isinstance(node, _nodes.Scalar):
        return node.type_name
    if isinstance(node, (_nodes.Union, _nodes.Obj)):
        return "object"
    if isinstance(node, _nodes.Choice):
        return " or ".join(_type_text(option) if not isinstance(option, _nodes.Array)
                           else "array" for option in node.options.values())
    return node.describe().replace(" | ", " or ")


def _shape_text(node) -> str:
    """Longer structural description for containers; empty for plain scalars."""
    if isinstance(node, (_nodes.Array, _nodes.Mapping)):
        return node.describe().replace(" | ", " or ").replace("'", "")
    return ""


def _unit(node) -> str:
    node = _inner(node)
    return getattr(node, "unit", "") if isinstance(node, _nodes.Scalar) else ""


def _constraints(node) -> List[str]:
    node = _inner(node)
    if not isinstance(node, _nodes.Scalar):
        return []
    out = []
    if node.exclusive_minimum is not None:
        out.append("&gt; {}".format(node.exclusive_minimum))
    if node.minimum is not None:
        out.append("&ge; {}".format(node.minimum))
    if node.maximum is not None:
        out.append("&le; {}".format(node.maximum))
    return out


def _choices(node) -> List[str]:
    node = _inner(node)
    if isinstance(node, _nodes.Scalar) and node.choices:
        return [json.dumps(c) for c in node.choices]
    return []


def _status_html(field) -> str:
    if field.required:
        return '<span class="rs-badge rs-badge--required">required</span>'
    if field.has_default and field.default not in ({}, None):
        return ('<span class="rs-badge rs-badge--default">default '
                '<code>{}</code></span>').format(html.escape(json.dumps(field.default)))
    return '<span class="rs-badge rs-badge--optional">optional</span>'


def _status_md(field) -> str:
    if field.required:
        return "**required**"
    if field.has_default and field.default not in ({}, None):
        return "`{}`".format(json.dumps(field.default))
    return "optional"


class Renderer:
    def __init__(self, page_uri: str, targets: Dict[int, str], anchor_prefix: str = ""):
        self.page_uri = page_uri
        self.anchor_prefix = anchor_prefix
        self.targets = targets
        self.glance: List[str] = []
        self.body: List[str] = []

    def link_for(self, node) -> Optional[str]:
        uri = self.targets.get(id(_inner(node)))
        if uri and uri != self.page_uri:
            return _rel(self.page_uri, uri)
        return None

    def obj(self, node, prefix: str = "") -> None:
        for name, field in node.fields.items():
            self.field(prefix + name, field)
        for group in getattr(node, "require_exactly_one", ()) or ():
            text = "Give exactly one of " + ", ".join("`{}{}`".format(prefix, k) for k in group) + "."
            self.body.append('!!! info "Mutually exclusive"\n    {}\n'.format(text))

    def field(self, key: str, field) -> None:
        node = field.node
        anchor = self.anchor_prefix + key.replace(".", "-").replace("[]", "").replace("*", "x")
        unit = _unit(node)
        # The unit is shown once, as its own badge, not inside the type.
        strip_unit = (lambda s: s.replace(" [{}]".format(unit), "")) if unit else (lambda s: s)
        type_text = strip_unit(_type_text(node))
        link = self.link_for(node)

        glance_type = type_text + (" · {}".format(unit) if unit else "")
        name_cell = "[`{}`](#{})".format(key, anchor)
        self.glance.append("| {} | {} | {} |".format(name_cell, glance_type, _status_md(field)))

        meta = ['<span class="rs-type">{}</span>'.format(html.escape(type_text))]
        shape = strip_unit(_shape_text(node))
        if shape and shape != type_text:
            meta.append('<span class="rs-shape"><code>{}</code></span>'.format(html.escape(shape)))
        if unit:
            meta.append('<span class="rs-unit">{}</span>'.format(html.escape(unit)))
        for constraint in _constraints(node):
            meta.append('<span class="rs-constraint">{}</span>'.format(constraint))
        meta.append(_status_html(field))

        self.body.append("### `{}` {{ #{} .rs-key }}\n".format(key, anchor))
        self.body.append('<div class="rs-meta">{}</div>\n'.format(" ".join(meta)))
        choices = _choices(node)
        if choices:
            self.body.append('<div class="rs-choices"><span>one of</span> {}</div>\n'.format(
                " ".join("<code>{}</code>".format(html.escape(c)) for c in choices)))
        doc = node.doc or ""
        if field.note:
            doc = (doc + " " + field.note).strip()
        if doc:
            self.body.append('<div class="rs-doc">\n{}\n</div>\n'.format(_prose(doc)))
        if link:
            self.body.append('<div class="rs-see">Documented in full on '
                             '<a href="{}">its own page</a>.</div>\n'.format(
                                 link.replace(".md", "/").replace("index/", "")))

        inner = _inner(node)
        if link:
            return
        if isinstance(node, _nodes.Obj):
            self.obj(node, key + ".")
        elif isinstance(node, _nodes.Choice):
            for shape_name, option in node.options.items():
                if isinstance(option, _nodes.Obj):
                    self.body.append('<div class="rs-variant">Written as an object, '
                                     'with these keys:</div>\n')
                    self.obj(option, key + ".")
                else:
                    self.body.append(
                        '<div class="rs-variant">Written as an array: <code>{}</code>. {}</div>\n'
                        .format(html.escape(option.describe()), html.escape(option.doc or "")))
        elif isinstance(inner, _nodes.Obj) and inner is not node:
            self.obj(inner, key + "[]." if isinstance(node, _nodes.Array) else key + ".*.")


def _example_for(page: Page):
    example = _report._find_example(page.node)
    if example is None and page.prefix == "tasks[]":
        # Every key with a default, which is what `rsatb show` would fill in.
        example = {"type": page.title}
        for name, field in page.node.fields.items():
            if name == "label":
                continue
            if field.has_default:
                example[name] = field.default
    return example


def render_page(page: Page, targets: Dict[int, str]) -> str:
    r = Renderer(page.uri, targets)
    if isinstance(page.node, _nodes.Obj):
        r.obj(page.node, "")
    lines = []
    lines.append("---\ndescription: {}\n---\n".format(
        json.dumps(_first_paragraph(page.node.doc or page.title))))
    lines.append('<p class="rs-kicker">{}</p>\n'.format(page.kicker))
    lines.append("# {}\n".format(page.title))
    lines.append('<div class="rs-lead">\n{}\n</div>\n'.format(_prose(page.node.doc or "")))

    facts = []
    if page.selector:
        facts.append("**Select with** `{}`".format(page.selector))
    if page.prefix and page.prefix != "tasks[]":
        facts.append("**Location** `{}`".format(page.prefix))
    if page.prefix == "tasks[]":
        facts.append("**Location** an entry of `tasks`")
    facts.append("**Same content offline** `rsatb doc {}`".format(page.cli))
    lines.append('<div class="rs-facts" markdown>\n' + "  \n".join(facts) + "\n</div>\n")

    if page.prefix == "tasks[]":
        lines.append('!!! abstract "Results"\n    What this task writes, array by array, is listed '
                     'under [Output files]({}#{}).\n'.format(
                         _rel(page.uri, "reference/outputs.md"), page.title))

    if r.glance:
        lines.append("## Key summary\n")
        lines.append("| Key | Type | Default |\n|---|---|---|")
        lines.extend(r.glance)
        lines.append("")

    example = _example_for(page)
    if example is not None:
        title = "Every key at its default" if page.prefix == "tasks[]" and \
            _report._find_example(page.node) is None else "Example"
        lines.append("## {}\n".format(title))
        lines.append("```json\n{}\n```\n".format(_report.pretty_json(example)))

    if r.body:
        lines.append("## Keys\n")
        lines.extend(r.body)
    return "\n".join(lines).rstrip() + "\n"


def render_radial(targets) -> str:
    uri = REF + "/model/radial.md"
    out = ["---\ndescription: {}\n---\n".format(json.dumps(_first_paragraph(_spec.RADIAL.doc))),
           '<p class="rs-kicker">model.sktb</p>\n', "# Radial functions\n",
           '<div class="rs-lead">\n{}\n<p>Each channel in <code>pairs</code> is one of '
           'these forms, selected by <code>kind</code>.</p>\n</div>\n'.format(_prose(_spec.RADIAL.doc)),
           '<div class="rs-facts" markdown>\n**Location** `model.hamiltonian.pairs.<A-B>.<channel>`  \n'
           '**Same content offline** `rsatb doc radial`\n</div>\n']
    for name, variant in _spec.RADIAL.variants.items():
        r = Renderer(uri, targets, anchor_prefix=name + "-")
        r.obj(variant, "")
        out.append('## `{0}` {{ #{0} }}\n'.format(name))
        out.append('<div class="rs-lead">\n{}\n</div>\n'.format(_prose(variant.doc)))
        out.append("| Key | Type | Default |\n|---|---|---|")
        out.extend(r.glance)
        out.append("")
        if variant.example is not _nodes.MISSING:
            out.append("```json\n{}\n```\n".format(_report.pretty_json(variant.example)))
        # Demote the per-key headings one level under the variant heading.
        out.extend(line.replace("### `", "#### `", 1) if line.startswith("### `") else line
                   for line in r.body)
    return "\n".join(out).rstrip() + "\n"


def _union_index(uri: str, title: str, kicker: str, union, child_dir: str, lead_extra: str,
                 cli: str) -> str:
    out = ["---\ndescription: {}\n---\n".format(json.dumps(_first_paragraph(union.doc))),
           '<p class="rs-kicker">{}</p>\n'.format(kicker), "# {}\n".format(title),
           '<div class="rs-lead">\n{}\n</div>\n'.format(_prose(union.doc)), lead_extra,
           '<div class="rs-facts" markdown>\n**Selected by** `"{}"`  \n'
           '**Same content offline** `rsatb doc {}`\n</div>\n'.format(union.tag, cli),
           '<div class="grid cards rs-cards" markdown>\n']
    for name, variant in union.variants.items():
        out.append("-   [`{0}`]({1}/{0}.md)\n\n    ---\n\n    {2}\n".format(
            name, child_dir, html.escape(_first_paragraph(variant.doc), quote=False)))
    out.append("</div>\n")
    return "\n".join(out)


TASK_GROUPS = (
    ("Spectrum and populations", ("energy_bounds", "dos", "pdos", "occupation", "fermi_level")),
    ("Real-space electronic structure", ("ldos", "density")),
    ("Electrical and optical response", ("dc", "dc_spectrum", "ac")),
    ("Dynamics and quantum geometry", ("msd", "quantum_metric")),
)


def render_tasks_index() -> str:
    union = _spec.TASK
    out = ["---\ndescription: {}\n---\n".format(json.dumps(_first_paragraph(union.doc))),
           '<p class="rs-kicker">Input section</p>\n', "# tasks\n",
           '<div class="rs-lead">\n{}\n</div>\n'.format(_prose(union.doc)),
           '<div class="rs-facts" markdown>\n**Location** `tasks`, a list  \n'
           '**Selected by** `"type"`  \n**Same content offline** `rsatb doc tasks`\n</div>\n',
           "Every task accepts an optional `label`, the name of its result in the "
           "output. Two tasks of the same type need different labels:\n",
           '```json\n"tasks": [\n  {"type": "dos", "label": "coarse", "dE": 0.3},\n'
           '  {"type": "dos", "label": "fine",   "dE": 0.01}\n]\n```\n']
    seen = set()
    for group, names in TASK_GROUPS:
        out.append("## {}\n".format(group))
        out.append("| Task | What it computes |\n|---|---|")
        for name in names:
            seen.add(name)
            variant = union.variants[name]
            out.append("| [`{0}`]({0}.md) | {1} |".format(
                name, html.escape(_first_paragraph(variant.doc), quote=False).replace("|", "/")))
        out.append("")
    missing = sorted(set(union.variants) - seen)
    if missing:
        raise PluginError("tasks not placed in a group on the tasks index: " + ", ".join(missing))
    out.append("## What each task needs\n")
    out.append(
        "Which tasks a model can run depends on the operators it provides. "
        "`rsatb check` reports a task that the model cannot run before the run "
        "starts.\n\n"
        "| Task | Needs |\n|---|---|\n"
        "| `ac`, `dc`, `dc_spectrum`, `quantum_metric` | the velocity operator |\n"
        "| `msd` | the velocity operator, and the Gamma point in `run.kpoints` |\n"
        "| `ldos`, `density` | an `lcao` model, and a single k-point at Gamma |\n"
        "| everything else | any model |\n\n"
        "[Choose a model](../../../guides/choose-model.md) lists which models "
        "provide the velocity operator.\n")
    return "\n".join(out)


def render_input_index() -> str:
    root = _spec.ROOT
    out = ["---\ndescription: {}\n---\n".format(json.dumps(_first_paragraph(root.doc))),
           '<p class="rs-kicker">Reference</p>\n', "# Input file\n",
           '<div class="rs-lead">\n{}\n</div>\n'.format(_prose(root.doc)),
           '<div class="rs-facts" markdown>\n**Format version** `{}`  \n'
           '**Machine-readable** [`rsatb.schema.json`](../../rsatb.schema.json) '
           '(JSON Schema 2020-12, the same as the output of `rsatb schema`)\n</div>\n'.format(
               _spec.INPUT_VERSION),
           "## Top-level keys\n",
           "| Key | Type | Default | Documented in |\n|---|---|---|---|"]
    pages = {"structure": "structure.md", "model": "model/index.md", "run": "run.md",
             "tasks": "tasks/index.md", "output": "output.md"}
    for name, field in root.fields.items():
        where = "[{}]({})".format(name, pages[name]) if name in pages else "below"
        out.append("| `{}` | {} | {} | {} |".format(
            name, _type_text(field.node), _status_md(field), where))
    out.append("")
    out.append("""## The smallest complete input

```json title="minimal.json"
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
  "tasks": [{"type": "dos", "dE": 0.1}]
}
```

## Rules the format keeps

**Unknown keys are errors.** A typo cannot quietly become a default; `rsatb check`
names the path and suggests the key you meant:

```text
error: run.kpoint: unknown key 'kpoint'
    hint: did you mean 'kpoints'?
```

**Comments are keys that start with `_`.** `"_comment"`, `"_note_convergence"` and
the like are carried through uninterpreted at any level.

**Paths are relative to the input file**, not to the working directory, so an
input and its data move together.

**Nothing is declared that can be derived.** Orbital counts and graph cutoffs are
read from the model's parameters or basis files. `tb_file` is the one exception,
because its file formats carry no basis header.

**Units** are eV for energies, Angstrom for lengths, Kelvin for temperatures and
femtoseconds for times, unless a key says otherwise. k-points are fractional.
""")
    return "\n".join(out)


def generated_pages() -> Dict[str, str]:
    targets = _link_targets()
    out = {}
    for page in _pages():
        out[page.uri] = render_page(page, targets)
    out[REF + "/model/radial.md"] = render_radial(targets)
    out[REF + "/index.md"] = render_input_index()
    out[REF + "/tasks/index.md"] = render_tasks_index()
    out[REF + "/model/index.md"] = _union_index(
        REF + "/model/index.md", "model", "Input section", _spec.MODEL, ".",
        '<div class="rs-lead"><p>The guide <a href="../../../guides/choose-model/">'
        'Choose a model</a> explains which model type fits which data.</p></div>\n',
        "model")
    # Name each page after its heading in the front matter too: section
    # indexes would otherwise be titled "Index" in llms.txt.
    for uri, content in out.items():
        heading = re.search(r"^# (.+)$", content, re.MULTILINE).group(1)
        out[uri] = content.replace("---\n", "---\ntitle: {}\n".format(json.dumps(heading)), 1)
    return out


# --------------------------------------------------------------------------
# Embedded inputs and validation
# --------------------------------------------------------------------------

_INCLUDE = re.compile(r"^(?P<indent>[ \t]*)<!--\s*rsatb-input:\s*(?P<path>\S+)\s*-->[ \t]*$",
                      re.MULTILINE)
_TEMPLATE = re.compile(r"^(?P<indent>[ \t]*)<!--\s*rsatb-template:\s*(?P<name>[\w-]+)"
                       r"(?:\s+as\s+(?P<file>\S+))?\s*-->[ \t]*$", re.MULTILINE)
_FENCE = re.compile(r"^(?P<indent>[ \t]*)```json(?P<attrs>[^\n]*)\n(?P<body>.*?)^(?P=indent)```",
                    re.MULTILINE | re.DOTALL)


def _strip_comments(value):
    if isinstance(value, dict):
        return {k: _strip_comments(v) for k, v in value.items() if not str(k).startswith("_")}
    if isinstance(value, list):
        return [_strip_comments(v) for v in value]
    return value


def _validate(document, where: str, base_dir: Optional[str]) -> None:
    _, issues = _check.resolve(document, base_dir, check_paths=base_dir is not None)
    errors = [issue for issue in issues if issue.is_error]
    if errors:
        raise PluginError("{}: input does not validate:\n  {}".format(
            where, "\n  ".join(str(e) for e in errors)))


def _expand_includes(markdown: str, page_src: str) -> str:
    def replace(match):
        rel = match.group("path")
        path = os.path.join(ROOT_DIR, rel)
        if not os.path.isfile(path):
            raise PluginError("{}: rsatb-input names a missing file: {}".format(page_src, rel))
        with open(path) as handle:
            document = json.load(handle)
        _validate(document, "{} (including {})".format(page_src, rel), os.path.dirname(path))
        body = _report.pretty_json(_strip_comments(document), width=68)
        indent = match.group("indent")
        block = '```json title="{}"\n{}\n```'.format(rel, body)
        # Marked so the validation pass below does not check it a second time
        # without its base directory.
        return "\n".join(indent + line for line in ("<!-- rsatb-validated -->\n" + block).splitlines())
    markdown = _INCLUDE.sub(replace, markdown)

    def replace_template(match):
        # Exactly what `rsatb template <name>` prints.
        name = match.group("name")
        try:
            document = _templates.get(name)
        except KeyError:
            raise PluginError("{}: unknown template {!r}".format(page_src, name))
        document = {"_comment": _templates.DESCRIPTIONS[name], **dict(document)}
        title = match.group("file") or (name + ".json")
        block = '```json title="{}"\n{}\n```'.format(title, _report.pretty_json(document))
        indent = match.group("indent")
        return "\n".join(indent + line for line in block.splitlines())
    return _TEMPLATE.sub(replace_template, markdown)


def _validate_fences(markdown: str, page_src: str) -> None:
    for match in _FENCE.finditer(markdown):
        before = markdown[:match.start()].rstrip().splitlines()
        if before and before[-1].strip() == "<!-- rsatb-validated -->":
            continue
        indent = match.group("indent")
        body = "\n".join(line[len(indent):] if line.startswith(indent) else line
                         for line in match.group("body").splitlines())
        try:
            document = json.loads(body)
        except ValueError:
            continue  # a fragment, not a document
        if isinstance(document, dict) and "structure" in document and "model" in document:
            _validate(document, page_src, None)


# --------------------------------------------------------------------------
# MkDocs events
# --------------------------------------------------------------------------

def _check_llmstxt_lists(config, uris) -> None:
    """Every generated reference page must be listed for llms.txt, or a new
    task in spec.py would silently be missing from the agent-facing index."""
    plugin = config.plugins.get("llmstxt")
    if plugin is None:
        return
    listed = set()
    for entries in plugin.config.sections.values():
        for entry in entries:
            listed.add(next(iter(entry)) if isinstance(entry, dict) else entry)
    missing = sorted(set(uris) - listed)
    if missing:
        raise PluginError("mkdocs.yml llmstxt sections do not list: " + ", ".join(missing))


def on_files(files, config):
    pages = generated_pages()
    _check_llmstxt_lists(config, pages)
    for uri, content in pages.items():
        existing = files.get_file_from_path(uri)
        if existing is not None:
            files.remove(existing)
        files.append(File.generated(config, uri, content=content))
    files.append(File.generated(config, "rsatb.schema.json",
                                content=json.dumps(_report.json_schema(), indent=2) + "\n"))
    for name in FIGURES:
        src = os.path.join(ROOT_DIR, "doc", "fig", name)
        if os.path.isfile(src):
            files.append(File.generated(config, "assets/figures/" + name, abs_src_path=src))
        else:
            log.warning("figure missing: doc/fig/%s", name)
    return files


def on_page_markdown(markdown, page, config, files):
    markdown = _expand_includes(markdown, page.file.src_uri)
    _validate_fences(markdown, page.file.src_uri)
    if page.file.src_uri.startswith(REF + "/"):
        page.edit_url = config["repo_url"].rstrip("/") + "/blob/main/" + SPEC_SOURCE
    return markdown
