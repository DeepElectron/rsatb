"""Declarative schema nodes.

One tree describes every input field, and it is walked three ways: to validate
an input, to fill its defaults, and to render documentation and JSON Schema.
A field is therefore written once -- the property the C++ config structs and
their pybind bindings already have, extended to the JSON layer.

Nothing here imports the compiled extension, so `rsatb check`, `rsatb doc`,
`rsatb schema` and `rsatb template` work on a machine that cannot build or run
the solver.
"""

from __future__ import annotations

import difflib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

ERROR = "error"
WARNING = "warning"

MISSING = object()


class Issue:
    """One problem found in an input, located by a path into the document."""

    def __init__(self, path: str, message: str, severity: str = ERROR, hint: str = ""):
        self.path = path
        self.message = message
        self.severity = severity
        self.hint = hint

    @property
    def is_error(self) -> bool:
        return self.severity == ERROR

    def __str__(self) -> str:
        where = self.path or "<root>"
        text = "{}: {}: {}".format(self.severity, where, self.message)
        if self.hint:
            text += "\n    hint: " + self.hint
        return text

    def as_dict(self) -> dict:
        return {
            "severity": self.severity,
            "path": self.path,
            "message": self.message,
            "hint": self.hint,
        }


def join_path(path: str, key: Any) -> str:
    if isinstance(key, int):
        return "{}[{}]".format(path, key)
    if path:
        return "{}.{}".format(path, key)
    return str(key)


def suggest(name: str, candidates: Sequence[str], label: str = "known keys") -> str:
    """Returns a 'did you mean' hint, or the full list when nothing is close."""
    close = difflib.get_close_matches(name, list(candidates), n=3, cutoff=0.6)
    if close:
        return "did you mean " + " or ".join(repr(c) for c in close) + "?"
    if candidates:
        return label + ": " + ", ".join(sorted(candidates))
    return ""


# --- nodes ----------------------------------------------------------------


class Node:
    """Base of every schema node."""

    type_name = "any"

    def __init__(self, doc: str, example: Any = MISSING):
        self.doc = doc
        self.example = example

    def check(self, value: Any, path: str) -> Tuple[Any, List[Issue]]:
        """Validates `value` and returns it with defaults filled in."""
        return value, []

    def describe(self) -> str:
        """One-line type summary used by the documentation renderer."""
        return self.type_name

    def json_schema(self) -> dict:
        return {"description": self.doc}


class Scalar(Node):
    """A single number, string or boolean."""

    def __init__(self, doc, type_name="number", choices=None, minimum=None,
                 maximum=None, exclusive_minimum=None, unit="", example=MISSING):
        super().__init__(doc, example)
        self.type_name = type_name
        self.choices = tuple(choices) if choices else ()
        self.minimum = minimum
        self.maximum = maximum
        self.exclusive_minimum = exclusive_minimum
        self.unit = unit

    def check(self, value, path):
        issues: List[Issue] = []
        if self.type_name == "boolean":
            if not isinstance(value, bool):
                return value, [Issue(path, "expected a boolean, got {}".format(_kind(value)))]
        elif self.type_name == "integer":
            if isinstance(value, bool) or not isinstance(value, int):
                return value, [Issue(path, "expected an integer, got {}".format(_kind(value)))]
        elif self.type_name == "number":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return value, [Issue(path, "expected a number, got {}".format(_kind(value)))]
            value = float(value)
        elif self.type_name in ("string", "path"):
            if not isinstance(value, str):
                return value, [Issue(path, "expected a string, got {}".format(_kind(value)))]

        if self.choices and value not in self.choices:
            issues.append(Issue(
                path,
                "{!r} is not a supported value".format(value),
                hint=suggest(str(value), [str(c) for c in self.choices],
                             "supported values")))
        if self.minimum is not None and value < self.minimum:
            issues.append(Issue(path, "must be >= {}, got {}".format(self.minimum, value)))
        if self.exclusive_minimum is not None and value <= self.exclusive_minimum:
            issues.append(Issue(path, "must be > {}, got {}".format(self.exclusive_minimum, value)))
        if self.maximum is not None and value > self.maximum:
            issues.append(Issue(path, "must be <= {}, got {}".format(self.maximum, value)))
        return value, issues

    def describe(self):
        text = self.type_name
        if self.choices:
            text = " | ".join(repr(c) for c in self.choices)
        if self.unit:
            text += " [{}]".format(self.unit)
        return text

    def json_schema(self):
        schema: Dict[str, Any] = {"description": self.doc}
        schema["type"] = "string" if self.type_name == "path" else self.type_name
        if self.choices:
            schema["enum"] = list(self.choices)
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.exclusive_minimum is not None:
            schema["exclusiveMinimum"] = self.exclusive_minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.unit:
            schema["x-unit"] = self.unit
        return schema


class Array(Node):
    """A JSON list whose entries all share one schema."""

    type_name = "array"

    def __init__(self, item: Node, doc: str, length: Optional[int] = None,
                 min_length: int = 0, example=MISSING):
        super().__init__(doc, example)
        self.item = item
        self.length = length
        self.min_length = min_length

    def check(self, value, path):
        if not isinstance(value, list):
            return value, [Issue(path, "expected a list, got {}".format(_kind(value)))]
        issues: List[Issue] = []
        if self.length is not None and len(value) != self.length:
            issues.append(Issue(path, "expected exactly {} entries, got {}".format(
                self.length, len(value))))
        if len(value) < self.min_length:
            issues.append(Issue(path, "expected at least {} entries, got {}".format(
                self.min_length, len(value))))
        resolved = []
        for index, entry in enumerate(value):
            entry, entry_issues = self.item.check(entry, join_path(path, index))
            resolved.append(entry)
            issues.extend(entry_issues)
        return resolved, issues

    def describe(self):
        if self.length is not None:
            return "[{} x {}]".format(self.length, self.item.describe())
        return "[{}, ...]".format(self.item.describe())

    def json_schema(self):
        schema = {"description": self.doc, "type": "array", "items": self.item.json_schema()}
        if self.length is not None:
            schema["minItems"] = self.length
            schema["maxItems"] = self.length
        elif self.min_length:
            schema["minItems"] = self.min_length
        return schema


class Mapping(Node):
    """A JSON object with caller-chosen keys and one schema for the values."""

    type_name = "object"

    def __init__(self, value: Node, doc: str, key_doc: str = "key",
                 key_check: Optional[Callable[[str, str], List[Issue]]] = None,
                 example=MISSING):
        super().__init__(doc, example)
        self.value = value
        self.key_doc = key_doc
        self.key_check = key_check

    def check(self, value, path):
        if not isinstance(value, dict):
            return value, [Issue(path, "expected an object, got {}".format(_kind(value)))]
        issues: List[Issue] = []
        resolved = {}
        for key, entry in value.items():
            key_path = join_path(path, key)
            if isinstance(key, str) and key.startswith("_"):
                resolved[key] = entry  # a note, as in Obj; see the comment there
                continue
            if self.key_check is not None:
                issues.extend(self.key_check(key, key_path))
            entry, entry_issues = self.value.check(entry, key_path)
            resolved[key] = entry
            issues.extend(entry_issues)
        return resolved, issues

    def describe(self):
        return "{{{}: {}}}".format(self.key_doc, self.value.describe())

    def json_schema(self):
        return {
            "description": self.doc,
            "type": "object",
            "additionalProperties": self.value.json_schema(),
        }


class Field:
    """A named entry of an `Obj`: a node plus whether and how it defaults."""

    def __init__(self, node: Node, required: bool = False, default: Any = MISSING,
                 note: str = ""):
        self.node = node
        self.required = required
        self.default = default
        self.note = note

    @property
    def has_default(self) -> bool:
        return self.default is not MISSING


class Obj(Node):
    """A JSON object with a fixed set of known keys.

    Closed by default: an unknown key is an error, not a silently ignored
    value. A typo has to fail, or it becomes a different calculation.
    """

    type_name = "object"

    def __init__(self, doc: str, fields: Dict[str, Field], closed: bool = True,
                 require_exactly_one: Sequence[Sequence[str]] = (),
                 checks: Sequence[Callable[[dict, str], List[Issue]]] = (),
                 example=MISSING):
        super().__init__(doc, example)
        self.fields = fields
        self.closed = closed
        self.require_exactly_one = [list(group) for group in require_exactly_one]
        self.checks = list(checks)

    def check(self, value, path):
        if not isinstance(value, dict):
            return value, [Issue(path, "expected an object, got {}".format(_kind(value)))]

        issues: List[Issue] = []
        resolved: Dict[str, Any] = {}

        if self.closed:
            for key in value:
                if key in self.fields:
                    continue
                # A leading underscore marks a note rather than a setting. It is
                # the one thing carried through uninterpreted, so an input can
                # explain itself without being rejected for it.
                if isinstance(key, str) and key.startswith("_"):
                    resolved[key] = value[key]
                    continue
                issues.append(Issue(
                    join_path(path, key), "unknown key {!r}".format(key),
                    hint=suggest(key, self.fields.keys())))

        for name, field in self.fields.items():
            key_path = join_path(path, name)
            if name in value:
                entry, entry_issues = field.node.check(value[name], key_path)
                resolved[name] = entry
                issues.extend(entry_issues)
            elif field.required:
                issues.append(Issue(path, "missing required key {!r}".format(name),
                                    hint=field.node.doc))
            elif field.has_default:
                default = field.default
                if isinstance(default, (dict, list)):
                    # Resolve the default through the schema so nested
                    # defaults are filled in too.
                    default, _ = field.node.check(_copy(default), key_path)
                resolved[name] = default

        for group in self.require_exactly_one:
            present = [name for name in group if name in value]
            if len(present) == 0:
                issues.append(Issue(path, "exactly one of {} is required".format(
                    ", ".join(repr(n) for n in group))))
            elif len(present) > 1:
                issues.append(Issue(path, "{} are mutually exclusive; give exactly one".format(
                    ", ".join(repr(n) for n in present))))

        if not any(issue.is_error for issue in issues):
            for check in self.checks:
                issues.extend(check(resolved, path))
        return resolved, issues

    def describe(self):
        return "object"

    def json_schema(self):
        schema: Dict[str, Any] = {
            "description": self.doc,
            "type": "object",
            "properties": {name: field.node.json_schema() for name, field in self.fields.items()},
            "additionalProperties": not self.closed,
        }
        required = [name for name, field in self.fields.items() if field.required]
        if required:
            schema["required"] = required
        for name, field in self.fields.items():
            if field.has_default:
                schema["properties"][name]["default"] = field.default
        if self.require_exactly_one:
            schema["oneOf"] = [{"required": [name]} for group in self.require_exactly_one
                               for name in group]
        return schema


class Union(Node):
    """A tagged union: one key selects which set of the others applies."""

    type_name = "object"

    def __init__(self, doc: str, tag: str, variants: Dict[str, Obj], tag_doc: str = ""):
        super().__init__(doc)
        self.tag = tag
        self.variants = variants
        self.tag_doc = tag_doc or "Selects the variant."

    def check(self, value, path):
        if not isinstance(value, dict):
            return value, [Issue(path, "expected an object, got {}".format(_kind(value)))]
        if self.tag not in value:
            return value, [Issue(
                path, "missing required key {!r}".format(self.tag),
                hint="one of: " + ", ".join(sorted(self.variants)))]
        tag_value = value[self.tag]
        if tag_value not in self.variants:
            return value, [Issue(
                join_path(path, self.tag), "{!r} is not a supported {}".format(tag_value, self.tag),
                hint=suggest(str(tag_value), self.variants.keys(),
                             "supported " + self.tag + "s"))]

        variant = self.variants[tag_value]
        body = {k: v for k, v in value.items() if k != self.tag}
        resolved, issues = variant.check(body, path)
        if isinstance(resolved, dict):
            resolved = dict(resolved)
            resolved[self.tag] = tag_value
        return resolved, issues

    def describe(self):
        return " | ".join(sorted(self.variants))

    def json_schema(self):
        branches = []
        for name, variant in self.variants.items():
            schema = variant.json_schema()
            schema["properties"] = dict(schema.get("properties", {}))
            schema["properties"][self.tag] = {"const": name, "description": self.tag_doc}
            schema["required"] = sorted(set(schema.get("required", [])) | {self.tag})
            branches.append(schema)
        return {"description": self.doc, "oneOf": branches}


class Choice(Node):
    """One value that may be written in more than one JSON shape.

    Dispatch is on the JSON type alone (`array` vs `object` vs `string`), so
    the reading of an input is never guessed from which keys happen to be
    present.
    """

    def __init__(self, doc: str, options: Dict[str, Node]):
        super().__init__(doc)
        self.options = options

    def check(self, value, path):
        kind = _kind(value)
        if kind not in self.options:
            return value, [Issue(path, "expected {}, got {}".format(
                " or ".join(sorted(self.options)), kind))]
        return self.options[kind].check(value, path)

    def describe(self):
        return " | ".join(node.describe() for node in self.options.values())

    def json_schema(self):
        return {"description": self.doc,
                "anyOf": [node.json_schema() for node in self.options.values()]}


# --- helpers --------------------------------------------------------------


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    if value is None:
        return "null"
    return type(value).__name__


def _copy(value):
    if isinstance(value, dict):
        return {k: _copy(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_copy(v) for v in value]
    return value


def validate(node: Node, document: Any) -> Tuple[Any, List[Issue]]:
    """Validates a whole document, returning it with defaults filled in."""
    return node.check(document, "")
