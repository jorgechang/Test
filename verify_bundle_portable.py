#!/usr/bin/env python3
"""Run verify_bundle.py with Python-version-independent AST dumping.

The v13.45 logic hashes were generated with Python 3.13, where ast.dump()
omits empty optional fields by default. Isaac Sim 6.0.1 on OSCAR uses Python
3.12, whose ast.dump() includes those empty fields, producing different hashes
for identical source. This wrapper normalizes the dump format before executing
the original verifier. It does not modify the task or training code.
"""
from __future__ import annotations

import ast
import pathlib
import runpy

_ORIGINAL_DUMP = ast.dump


def _compact_dump(node: ast.AST, annotate_fields: bool = True, include_attributes: bool = False, *, indent=None, **kwargs) -> str:
    """Match Python 3.13 ast.dump(..., show_empty=False) on older Python."""
    try:
        return _ORIGINAL_DUMP(
            node,
            annotate_fields=annotate_fields,
            include_attributes=include_attributes,
            indent=indent,
            show_empty=False,
        )
    except TypeError:
        pass

    def render(value):
        if isinstance(value, ast.AST):
            fields = []
            for name in value._fields:
                field_value = getattr(value, name, None)
                if field_value is None or field_value == []:
                    continue
                rendered = render(field_value)
                fields.append(f"{name}={rendered}" if annotate_fields else rendered)
            if include_attributes:
                for name in getattr(value, "_attributes", ()):
                    if hasattr(value, name):
                        rendered = render(getattr(value, name))
                        fields.append(f"{name}={rendered}" if annotate_fields else rendered)
            return f"{value.__class__.__name__}({', '.join(fields)})"
        if isinstance(value, list):
            return "[" + ", ".join(render(item) for item in value) + "]"
        if isinstance(value, tuple):
            body = ", ".join(render(item) for item in value)
            if len(value) == 1:
                body += ","
            return "(" + body + ")"
        return repr(value)

    return render(node)


ast.dump = _compact_dump
HERE = pathlib.Path(__file__).resolve().parent
runpy.run_path(str(HERE / "verify_bundle.py"), run_name="__main__")
