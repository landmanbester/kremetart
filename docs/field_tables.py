"""Sphinx extension: show an API object's Parameters, Raises and Returns as tables.

Sphinx renders each of these docstring sections as a bulleted ``name (type) – description`` list. This
rewrites the list, after the docstrings are read, as a table with one column per part, so a reader can
scan names and types down a column.
"""

from docutils import nodes
from sphinx import addnodes
from sphinx.application import Sphinx

_SEPARATORS = (" – ", " -- ")


def _split(paragraph: nodes.paragraph) -> tuple[list[nodes.Node], list[nodes.Node], list[nodes.Node]]:
    """Split ``name (type) – description`` into its three parts; missing parts come back empty."""
    children = list(paragraph.children)
    name: list[nodes.Node] = []
    if children and isinstance(children[0], addnodes.literal_strong):
        name = [children.pop(0)]
    kind: list[nodes.Node] = []
    if name and children and children[0].astext() == " (":
        children.pop(0)
        while children and children[0].astext() != ")":
            kind.append(children.pop(0))
        if children:
            children.pop(0)
    elif not name:
        # A Raises item has no name: the exception type comes first.
        while children and children[0].astext() not in _SEPARATORS:
            kind.append(children.pop(0))
    if children and children[0].astext() in _SEPARATORS:
        children.pop(0)
    return name, kind, children


def _cell(content: list[nodes.Node]) -> nodes.entry:
    entry = nodes.entry()
    if content and all(isinstance(node, nodes.Body) for node in content):
        entry.extend(content)
    else:
        entry += nodes.paragraph("", "", *content)
    return entry


def _table(header: tuple[str, ...], rows: list[list[list[nodes.Node]]]) -> nodes.table:
    table = nodes.table(classes=["field-table"])
    group = nodes.tgroup(cols=len(header))
    table += group
    group.extend(nodes.colspec(colwidth=1) for _ in header)
    head = nodes.thead()
    head += nodes.row("", *(nodes.entry("", nodes.paragraph(text=title)) for title in header))
    body = nodes.tbody()
    body.extend(nodes.row("", *(_cell(part) for part in row)) for row in rows)
    group += head
    group += body
    return table


def _items(body: nodes.field_body) -> list[tuple[nodes.paragraph, list[nodes.Node]]]:
    """Each item's first paragraph and any further blocks."""
    if len(body) == 1 and isinstance(body[0], nodes.bullet_list):
        items = [list(item.children) for item in body[0].children]
    else:
        items = [list(body.children)]
    return [(item[0], item[1:]) for item in items if item and isinstance(item[0], nodes.paragraph)]


def _tabulate(field: nodes.field, header: tuple[str, ...], columns: tuple[int, ...]) -> None:
    rows = []
    for paragraph, rest in _items(field[1]):
        name, kind, text = _split(paragraph)
        description = [nodes.paragraph("", "", *text), *rest] if rest else text
        parts = (name, kind, description)
        rows.append([parts[i] for i in columns])
    if rows:
        _replace_body(field, _table(header, rows))


def _replace_body(field: nodes.field, table: nodes.table) -> None:
    body = field[1]
    body.clear()
    body += table  # sets the parent links later transforms rely on


def _rewrite(app: Sphinx, doctree: nodes.document) -> None:
    for field_list in list(doctree.findall(nodes.field_list)):
        fields = {field[0].astext(): field for field in field_list.children if isinstance(field, nodes.field)}
        if "Parameters" in fields:
            _tabulate(fields["Parameters"], ("Name", "Type", "Description"), (0, 1, 2))
        if "Raises" in fields:
            _tabulate(fields["Raises"], ("Exception", "When"), (1, 2))
        if "Returns" in fields and "Return type" in fields:
            returned = fields["Returns"][1].deepcopy().children
            kind = fields["Return type"][1].deepcopy().children
            _replace_body(fields["Returns"], _table(("Type", "Description"), [kind, returned]))
            field_list.remove(fields["Return type"])


def setup(app: Sphinx) -> dict[str, bool]:
    app.connect("doctree-read", _rewrite)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
