"""
Computation graph of a model, for inspection and documentation.

A model is a fixed composition of Funcs (Sum, Product, Compose, ...). This
module walks that composition and builds a graph whose nodes are the
parameters, the functions and (for a MaxLike problem) the likelihood,
constraints and regularizations, and whose edges show how values flow:
which parameter feeds which function, with which weight and onto which
feature axes of the output.

The graph can always be exported to Graphviz (DOT) or Mermaid text.
Rendering to an image, and inline display in Jupyter, use the optional
'graphviz' package (pip install maxlike[graph]) and need Graphviz itself.
"""
import shutil
import subprocess
import numpy as np

# node kinds and their Graphviz / Mermaid styling
_DOT_STYLE = {
    "param": 'shape=ellipse, style=filled, fillcolor="#dbeafe"',
    "func": 'shape=box, style="rounded,filled", fillcolor="#f8fafc"',
    "op": 'shape=box, style="rounded,filled", fillcolor="#fef3c7"',
    "likelihood": 'shape=box, style="filled,bold", fillcolor="#dcfce7"',
    "constraint": 'shape=note, style=filled, fillcolor="#fee2e2"',
    "regularization": 'shape=note, style=filled, fillcolor="#ede9fe"',
}
_MERMAID_SHAPE = {
    "param": ('(["', '"])'),
    "func": ('["', '"]'),
    "op": ('{{"', '"}}'),
    "likelihood": ('[["', '"]]'),
    "constraint": ('>"', '"]'),
    "regularization": ('>"', '"]'),
}
_SKIP_ATTRS = {"grad", "hess", "eval", "null"}


class Graph:
    """Nodes and edges of a model's computation graph."""

    def __init__(self):
        self.nodes = []   # (id, label, kind)
        self.edges = []   # (source id, target id, label, dashed)

    def add_node(self, label, kind="func"):
        node_id = "n%d" % len(self.nodes)
        self.nodes.append((node_id, label, kind))
        return node_id

    def add_edge(self, source, target, label="", dashed=False):
        self.edges.append((source, target, label, dashed))

    def __repr__(self):
        return "<Graph: %d nodes, %d edges>" % (len(self.nodes),
                                                len(self.edges))

    # ----------------------------------------------------------------------
    # export
    # ----------------------------------------------------------------------
    def to_dot(self, rankdir="LR"):
        """Graphviz DOT source."""
        def q(text):
            return '"%s"' % text.replace("\\", "\\\\").replace(
                '"', '\\"').replace("\n", "\\n")

        lines = ["digraph model {",
                 "  rankdir=%s;" % rankdir,
                 '  node [fontname="Helvetica", fontsize=11];',
                 '  edge [fontname="Helvetica", fontsize=9, color="#475569"];']
        for node_id, label, kind in self.nodes:
            lines.append("  %s [label=%s, %s];" % (node_id, q(label),
                                                    _DOT_STYLE[kind]))
        for source, target, label, dashed in self.edges:
            attrs = []
            if label:
                attrs.append("label=%s" % q(label))
            if dashed:
                attrs.append("style=dashed")
            lines.append("  %s -> %s%s;" % (
                source, target, " [%s]" % ", ".join(attrs) if attrs else ""))
        lines.append("}")
        return "\n".join(lines)

    def to_mermaid(self, direction="LR"):
        """Mermaid flowchart source (rendered natively by GitHub)."""
        def q(text):
            return text.replace('"', "#quot;").replace("\n", "<br>")

        lines = ["flowchart %s" % direction]
        for node_id, label, kind in self.nodes:
            left, right = _MERMAID_SHAPE[kind]
            lines.append("  %s%s%s%s" % (node_id, left, q(label), right))
        for source, target, label, dashed in self.edges:
            arrow = "-.->" if dashed else "-->"
            if label:
                arrow += '|"%s"|' % q(label)
            lines.append("  %s %s %s" % (source, arrow, target))
        return "\n".join(lines)

    def render(self, path, format=None, rankdir="LR"):
        """
        Renders the graph to an image file (png, svg, pdf, ...).

        Uses the optional 'graphviz' package (pip install maxlike[graph])
        if installed, else the Graphviz 'dot' program directly; either way
        Graphviz itself must be installed.
        """
        path = str(path)
        if format is None:
            format = path.rsplit(".", 1)[-1]
        source = self.to_dot(rankdir)
        try:
            import graphviz
        except ImportError:
            graphviz = None
        if graphviz is not None:
            data = graphviz.Source(source).pipe(format=format)
        else:
            dot = shutil.which("dot")
            if dot is None:
                raise RuntimeError(
                    "rendering needs Graphviz: install it (and optionally "
                    "pip install maxlike[graph]), or use to_dot() / "
                    "to_mermaid() to get the graph as text")
            data = subprocess.run([dot, "-T" + format], input=source.encode(),
                                  capture_output=True, check=True).stdout
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _repr_mimebundle_(self, include=None, exclude=None):
        # inline display in Jupyter when the optional graphviz package is
        # installed; plain text otherwise
        bundle = {"text/plain": repr(self)}
        try:
            import graphviz
            bundle["image/svg+xml"] = graphviz.Source(
                self.to_dot()).pipe(format="svg").decode()
        except Exception:
            pass
        return bundle


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------
def _fmt_number(x):
    x = float(x)
    return "%d" % x if x == int(x) else "%.4g" % x


def _fmt_value(v):
    if isinstance(v, bool) or isinstance(v, str):
        return repr(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return _fmt_number(v)
    if isinstance(v, (list, tuple, np.ndarray)):
        arr = np.asarray(v, dtype=object)
        if arr.size <= 4 and all(
                isinstance(e, (int, float, np.integer, np.floating))
                for e in arr.ravel()):
            return "[%s]" % ", ".join(_fmt_number(e) for e in arr.ravel())
        return "<%s>" % "x".join(map(str, np.shape(v))) \
            if isinstance(v, np.ndarray) else "<%d items>" % len(v)
    return None


def _func_label(foo):
    attrs = []
    for k, v in vars(foo).items():
        if k.startswith("_") or k in _SKIP_ATTRS or callable(v):
            continue
        text = _fmt_value(v)
        if text is not None:
            attrs.append("%s=%s" % (k, text))
    return "%s(%s)" % (type(foo).__name__, ", ".join(attrs))


def _axes(seq):
    return ", ".join(str(k) for k in seq)


def _wrap_label(weight, wrap):
    parts = []
    if weight != 1:
        parts.append("w=%s" % _fmt_number(weight))
    if len(wrap.feat_map):
        parts.append("axes (%s)" % _axes(wrap.feat_map))
    if len(wrap.dim_map) and wrap.n_dim:
        parts.append("dims (%s)" % _axes(wrap.dim_map))
    if wrap.feat_flip:
        parts.append("flip (%s)" % _axes(np.atleast_1d(wrap.feat_flip)))
    return ", ".join(parts)


# --------------------------------------------------------------------------
# walking the composition
# --------------------------------------------------------------------------
class _LazyParams:
    """Parameter nodes created on first use (for a standalone Func)."""

    def __init__(self, graph, names):
        self.graph = graph
        self.names = names or {}
        self.nodes = {}

    def __getitem__(self, k):
        if k not in self.nodes:
            name = self.names.get(k, "param %d" % k)
            self.nodes[k] = self.graph.add_node(name, "param")
        return self.nodes[k]

    def take(self, indexes):
        return [self[k] for k in indexes]

    def all(self, default=1):
        return [self[k] for k in range(max(len(self.nodes), default))]


class _Params(list):
    def take(self, indexes):
        return [self[k] for k in indexes]

    def all(self, default=1):
        return list(self)


def _connect(graph, inputs, target):
    for k, source in enumerate(inputs):
        graph.add_edge(source, target, "in %d" % k if len(inputs) > 1 else "")


def _walk(graph, foo, inputs):
    """Adds the nodes of 'foo' applied to 'inputs'; returns its output node."""
    # imported here to avoid circular imports
    from .func.func_base import Affine, Compose
    from .func.func_op import Sum, Product, FuncWrap, Collapse
    from .func.func import Linear

    if isinstance(foo, Compose):
        inner = _Params(_walk(graph, g, inputs) for g in foo.g_list)
        return _walk(graph, foo.f, inner)

    if isinstance(foo, (Sum, Product)):
        if isinstance(foo, Sum):
            atoms = foo.atoms
            label = "Sum"
            if foo.b:
                label += " + %s" % _fmt_number(foo.b)
        else:
            atoms = [(1, wrap) for wrap in foo.atoms]
            label = "Product"
        label += " (%d feature axes%s)" % (
            foo.n_feat, ", %d dist axes" % foo.n_dim if foo.n_dim else "")
        node = graph.add_node(label, "op")
        for weight, wrap in atoms:
            out = _walk(graph, wrap.foo, _Params(inputs.take(wrap.param_map)))
            graph.add_edge(out, node, _wrap_label(weight, wrap))
        return node

    if isinstance(foo, FuncWrap):
        node = graph.add_node("FuncWrap (%s feature axes)" % foo.n_feat, "op")
        out = _walk(graph, foo.foo, _Params(inputs.take(foo.param_map)))
        graph.add_edge(out, node, _wrap_label(1, foo))
        return node

    if isinstance(foo, Affine):
        terms = []
        if foo.a != 1:
            terms.append("%s *" % _fmt_number(foo.a))
        label = "Affine: %s f%s" % (
            " ".join(terms), " + %s" % _fmt_number(foo.b) if foo.b else "")
        node = graph.add_node(label.replace("  ", " "), "op")
        out = _walk(graph, foo.base, inputs)
        graph.add_edge(out, node)
        return node

    if isinstance(foo, Collapse):
        what = []
        if foo.sum_val is True:
            what.append("all feature axes")
        elif foo.sum_val is not False:
            what.append("feature axes (%s)" % _axes(
                np.atleast_1d(foo.sum_val)))
        if foo.sum_dim:
            what.append("dist axes")
        node = graph.add_node("Collapse: sum over %s" % (
            " and ".join(what) or "nothing"), "op")
        out = _walk(graph, foo.foo, inputs)
        graph.add_edge(out, node)
        return node

    # elementary function: it uses all the inputs of its context
    node = graph.add_node(_func_label(foo), "func")
    if isinstance(inputs, _LazyParams):
        count = len(foo.weights) if isinstance(foo, Linear) else 1
        inputs = inputs.take(range(count))
    _connect(graph, list(inputs), node)
    return node


def func_graph(foo, param_names=None):
    """Graph of a Func; parameter nodes are created as they are used."""
    graph = Graph()
    names = dict(enumerate(param_names)) if param_names else None
    _walk(graph, foo, _LazyParams(graph, names))
    return graph


def maxlike_graph(mle, param_names=None):
    """
    Graph of a MaxLike problem: parameters, model, likelihood, constraints
    and regularizations.
    """
    graph = Graph()
    params = _Params()
    for k, p in enumerate(getattr(mle, "params", ())):
        name = param_names[k] if param_names else "param %d" % k
        shape = "scalar" if p.shape == () else "shape %s" % (p.shape,)
        fixed = p.mask.sum()
        label = "%s\n%s%s" % (name, shape,
                              ", %d fixed" % fixed if fixed else "")
        params.append(graph.add_node(label, "param"))

    like = graph.add_node("%s likelihood" % type(mle).__name__, "likelihood")
    if mle.model is not None:
        out = _walk(graph, mle.model, params)
        graph.add_edge(out, like, "y")

    for param_map, _, g in mle.constraint:
        node = graph.add_node("constraint: %s = 0" % _func_label(g),
                              "constraint")
        for k in param_map:
            graph.add_edge(params[k], node, dashed=True)

    for param_map, h in mle.reg:
        node = graph.add_node("regularization: %s" % _func_label(h),
                              "regularization")
        for k in param_map:
            graph.add_edge(params[k], node, dashed=True)
        graph.add_edge(node, like, "- penalty", dashed=True)

    return graph
