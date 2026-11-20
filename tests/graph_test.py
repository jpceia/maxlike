import re
import os
import shutil
import tempfile
import unittest
import numpy as np
import maxlike
from maxlike.func import (
    X, Vector, Sum, Linear, Logistic, Collapse, Quadratic)
from maxlike.func.tennis import Game, TieBreak, Set
from .derivatives_test import ELEMENTARY, composite


def poisson_problem(n=5):
    mle = maxlike.Poisson()
    mle.model = Sum(3)
    mle.model.add(X(), 0, 0)
    mle.model.add(-X(), 1, 1)
    mle.model.add(Vector(np.arange(2) - .5), 2, 2)
    mle.add_constraint([0, 1], Linear([1, 1]))
    mle.add_param(np.zeros(n))
    mle.add_param(np.zeros(n), np.arange(n) == 0)
    mle.add_param(0.)
    return mle


def check_consistent(test, graph):
    ids = {node_id for node_id, _, _ in graph.nodes}
    for source, target, _, _ in graph.edges:
        test.assertIn(source, ids)
        test.assertIn(target, ids)
    dot = graph.to_dot()
    test.assertTrue(dot.startswith("digraph"))
    test.assertEqual(dot.count("->"), len(graph.edges))
    mermaid = graph.to_mermaid()
    test.assertTrue(mermaid.startswith("flowchart"))
    # one line per node and per edge: no raw line breaks inside labels
    test.assertEqual(len(mermaid.splitlines()),
                     1 + len(graph.nodes) + len(graph.edges))


class Test(unittest.TestCase):

    def test_poisson_problem(self):
        g = poisson_problem().graph(["attack", "defence", "home"])
        check_consistent(self, g)
        labels = {label: kind for _, label, kind in g.nodes}
        self.assertEqual(labels["attack\nshape (5,)"], "param")
        self.assertEqual(labels["defence\nshape (5,), 1 fixed"], "param")
        self.assertEqual(labels["home\nscalar"], "param")
        self.assertEqual(labels["Poisson likelihood"], "likelihood")
        self.assertEqual(labels["Sum (3 feature axes)"], "op")
        self.assertIn("Vector(vector=[-0.5, 0.5])", labels)
        self.assertIn("constraint: Linear(weights=[1, 1]) = 0", labels)
        edge_labels = sorted(label for _, _, label, _ in g.edges if label)
        self.assertEqual(edge_labels,
                         ["axes (0)", "axes (2)", "w=-1, axes (1)", "y"])
        self.assertEqual(sum(dashed for *_, dashed in g.edges), 2)

    def test_regularization(self):
        mle = poisson_problem()
        mle.add_regularization(2, Quadratic(0, 2))
        g = mle.graph()
        labels = [label for _, label, _ in g.nodes]
        self.assertIn("regularization: Quadratic(u=0, s=2)", labels)
        self.assertIn("- penalty", [label for _, _, label, _ in g.edges])

    def test_tennis_compose(self):
        F = Sum(2)
        F.add(X(), 0, 0)
        F.add(-X(), 0, 1)
        model = Set() @ [
            Collapse(Game(), False),
            Collapse(TieBreak(), False)] @ Logistic() @ F
        g = model.graph(["strength"])
        check_consistent(self, g)
        ids = {label: node_id for node_id, label, _ in g.nodes}
        # the logistic output feeds both Game and TieBreak
        logistic = ids["Logistic()"]
        targets = {t for s, t, _, _ in g.edges if s == logistic}
        self.assertEqual(targets, {ids["Game()"], ids["TieBreak()"]})
        # Set receives its two inputs in order
        inputs = sorted(label for _, t, label, _ in g.edges
                        if t == ids["Set()"])
        self.assertEqual(inputs, ["in 0", "in 1"])
        # one parameter node, shared by both X() leaves
        self.assertEqual([k for _, _, k in g.nodes].count("param"), 1)

    def test_all_funcs(self):
        cases = dict(ELEMENTARY)
        cases.update(composite())
        for name, (foo, _) in cases.items():
            with self.subTest(name):
                check_consistent(self, foo.graph())

    @unittest.skipUnless(shutil.which("dot"), "Graphviz is not installed")
    def test_render(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.svg")
            poisson_problem().graph().render(path)
            with open(path) as f:
                self.assertTrue(re.search(r"<svg", f.read()))


if __name__ == "__main__":
    unittest.main()
