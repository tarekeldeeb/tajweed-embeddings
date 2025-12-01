"""Tree utilities for rule classification."""

from collections import namedtuple


class Exemplar:  # pylint: disable=too-few-public-methods
    """Simple container for labeled attribute exemplars."""
    __slots__ = ("label", "attributes", "count")

    def __init__(self, label, attributes, count):
        self.label = label
        self.attributes = attributes
        self.count = count


BooleanTreeNode = namedtuple("BooleanTreeNode", "attribute value gt lt")
FinalTreeNode = namedtuple("FinalTreeNode", "label count")


# I like namedtuples, maybe too much.
def tree2json(node):
    """Serialize a decision tree node to JSON-friendly dict."""
    # NB we don't save the counts - they're not needed at runtime.
    if hasattr(node, "label"):
        return {"label": node.label}
    return {
        "attribute": node.attribute,
        "value": node.value,
        "gt": tree2json(node.gt),
        "lt": tree2json(node.lt),
    }


def json2tree(node):
    """Deserialize a JSON dict into a decision tree node."""
    if "label" in node:
        return FinalTreeNode(node["label"], 0)

    return BooleanTreeNode(node["attribute"],
                           node["value"],
                           json2tree(node["gt"]),
                           json2tree(node["lt"]))
