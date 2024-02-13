from typing import Union
from torchtyping import TensorType as TT

import torch
import networkx as nx

from actions.action_meta import ActionMeta, DefaultActionMeta


# import matplotlib.pyplot as plt
def visualize_tree_graph(graph: Union[TT["state_shape", torch.int8], nx.DiGraph], action_meta: ActionMeta):
    # requires additional package: matplotlib, pygraphviz
    import matplotlib.pyplot as plt
    from networkx.drawing.nx_agraph import graphviz_layout

    if isinstance(graph, TT):
        graph = construct_tree_graph(graph, action_meta=action_meta)

    action_names = action_meta.action_names
    pos = graphviz_layout(graph, prog='dot')
    labels = {n: action_names[graph.nodes[n]['value']] for n in graph.nodes}
    edge_labels = nx.get_edge_attributes(graph, 'label')
    nx.draw(graph, pos, with_labels=True, arrows=True, labels=labels)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)
    plt.show()


def construct_tree_graph(tensor: TT["state_shape", torch.int8], action_meta: ActionMeta) -> nx.DiGraph:
    action_arities = action_meta.action_arities

    # first entry of tensor is a placeholder for token remained
    pre_order = tensor[1:].tolist()
    G = nx.DiGraph()
    itx = iter(range(len(pre_order)))

    def construct_tree_helper(parent_id=None, edge_label=None):
        node_id = next(itx)
        value = pre_order[node_id]
        arity = action_arities[value]

        print(node_id, value, arity)

        # Add the node to the graph. If it's the root, no edge_label is needed.
        G.add_node(node_id, value=value)

        # If there's a parent, add an edge with an attribute to indicate left or right child
        if parent_id is not None:
            G.add_edge(parent_id, node_id, label=edge_label)

        # For a binary node, recursively construct left then right child
        if arity == 2:
            construct_tree_helper(node_id, 'l')  # Left child
            construct_tree_helper(node_id, 'r')  # Right child

        # If the node is unary (assuming it has only a left child for simplicity)
        elif arity == 1:
            construct_tree_helper(node_id, 'l')

        # Leaf nodes don't have children

    construct_tree_helper()

    return G


def evaluate_tree_graph(graph: nx.DiGraph, action_meta: ActionMeta, data: TT["num_data", "num_features"]):
    assert graph.nodes[0], "Tree graph cannot be empty"
    assert data.shape[1] == action_meta.feat_num, "Mismatch in number of features"
    action_fns = action_meta.action_fns

    def evaluate_node(node_idx):
        value = graph.nodes[node_idx]["value"]
        fn = action_fns[value]
        children = list(graph.successors(node_idx))

        lc = len(children)

        if lc == 2:  # binary operator
            children.sort()
            return fn(evaluate_node(children[0]), evaluate_node(children[1]))
        elif lc == 1:
            return fn(evaluate_node(children[0]))
        else:
            return data[:, fn]

    return evaluate_node(0)


if __name__ == "__main__":
    def main():
        action_meta = DefaultActionMeta(num_features=2, has_constant=False)
        print(action_meta.action_arities)
        tensor = torch.tensor([0, 8, 0, 1], dtype=torch.int8)

        tree_graph = construct_tree_graph(tensor, action_meta=action_meta)
        # visualize_tree_graph(tree_graph, action_meta=action_meta)

        fake_data = 2 * torch.ones(10, 2)
        fake_data[:, 1] = torch.arange(10)
        print(fake_data)
        print(evaluate_tree_graph(tree_graph, action_meta=action_meta, data=fake_data))

    main()
