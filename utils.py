import numpy as np
import pandas as pd
import networkx as nx


def predictions_to_csv(filename, predictions):
    ids = list(range(1, 2001))
    df = pd.DataFrame({"Id": ids, "Predicted": predictions})
    df.to_csv(filename, sep=',', index=False)

def plot_graph(g, with_labels = True, **kwargs):
    """0: orange,
       1: blue,
       2: green,
       3: red,
       4: purple,
       5: brown,
       6: pink,
       7: gray,
       9 and more: cyan"""
    def f_col(i):
        if i == 0 : return "C1" #orange
        if i == 1 : return "C0" #blue
        if i >= 9 :
            return "C9"
        return f"C{i}"

    node_labels = np.array([v[1][0] for v in g.nodes("labels")])
    if max(node_labels) >= 9:
        print(f"Warning, some labels are >= 9 : {node_labels[node_labels >= 9]}")
    node_colors = [f_col(l) for l in node_labels]
    nx.draw(g, node_color = node_colors, with_labels=with_labels, **kwargs)