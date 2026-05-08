import numpy as np
from scipy.sparse import csr_matrix, diags
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

def create_admittance_matrices():
    N = 9
    # Branch data: (from_bus, to_bus, resistance, reactance, susceptance)
    branch_data = np.array([
        [0, 3, 0.0, 0.0576, 0.0],
        [3, 4, 0.017, 0.092, 0.158],
        [5, 4, 0.039, 0.17, 0.358],
        [2, 5, 0.0, 0.0586, 0.0],
        [5, 6, 0.0119, 0.1008, 0.209],
        [7, 6, 0.0085, 0.072, 0.149],
        [1, 7, 0.0, 0.0625, 0.0],
        [7, 8, 0.032, 0.161, 0.306],
        [3, 8, 0.01, 0.085, 0.176],
    ])

    M = branch_data.shape[0]  # Number of branches
    base_MVA = 100

    # Build incidence matrix A
    from_bus = branch_data[:, 0].astype(int)
    to_bus = branch_data[:, 1].astype(int)
    A = csr_matrix((np.ones(M), (from_bus, np.arange(M))), shape=(N, M)) + \
        csr_matrix((-np.ones(M), (to_bus, np.arange(M))), shape=(N, M))

    # Network impedance
    z = (branch_data[:, 2] + 1j * branch_data[:, 3]) / base_MVA

    # Bus admittance matrix Y_0
    Y_0 = A @ diags(1.0 / z) @ A.T

    # Shunt admittance from line charging
    y_sh = 0.5 * (1j * branch_data[:, 4]) * base_MVA
    Y_sh_diag = np.array((A @ diags(y_sh) @ A.T).diagonal()).flatten()
    Y_sh = diags(Y_sh_diag)

    # Extract conductance and susceptance matrices
    G0 = np.real(Y_0.toarray())  # Conductance matrix
    B0 = np.imag(Y_0.toarray())  # Susceptance matrix
    G_sh = np.real(Y_sh.toarray())  # Shunt conductance
    B_sh = np.imag(Y_sh.toarray())  #

    return G0, B0, G_sh, B_sh


def plot_power_flows(flow_P, filename="figures/power_flow.pdf"):
    edges = [
        (0, 3),
        (3, 4),
        (3, 8),
        (7, 8),
        (1, 7),
        (5, 4),
        (5, 6),
        (2, 5),
        (7, 6),
    ]
    pos = {
        0: (0, 0),
        3: (2, 0),
        4: (4, 0),
        8: (2, -1.5),
        7: (2, -3),
        1: (0, -3),
        5: (4, -1.5),
        6: (4, -3),
        2: (5, -1.5),
    }
    gen_nodes = [0, 1, 2]
    sink_nodes = [8, 4, 6]
    demands = {4: 54.0, 6: 60.0, 8: 75.0}

    flows = {(u, v): -float(flow_P[v, u]) for (u, v) in edges}
    losses = {(u, v): float(flow_P[u, v] + flow_P[v, u]) for (u, v) in edges}

    G = nx.DiGraph()
    G.add_edges_from(edges)

    node_size = 900
    edge_width = 3
    other_nodes = [n for n in G.nodes() if n not in gen_nodes + sink_nodes]

    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw_networkx_nodes(
        G, pos, nodelist=other_nodes, node_shape="o", node_color="#9ecae1",
        node_size=node_size, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=gen_nodes, node_shape="s", node_color="#31a354",
        node_size=node_size, ax=ax,
    )
    nx.draw_networkx_nodes(
        G, pos, nodelist=sink_nodes, node_shape="D", node_color="#fb6a4a",
        node_size=node_size, ax=ax,
    )

    nx.draw_networkx_labels(G, pos, ax=ax)

    for n in sink_nodes:
        if n in demands:
            x, y = pos[n]
            ax.annotate(
                f"demand = {demands[n]:.1f}",
                xy=(x, y),
                xytext=(25, 6),
                textcoords="offset points",
                fontsize=11,
                ha="left",
                va="center",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.2),
            )

    def shorten_point(a, b, offset_pixels):
        a_disp = ax.transData.transform(a)
        b_disp = ax.transData.transform(b)
        vec = b_disp - a_disp
        dist = np.hypot(vec[0], vec[1])
        if dist == 0:
            return a
        u = vec / dist
        new_a_disp = a_disp + u * offset_pixels
        return ax.transData.inverted().transform(new_a_disp)

    radius_pts = np.sqrt(node_size) / 2.0
    radius_pixels = radius_pts * (fig.dpi / 72.0)

    for (u, v) in edges:
        start = np.array(pos[u])
        end = np.array(pos[v])
        new_start = shorten_point(start, end, radius_pixels)
        new_end = shorten_point(end, start, radius_pixels)
        arrow = FancyArrowPatch(
            posA=new_start,
            posB=new_end,
            arrowstyle="->",
            mutation_scale=15,
            linewidth=edge_width,
            color="#444444",
            zorder=1,
        )
        ax.add_patch(arrow)

    vertical_branches = {(3, 8), (5, 4), (5, 6), (7, 8)}
    for (u, v) in edges:
        f = flows[(u, v)]
        loss = losses[(u, v)]
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length == 0:
            offx, offy = 0.0, 0.0
        else:
            ux, uy = -dy / length, dx / length
            offset = 0.12
            offx, offy = ux * offset, uy * offset

        label = f"{f:.2f}\n({loss:.2f})"

        if (u, v) in vertical_branches:
            x_offset_pts = 14
            if (u, v) == (7, 8) or (u, v) == (5, 4):
                x_offset_pts *= 2
        else:
            x_offset_pts = 0

        ax.annotate(
            label,
            xy=(mx + offx, my + offy),
            xytext=(x_offset_pts, 0),
            xycoords="data",
            textcoords="offset points",
            fontsize=9,
            ha="center",
            va="center",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.3),
        )

    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(filename)
