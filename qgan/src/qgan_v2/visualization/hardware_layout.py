"""Shared hardware placement diagrams for packed and reconstructed circuits.

Only uses the supplied circuits and hardware information; never fetches a backend.
"""

import matplotlib.pyplot as plt
import numpy as np


# Get backend qubit coordinates or create a fallback grid
def get_hardware_positions(hardware_info):
    configuration = hardware_info['configuration']
    coords = getattr(configuration, 'coords', None)
    if coords:
        return {
            index: (float(x), -float(y))
            for index, (x, y) in enumerate(coords)
        }

    n_qubits = hardware_info['target'].num_qubits
    width = int(np.ceil(np.sqrt(n_qubits)))
    return {
        index: (index % width, -(index // width))
        for index in range(n_qubits)
    }


# Get the undirected hardware coupling edges
def get_hardware_edges(hardware_info):
    coupling_map = hardware_info['target'].build_coupling_map()
    return sorted({
        tuple(sorted((int(a), int(b))))
        for a, b in coupling_map.get_edges()
        if a != b
    })


# Get the physical two-qubit edges used by a transpiled circuit
def get_circuit_edges(circuit):
    return sorted({
        tuple(sorted(circuit.find_bit(qubit).index for qubit in instruction.qubits))
        for instruction in circuit.data
        if len(instruction.qubits) == 2 and instruction.operation.name != 'barrier'
    })


# Draw initial logical placement and active circuit links on the full chip.
def plot_hardware_layout(job, hardware_info, title):
    """Return a figure without displaying it or retrieving backend information."""
    fig, ax = plt.subplots(figsize=(14, 7))
    draw_hardware_layout_on_axes(job, hardware_info, title, ax)
    fig.tight_layout()
    return fig


def draw_hardware_layout_on_axes(job, hardware_info, title, ax, *, compact=False, show_legend=True):
    """Draw on supplied axes, sharing the packed layout style across panels.

    Compact panels label initial logical qubits (q0, q1, ...). Full-size packed
    plots keep both indices unless the job requests ``logical_labels_only``.
    Returns the axes without displaying.
    """
    positions = get_hardware_positions(hardware_info)
    hardware_edges = get_hardware_edges(hardware_info)
    circuit_edges = get_circuit_edges(job['circuit'])
    layout_groups = job['layout_groups']
    logical_labels_only = job.get('logical_labels_only', compact)
    layout_labels = job.get(
        'layout_labels',
        [f'copy {copy_index}' for copy_index in range(len(layout_groups))],
    )
    selected_qubits = set().union(*(set(group) for group in layout_groups))
    active_qubits = {
        job['circuit'].find_bit(qubit).index
        for instruction in job['circuit'].data
        if instruction.operation.name != 'barrier'
        for qubit in instruction.qubits
    }
    routing_qubits = active_qubits - selected_qubits
    colors = plt.get_cmap('tab20')(
        np.linspace(0, 1, max(len(layout_groups), 1))
    )
    colors = job.get('group_colors', colors)
    label_offsets = job.get('logical_label_offsets', [0] * len(layout_groups))

    for a, b in hardware_edges:
        ax.plot(
            [positions[a][0], positions[b][0]],
            [positions[a][1], positions[b][1]],
            color='0.86',
            linewidth=0.7,
            zorder=1,
        )

    idle_qubits = [qubit for qubit in positions if qubit not in selected_qubits | routing_qubits]
    ax.scatter(
        [positions[qubit][0] for qubit in idle_qubits],
        [positions[qubit][1] for qubit in idle_qubits],
        s=8 if compact else 28,
        color='0.78',
        edgecolors='white',
        linewidths=0.4,
        zorder=2,
        label='idle',
    )

    for a, b in circuit_edges:
        ax.plot(
            [positions[a][0], positions[b][0]],
            [positions[a][1], positions[b][1]],
            color='black',
            linewidth=1.5 if compact else 2,
            alpha=0.45,
            zorder=3,
        )

    for copy_index, group in enumerate(layout_groups):
        ax.scatter(
            [positions[qubit][0] for qubit in group],
            [positions[qubit][1] for qubit in group],
            s=90 if compact else 300,
            color=[colors[copy_index]],
            edgecolors='black',
            linewidths=0.8,
            zorder=4,
            label=layout_labels[copy_index],
        )
        for local_index, qubit in enumerate(group):
            ax.text(
                positions[qubit][0],
                positions[qubit][1],
                f'q{local_index + label_offsets[copy_index]}' if logical_labels_only else f'{qubit}\nq{local_index}',
                ha='center',
                va='center',
                fontsize=6 if compact else 7,
                color='white',
                zorder=5,
            )

    if routing_qubits:
        ax.scatter(
            [positions[q][0] for q in sorted(routing_qubits)],
            [positions[q][1] for q in sorted(routing_qubits)],
            s=90 if compact else 300, color='#f4a340', edgecolors='black', linewidths=0.8,
            zorder=4, label='additional active qubits',
        )
        for q in sorted(routing_qubits):
            ax.text(*positions[q], 'aux' if logical_labels_only else str(q),
                    ha='center', va='center', fontsize=6 if compact else 7, zorder=5)

    if not getattr(hardware_info['configuration'], 'coords', None):
        title += '\nSchematic grid: physical coordinates unavailable'
    ax.set_title(title, fontsize=10 if compact else None)
    ax.set_aspect('equal')
    ax.axis('off')
    if show_legend:
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    return ax
