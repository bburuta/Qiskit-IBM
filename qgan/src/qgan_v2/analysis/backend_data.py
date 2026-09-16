"""Loading, calibration summaries, and plots for saved hardware snapshots.

Plotting dependencies are imported only by plotting functions. Reading a trusted
snapshot never creates an IBM Runtime service or downloads new calibrations.
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


QUBIT_METRICS = (
    "T1 (us)", "T2 (us)", "Readout error (%)", "Initialization error (%)",
    "P(1|0) (%)", "P(0|1) (%)", "Readout duration (ns)", "Frequency (GHz)",
)
QUBIT_ERROR_METRICS = ("Readout error (%)", "Initialization error (%)")
QUBIT_TIME_METRICS = ("T1 (us)", "T2 (us)", "Readout duration (ns)")
GATE_METRICS = ("Error (%)", "Duration (ns)")


def resolve_backend_path(file: str | Path, repo_root: str | Path) -> Path:
    """Resolve an absolute or repository-relative snapshot filename."""
    path = Path(file).expanduser()
    return (path if path.is_absolute() else Path(repo_root) / path).resolve()



def finite(v: Any) -> bool:
    return isinstance(v, (int, float, np.number)) and math.isfinite(v)


def fmt(v: Any) -> str:
    if v is None or (isinstance(v, (float, np.floating)) and not np.isfinite(v)):
        return 'N/A'
    if isinstance(v, (float, np.floating)):
        return f'{v:.6g}'
    return str(v)


def converted(params: Mapping[str, dict[str, Any]], name: str, unit: str) -> float | None:
    if name not in params:
        return None
    item = params[name]
    factors = {'s': 1, 'ms': 1e-3, 'us': 1e-6, 'µs': 1e-6, 'ns': 1e-9,
               'Hz': 1, 'kHz': 1e3, 'MHz': 1e6, 'GHz': 1e9, '': 1}
    original = item.get('unit', '')
    if original not in factors or unit not in factors:
        raise ValueError(f'Unsupported unit: {original} -> {unit}')
    return item['value'] * factors[original] / factors[unit]


def percentage(params: Mapping[str, dict[str, Any]], name: str) -> float | None:
    value = converted(params, name, '')
    return value * 100 if value is not None else None


def stats(values: Iterable[float | None]) -> dict[str, int | float | None]:
    """Describe finite calibration records without replacing missing data with zero."""
    values = np.asarray([v for v in values if finite(v)], dtype=float)
    if not len(values):
        return dict(N=0, Min=None, Q1=None, Median=None, Mean=None, Q3=None, Max=None)
    return {'N': len(values), 'Min': float(values.min()), 'Q1': float(np.quantile(values, .25)),
            'Median': float(np.median(values)), 'Mean': float(values.mean()),
            'Q3': float(np.quantile(values, .75)), 'Max': float(values.max())}


def load_backend_snapshot(file: str | Path, repo_root: str | Path) -> dict[str, Any]:
    """Read a trusted pickle and preserve missing calibrations and physical indices."""
    path = resolve_backend_path(file, repo_root)
    with path.open('rb') as stream:
        data = pickle.load(stream)
    required = {'configuration', 'properties', 'coupling_map'}
    if not isinstance(data, dict) or required - data.keys():
        raise ValueError(f'{path}: expected a repository backend snapshot with {sorted(required)}')
    config = data['configuration'].to_dict()
    props = data['properties'].to_dict()
    qubits, gates = [], []
    target = data.get('target')
    for q, values in enumerate(props['qubits']):
        params = {v['name']: v for v in values}
        frequency = converted(params, 'frequency', 'GHz')
        if frequency is None and target is not None and target.qubit_properties:
            qp = target.qubit_properties[q]
            if qp is not None and qp.frequency is not None:
                frequency = qp.frequency / 1e9
        qubits.append({'Qubit': q, 'T1 (us)': converted(params, 'T1', 'us'),
                       'T2 (us)': converted(params, 'T2', 'us'),
                       'Readout error (%)': percentage(params, 'readout_error'),
                       'Initialization error (%)': percentage(params, 'init_error'),
                       'P(1|0) (%)': percentage(params, 'prob_meas1_prep0'),
                       'P(0|1) (%)': percentage(params, 'prob_meas0_prep1'),
                       'Readout duration (ns)': converted(params, 'readout_length', 'ns'),
                       'Frequency (GHz)': frequency,
                       'Operational flag': bool(params['operational']['value']) if 'operational' in params else None})
    for g in props['gates']:
        params = {v['name']: v for v in g['parameters']}
        gates.append({'Gate': g['gate'], 'Qubits': tuple(g['qubits']),
                      'Error (%)': percentage(params, 'gate_error'),
                      'Duration (ns)': converted(params, 'gate_length', 'ns')})
    # Check completeness and units without assuming optional properties exist.
    assert len(qubits) == config['n_qubits']
    assert len(gates) == len(props['gates'])
    for r in qubits:
        for metric, accessor, factor in [('T1 (us)', 't1', 1e6), ('T2 (us)', 't2', 1e6),
                                          ('Readout error (%)', 'readout_error', 100),
                                          ('Readout duration (ns)', 'readout_length', 1e9)]:
            if finite(r[metric]):
                assert np.isclose(r[metric], getattr(data['properties'], accessor)(r['Qubit']) * factor)
    for r in gates:
        for metric, accessor, factor in [('Error (%)', 'gate_error', 100), ('Duration (ns)', 'gate_length', 1e9)]:
            if finite(r[metric]):
                assert np.isclose(r[metric], getattr(data['properties'], accessor)(r['Gate'], r['Qubits']) * factor)
    return {'path': path, 'data': data, 'config': config, 'properties': props,
            'qubits': qubits, 'gates': gates}


def qubit_calibration_summary(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize all finite qubit measurements in their displayed units."""
    return [{'Metric': metric, **stats(r[metric] for r in snapshot['qubits'])} for metric in QUBIT_METRICS]


def gate_calibration_summary(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    """Summarize provider records by gate type, retaining reciprocal records."""
    rows = []
    for gate in sorted({r['Gate'] for r in snapshot['gates']}):
        records = [r for r in snapshot['gates'] if r['Gate'] == gate]
        for metric in GATE_METRICS:
            rows.append({'Gate': gate, 'Metric': metric, **stats(r[metric] for r in records)})
    return rows


def plot_chip_layout(snapshot: dict[str, Any], repo_root: str | Path):
    """Return a calibrated physical-layout figure without displaying it."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import SymLogNorm
    import networkx as nx

    repo_root = Path(repo_root).resolve()
    config, qubits, gates = snapshot['config'], snapshot['qubits'], snapshot['gates']
    graph = nx.Graph()
    graph.add_nodes_from(range(config['n_qubits']))
    graph.add_edges_from(snapshot['data']['coupling_map'].get_edges())
    coords = config.get('coords')
    if coords is not None and len(coords) == graph.number_of_nodes():
        pos = {q: (float(xy[0]), -float(xy[1])) for q, xy in enumerate(coords)}
    else:
        pos = nx.spring_layout(graph, seed=0)
    cz = {}
    for r in gates:
        if r['Gate'] == 'cz' and finite(r['Error (%)']):
            cz.setdefault(tuple(sorted(r['Qubits'])), []).append(r['Error (%)'])
    node_vals = [r['Readout error (%)'] for r in qubits if finite(r['Readout error (%)'])]
    edge_vals = [float(np.mean(v)) for v in cz.values()]
    node_norm = SymLogNorm(linthresh=.1, vmin=0, vmax=max(max(node_vals, default=1), .1))
    edge_norm = SymLogNorm(linthresh=.1, vmin=0, vmax=max(max(edge_vals, default=1), .1))
    node_cmap, edge_cmap = plt.get_cmap('YlOrRd'), plt.get_cmap('viridis')
    lookup = {r['Qubit']: r for r in qubits}
    node_colors = [node_cmap(node_norm(lookup[q]['Readout error (%)']))
                   if finite(lookup[q]['Readout error (%)']) else '#bbbbbb' for q in graph.nodes()]
    edge_colors = [edge_cmap(edge_norm(np.mean(cz[tuple(sorted(pair))])))
                   if tuple(sorted(pair)) in cz else '#bbbbbb' for pair in graph.edges()]
    outlines = ['red' if lookup[q]['Operational flag'] is False else '#555555' for q in graph.nodes()]
    fig, ax = plt.subplots(figsize=(14, 11), layout='constrained')
    nx.draw_networkx_edges(graph, pos, ax=ax, edge_color=edge_colors, width=3)
    nx.draw_networkx_nodes(graph, pos, ax=ax, node_color=node_colors, node_size=320,
                           edgecolors=outlines, linewidths=1)
    nx.draw_networkx_labels(graph, pos, ax=ax, font_size=8)
    date = snapshot['data']['properties'].last_update_date.isoformat()
    ax.set_title(f"{config['backend_name']}\n{snapshot['path'].relative_to(repo_root) if snapshot['path'].is_relative_to(repo_root) else snapshot['path']}\nProperty last update: {date}")
    ax.set_aspect('equal'); ax.axis('off'); ax.margins(.04)
    fig.colorbar(ScalarMappable(norm=node_norm, cmap=node_cmap), ax=ax, shrink=.65, pad=.01,
                 label='Readout error (%)')
    fig.colorbar(ScalarMappable(norm=edge_norm, cmap=edge_cmap), ax=ax, shrink=.65, pad=.01,
                 label='CZ error (%) — mean per physical link')
    return fig


def backend_layout_description(snapshot: dict[str, Any]) -> str:
    """Describe the selected chip and distinguish physical from schematic coordinates."""
    config = snapshot['config']
    processor = config.get('processor_type') or {}
    links = {tuple(sorted(edge)) for edge in snapshot['data']['coupling_map'].get_edges()}
    coords = config.get('coords')
    layout_caption = (
        'Saved physical coordinates'
        if coords is not None and len(coords) == config['n_qubits']
        else 'Schematic connectivity layout — physical coordinates unavailable'
    )
    return (
        f"**{config['backend_name']}** · {config['n_qubits']} physical qubits · "
        f"{len(links)} physical links · processor "
        f"{processor.get('family', 'N/A')} revision {processor.get('revision', 'N/A')} · "
        f"native basis gates: {', '.join(config.get('basis_gates', []))}.\n\n{layout_caption}."
    )


def _plot_calibration_distributions(
    ax, categories, populations, comparison_colors, *, xlabel, linear_threshold, empty_message,
):
    """Draw quartile boxes and full-range whiskers, preserving absent and zero values."""
    comparison_labels = list(populations)
    count = len(comparison_labels)
    spacing = .75 / count
    for i, label in enumerate(comparison_labels):
        color = comparison_colors[label]
        for j, category in enumerate(categories):
            position = j + (i - (count - 1) / 2) * spacing
            summary = stats(populations[label][category])
            if not summary['N']:
                ax.text(.01, position, 'N/A', color=color, fontsize=7, va='center',
                        transform=ax.get_yaxis_transform())
                continue
            box = {'med': summary['Median'], 'q1': summary['Q1'], 'q3': summary['Q3'],
                   'whislo': summary['Min'], 'whishi': summary['Max'], 'fliers': []}
            ax.bxp([box], positions=[position], widths=spacing * .8,
                   orientation='horizontal', patch_artist=True, showfliers=False, manage_ticks=False,
                   boxprops={'facecolor': color, 'edgecolor': color, 'alpha': .65},
                   medianprops={'color': '#222222', 'linewidth': 1.2},
                   whiskerprops={'color': color, 'alpha': .65},
                   capprops={'color': color, 'alpha': .65})
            # A coloured median point also shows constant and zero calibrations.
            ax.plot(summary['Median'], position, 'o', color=color, markersize=3.5,
                    zorder=5, clip_on=False)
    ax.set_yticks(range(len(categories)), labels=categories)
    ax.set_ylim(max(len(categories), 1) - .5, -.5)
    ax.set_xscale('symlog', linthresh=linear_threshold)
    ax.set_xlim(left=0)
    ax.set_xlabel(xlabel)
    if not categories:
        ax.text(.5, .5, empty_message, ha='center', transform=ax.transAxes)
    ax.grid(axis='x', alpha=.25)
    ax.set_axisbelow(True)


def _calibration_comparison_figure(
    snapshots, categories, populations, title, height, *,
    xlabel='Error (%) — linear below 0.1%, logarithmic above',
    linear_threshold=.1, empty_message='No error estimates supplied',
):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not snapshots:
        raise ValueError('Choose at least one backend snapshot to compare.')
    palette = plt.get_cmap('tab10')
    colors = {label: palette(i % 10) for i, label in enumerate(snapshots)}
    handles = [Patch(facecolor=colors[label], alpha=.65,
                     label=f"{label} ({s['data']['properties'].last_update_date.date().isoformat()})")
               for label, s in snapshots.items()]
    fig, ax = plt.subplots(figsize=(11, height))
    _plot_calibration_distributions(
        ax, categories, populations, colors, xlabel=xlabel,
        linear_threshold=linear_threshold, empty_message=empty_message,
    )
    fig.suptitle(title, fontsize=15)
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .95),
               ncol=min(3, len(handles)), frameon=False, fontsize=9)
    fig.text(.5, .02,
             'Box: Q1–Q3 · centre line: median · whiskers: minimum–maximum · N/A: not supplied',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .86))
    return fig


def plot_qubit_error_comparison(snapshots: Mapping[str, dict[str, Any]]):
    """Return one figure comparing readout and initialization errors across snapshots."""
    categories = ['Readout', 'Initialization']
    populations = {
        label: {
            'Readout': [r['Readout error (%)'] for r in s['qubits']],
            'Initialization': [r['Initialization error (%)'] for r in s['qubits']],
        }
        for label, s in snapshots.items()
    }
    return _calibration_comparison_figure(snapshots, categories, populations, 'Qubit error comparison', 5)


def plot_gate_error_comparison(snapshots: Mapping[str, dict[str, Any]]):
    """Return one figure comparing all gate types with finite error estimates."""
    categories = sorted({r['Gate'] for s in snapshots.values() for r in s['gates']
                         if finite(r['Error (%)'])})
    populations = {
        label: {gate: [r['Error (%)'] for r in s['gates'] if r['Gate'] == gate]
                for gate in categories}
        for label, s in snapshots.items()
    }
    return _calibration_comparison_figure(
        snapshots, categories, populations, 'Gate/instruction error comparison',
        max(5, len(categories) * .62 + 2),
    )


def plot_qubit_time_comparison(snapshots: Mapping[str, dict[str, Any]]):
    """Compare T1, T2, and readout duration, displaying every time in microseconds."""
    categories = ['T1 (relaxation)', 'T2 (coherence)', 'Readout duration']
    populations = {
        label: {
            'T1 (relaxation)': [r['T1 (us)'] for r in s['qubits']],
            'T2 (coherence)': [r['T2 (us)'] for r in s['qubits']],
            'Readout duration': [r['Readout duration (ns)'] / 1000
                                 if finite(r['Readout duration (ns)']) else None
                                 for r in s['qubits']],
        }
        for label, s in snapshots.items()
    }
    return _calibration_comparison_figure(
        snapshots, categories, populations, 'Qubit calibration time comparison', 5,
        xlabel='Time (us) — linear below 1 us, logarithmic above',
        linear_threshold=1, empty_message='No time estimates supplied',
    )


def plot_gate_time_comparison(snapshots: Mapping[str, dict[str, Any]]):
    """Compare all gate/instruction types with finite duration estimates in nanoseconds."""
    categories = sorted({r['Gate'] for s in snapshots.values() for r in s['gates']
                         if finite(r['Duration (ns)'])})
    populations = {
        label: {gate: [r['Duration (ns)'] for r in s['gates'] if r['Gate'] == gate]
                for gate in categories}
        for label, s in snapshots.items()
    }
    return _calibration_comparison_figure(
        snapshots, categories, populations, 'Gate/instruction duration comparison',
        max(5, len(categories) * .62 + 2),
        xlabel='Duration (ns) — linear below 1 ns, logarithmic above',
        linear_threshold=1, empty_message='No duration estimates supplied',
    )
