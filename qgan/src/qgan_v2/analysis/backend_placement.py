"""Selectors and compact figures for Cartesian products of offline placements."""

from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from pathlib import Path
from typing import Any

from qgan_v2.analysis.backend_transpilation import BackendTranspilation, IMPLEMENTATION_OPTIONS, PRESET_OPTIONS


BACKEND_PLACEMENT_FILES = {
    'noisy': 'qgan/backends/ibm_basquecountry.pkl',
    'real-spsa': 'qgan/data/train/base-qml_torch-q4-real-SPSA-aerCPU-rand0-seed0/real_backend.pkl',
    'real-psr': 'qgan/data/train/base-qml_torch-q4-real-PSR-aerCPU-rand0-seed0/real_backend.pkl',
}
BACKEND_PLACEMENT_CONFIGS = {
    'noisy': {
        n: f'qgan/data/train/base-qml_torch-q{n}-noisy-SPSA-aerCPU-rand0-seed0/config.yaml'
        for n in (4, 8, 16)
    },
    'real-spsa': {4: 'qgan/data/train/base-qml_torch-q4-real-SPSA-aerCPU-rand0-seed0/config.yaml'},
    'real-psr': {4: 'qgan/data/train/base-qml_torch-q4-real-PSR-aerCPU-rand0-seed0/config.yaml'},
    'fake-real': {4: 'qgan/data/train/base-qml_torch-q4-fake_real-SPSA-aerCPU-rand0-seed0/config.yaml'},
}
PACKED_PLACEMENT_CONFIGS = {
    implementation: {
        backend: {4: f'qgan/data/train/base-runtime_packed-{packing}-q4-{execution}-SPSA-aerCPU-rand0-seed0/config.yaml'}
        for backend, execution in (('noisy', 'noisy'), ('fake-real', 'fake_real'))
    }
    for implementation, packing in (('runtime-packed-sep', 'separate'), ('runtime-packed-join', 'joined'))
}
BACKEND_LABELS = {'noisy': 'Noisy reference', 'real-spsa': 'Real SPSA reference',
                  'real-psr': 'Real PSR reference', 'fake-real': 'FakeSherbrooke'}
IMPLEMENTATION_LABELS = {'qml-torch': 'QML Torch', 'runtime-packed-sep': 'Runtime packed separate',
                         'runtime-packed-join': 'Runtime packed joined'}
CIRCUIT_LABELS = {'generator-discriminator': 'Generator–discriminator', 'real-discriminator': 'Real–discriminator'}
CIRCUIT_ALIASES = {'generator': 'generator-discriminator', 'discriminator': 'real-discriminator'}


def placement_group_color(label, fallback_index=0):
    """Give each real/fake copy its own stable color across comparison panels."""
    import matplotlib.colors as colors
    import matplotlib.pyplot as plt
    # Interleaved fake/real colors; neither idle grey nor auxiliary gold is used.
    palette = ('#0072B2', '#A6761D', '#7B3294', '#D55E00',
               '#009E73', '#CC79A7', '#C0392B', '#008B8B')
    parts = label.split()
    index = (2 * int(parts[1]) + int(parts[0] == 'real')
             if len(parts) == 2 and parts[0] in ('real', 'fake') and parts[1].isdigit()
             else 2 * fallback_index)
    return (palette[index] if index < len(palette)
            else colors.to_hex(plt.get_cmap('hsv')((index * .61803398875) % 1)))


def placement_group_label(label):
    parts = label.split()
    if len(parts) == 2 and parts[0] in ('real', 'fake') and parts[1].isdigit():
        return f'{parts[0].capitalize()} copy {int(parts[1]) + 1}'
    return label.capitalize()


@dataclass
class BackendPlacement:
    """One selected template in a combined figure."""

    backend: str
    n_qubits: int
    circuit: str
    template_index: int
    snapshot: dict[str, Any]
    transpilation: BackendTranspilation
    implementation: str = 'qml-torch'
    preset: str = 'base'

    @property
    def record(self) -> dict[str, Any]:
        return self.transpilation.records[self.template_index]


@dataclass
class BackendPlacementComparison:
    """Ordered panel selections, full circuit records, and exported figure paths."""

    placements: list[BackendPlacement]
    output_files: dict[str, Path] = field(default_factory=dict)
    skipped_combinations: list[dict[str, Any]] = field(default_factory=list)


def placement_selections(n_qubits, circuit, backend, implementation=None, preset=None):
    """Normalize scalar/iterable selectors and reject invalid or repeated choices."""
    def choices(value, name, allowed):
        if isinstance(value, (str, Integral)):
            values = (value,)
        else:
            try:
                values = tuple(value)
            except TypeError as error:
                raise ValueError(f'{name} must be a value or a sequence of values.') from error
        if not values:
            raise ValueError(f'{name} must contain at least one choice.')
        for item in values:
            if name == 'n_qubits':
                valid = isinstance(item, Integral) and not isinstance(item, bool) and item in allowed
            else:
                valid = isinstance(item, str) and item in allowed
            if not valid:
                raise ValueError(f'Invalid {name}: {item!r}. Choose from {tuple(allowed)}.')
        if len(set(values)) != len(values):
            raise ValueError(f'{name} choices must not repeat.')
        return tuple(int(item) for item in values) if name == 'n_qubits' else values

    circuits = tuple(CIRCUIT_ALIASES.get(item, item) for item in
                     choices(circuit, 'circuit', {**CIRCUIT_LABELS, **CIRCUIT_ALIASES}))
    if len(set(circuits)) != len(circuits):
        raise ValueError('circuit choices must not repeat, including aliases.')
    selection = (choices(n_qubits, 'n_qubits', (4, 8, 16)),
            circuits,
            choices(backend, 'backend', BACKEND_LABELS))
    if implementation is not None or preset is not None:
        selection = (*selection, choices('qml-torch' if implementation is None else implementation,
                                        'implementation', IMPLEMENTATION_OPTIONS))
    if preset is not None:
        selection = (*selection, choices(preset, 'preset', PRESET_OPTIONS))
    return selection


def placement_combinations(n_qubits, circuit, backend, implementation=None, preset=None):
    """Return panel order and column count, preserving each selector's order.

    Multiple sizes: columns are sizes, rows are backend/circuit pairs.
    Single size: columns are backends, optionally implementations too when the
    product fits in four columns; otherwise implementations group rows.
    """
    selections = placement_selections(n_qubits, circuit, backend, implementation, preset)
    sizes, circuits, backends = selections[:3]
    if len(sizes) > 1:
        combinations = [(b, c, n) for b in backends for c in circuits for n in sizes]
        columns = len(sizes)
    else:
        combinations = [(b, c, sizes[0]) for c in circuits for b in backends]
        columns = len(backends)
    if implementation is not None or preset is not None:
        implementations = selections[3]
        if len(sizes) == 1 and len(backends) * len(implementations) <= 4:
            combinations = [(b, c, sizes[0], impl) for c in circuits for impl in implementations for b in backends]
            columns = len(backends) * len(implementations)
        else:
            combinations = [(b, c, n, impl) for impl in implementations for b, c, n in combinations]
    if preset is not None:
        # With a single size, place presets side by side when the figure fits.
        presets = selections[4]
        if len(sizes) == 1 and columns * len(presets) <= 4:
            grouped_rows = [combinations[i:i + columns] for i in range(0, len(combinations), columns)]
            combinations = [(*item, p) for row in grouped_rows for p in presets for item in row]
            columns *= len(presets)
        else:
            combinations = [(*item, p) for p in presets for item in combinations]
    return combinations, columns


def fake_backend_snapshot():
    """Load FakeSherbrooke's bundled target/configuration without account access."""
    from qiskit_ibm_runtime.fake_provider import FakeSherbrooke
    backend = FakeSherbrooke()
    data = {'name': backend.name, 'target': backend.target,
            'configuration': backend.configuration(), 'properties': backend.properties(),
            'coupling_map': backend.coupling_map}
    return {'path': None, 'source_kind': 'builtin_fake_backend', 'data': data,
            'config': data['configuration'].to_dict(), 'properties': data['properties'].to_dict()}


def placement_layout_job(circuit, record):
    """Preserve packed groups and use global logical labels across branches."""
    groups = record.get('layout_groups', [record['initial_physical_qubits']])
    labels = record.get('layout_labels', ['initial placement'])
    return {'circuit': circuit, 'layout_groups': groups, 'layout_labels': labels,
            'group_colors': [placement_group_color(label, i) for i, label in enumerate(labels)],
            'logical_label_offsets': [sum(len(g) for g in groups[:i]) for i in range(len(groups))],
            'logical_labels_only': True}


def plot_backend_placements(placements: list[BackendPlacement], columns: int, *, title: str | None = None):
    """Return one figure; a single placement uses only the supplied figure title."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from qgan_v2.visualization.hardware_layout import draw_hardware_layout_on_axes, get_hardware_positions

    if not placements or columns < 1:
        raise ValueError('Choose nonempty placements and a positive column count.')
    columns = min(columns, len(placements))
    rows = (len(placements) + columns - 1) // columns
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(4.2 * columns, 3.7 * rows + .8), layout='constrained')
    positions = [point for p in placements for point in get_hardware_positions(p.snapshot['data']).values()]
    xs, ys = zip(*positions)
    pad_x = max((max(xs) - min(xs)) * .045, .5)
    pad_y = max((max(ys) - min(ys)) * .045, .5)
    for panel, ax in zip(placements, axes.flat):
        job = placement_layout_job(panel.transpilation.transpiled_circuits[panel.template_index], panel.record)
        variant = ' · size variant' if panel.transpilation.provenance['qubit_count_overridden'] else ''
        circuit_label = CIRCUIT_LABELS[panel.circuit]
        if panel.implementation == 'runtime-packed-join':
            circuit_label = ('Generator · fake → discriminator'
                             if panel.circuit == 'generator-discriminator' else
                             'Discriminator · real → D + fake → D')
        panel_title = f'{BACKEND_LABELS[panel.backend]} · {panel.n_qubits} qubits{variant}\n'
        if panel.preset != 'base' or len({p.preset for p in placements}) > 1:
            panel_title += f'{panel.preset} preset · '
        if panel.preset in ('ang', 'amp'):
            batch = panel.transpilation.provenance['requested_batch_size']
            panel_title += f'batch {batch}'
            if panel.implementation != 'qml-torch':
                panel_title += f' · {len(panel.record["layout_groups"])} packed copies'
            panel_title += '\n'
        if panel.implementation != 'qml-torch' or len({p.implementation for p in placements}) > 1:
            panel_title += IMPLEMENTATION_LABELS[panel.implementation] + '\n'
        panel_title += circuit_label
        if len(placements) == 1:
            panel_title = ''
        draw_hardware_layout_on_axes(
            job, panel.snapshot['data'], panel_title,
            ax, compact=True, show_legend=False,
        )
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)
    for ax in list(axes.flat)[len(placements):]:
        ax.remove()
    handles = [
        Line2D([], [], marker='o', linestyle='none', color='0.78', markersize=4, label='Idle qubits'),
        Line2D([], [], color='black', linewidth=1.5, alpha=.45, label='Used circuit links'),
    ]
    labels = dict.fromkeys(label for p in placements for label in p.record['layout_labels'])
    for label in labels:
        handles.append(Line2D([], [], marker='o', linestyle='none',
                              color=placement_group_color(label), markersize=7,
                              label=placement_group_label(label)))
    if any(set(p.record['active_physical_qubits']) - set(p.record['initial_physical_qubits']) for p in placements):
        handles.append(Line2D([], [], marker='o', linestyle='none', color='#f4a340', markersize=7,
                              label='Additional active qubits'))
    fig.legend(handles=handles, loc='outside lower center', ncol=min(5, len(handles)) if columns > 1 else 2,
               frameon=False, fontsize=9)
    fig.suptitle(title if title is not None else 'Offline reconstructed physical-qubit placements', fontsize=12)
    return fig
