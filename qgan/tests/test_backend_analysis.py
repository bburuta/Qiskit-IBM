"""Exercise backend snapshot paths, units, missing data, and notebook workflows."""

from datetime import datetime, timezone
import pickle

import numpy as np
import pytest

from qgan_v2.analysis import BackendAnalysis
from qgan_v2.analysis.backend_data import (
    backend_layout_description,
    converted,
    load_backend_snapshot,
    plot_chip_layout,
    plot_gate_error_comparison,
    plot_gate_time_comparison,
    plot_qubit_time_comparison,
    qubit_calibration_summary,
)


class DictionaryModel:
    def __init__(self, values):
        self.values = values

    def to_dict(self):
        return self.values


class PropertiesModel(DictionaryModel):
    last_update_date = datetime(2026, 6, 13, tzinfo=timezone.utc)

    def t1(self, qubit):
        return 2e-4

    def readout_error(self, qubit):
        return .01

    def readout_length(self, qubit):
        return 2e-6

    def gate_error(self, gate, qubits):
        return 1.0

    def gate_length(self, gate, qubits):
        return 8e-8


class CouplingModel:
    def get_edges(self):
        return [(0, 1), (1, 0)]


def write_snapshot(path, *, readout=.01):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        'configuration': DictionaryModel({'backend_name': 'test_chip', 'n_qubits': 2,
                                          'basis_gates': ['cz']}),
        'properties': PropertiesModel({
            'qubits': [[{'name': 'T1', 'value': .2, 'unit': 'ms'},
                        {'name': 'readout_error', 'value': readout, 'unit': ''},
                        {'name': 'readout_length', 'value': 2, 'unit': 'us'},
                        {'name': 'operational', 'value': 0, 'unit': ''}], []],
            'gates': [{'gate': 'cz', 'qubits': [0, 1], 'parameters': [
                {'name': 'gate_error', 'value': 1., 'unit': ''},
                {'name': 'gate_length', 'value': .08, 'unit': 'us'},
            ]}],
        }),
        'coupling_map': CouplingModel(),
    }
    with path.open('wb') as stream:
        pickle.dump(data, stream)


def test_loading_preserves_units_missing_values_and_large_errors(tmp_path):
    file = tmp_path / 'backends/chip.pkl'
    write_snapshot(file)
    snapshot = load_backend_snapshot('backends/chip.pkl', tmp_path)
    assert snapshot['path'] == file
    assert snapshot['qubits'][0]['T1 (us)'] == pytest.approx(200)
    assert snapshot['qubits'][0]['Readout duration (ns)'] == pytest.approx(2000)
    assert snapshot['qubits'][0]['Readout error (%)'] == 1
    assert snapshot['qubits'][0]['Operational flag'] is False
    assert snapshot['qubits'][1]['Operational flag'] is None
    assert snapshot['qubits'][0]['Initialization error (%)'] is None
    assert snapshot['qubits'][0]['Frequency (GHz)'] is None
    assert snapshot['gates'][0]['Error (%)'] == 100
    assert snapshot['gates'][0]['Duration (ns)'] == pytest.approx(80)
    summaries = {r['Metric']: r for r in qubit_calibration_summary(snapshot)}
    assert summaries['Readout error (%)']['N'] == 1
    assert summaries['Initialization error (%)']['N'] == 0
    assert summaries['Initialization error (%)']['Median'] is None
    assert 'Schematic' in backend_layout_description(snapshot)
    with pytest.raises(ValueError, match='Unsupported unit'):
        converted({'T1': {'value': 1, 'unit': 'unknown'}}, 'T1', 'us')


def test_workflow_reuses_paths_and_exports_only_when_requested(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, 'show', lambda: None)
    file = tmp_path / 'chip.pkl'
    write_snapshot(file)
    analysis = BackendAnalysis(tmp_path, figure_dir='figures')
    snapshot = analysis.load_backend('chip.pkl')
    assert analysis.load_backend(file) is snapshot
    assert analysis.load_backend(file, refresh=True) is not snapshot
    shown = analysis.show_backend(file)
    assert shown['path'] == file
    result = analysis.compare_errors({'First': 'chip.pkl', 'Second': file})
    assert result.snapshots['First'] is result.snapshots['Second']
    assert result.output_files == {}
    assert not analysis.figure_dir.exists()
    cached = analysis.load_backend(file, save_figures=True, filename='selected_backend')
    assert cached is shown
    assert (analysis.figure_dir / 'selected_backend.png').stat().st_size > 1000
    assert not (analysis.figure_dir / 'selected_backend.pdf').exists()
    assert not plt.get_fignums()
    assert all(r['Metric'].endswith('error (%)') for r in result.qubit_summaries['First'])
    assert all(r['Metric'] == 'Error (%)' for r in result.gate_summaries['First'])
    saved = analysis.compare_errors({'First': file}, save_figures=True)
    assert {p.name for p in saved.output_files.values()} == {
        'qubit_calibration_comparison.png', 'gate_calibration_comparison.png',
    }
    assert all(p.stat().st_size > 1000 for p in saved.output_files.values())
    assert not plt.get_fignums()
    with pytest.raises(ValueError, match='at least one'):
        analysis.compare_errors({})
    times = analysis.compare_times({'First': file})
    assert times.output_files == {}
    assert times.snapshots['First'] is saved.snapshots['First']
    assert {r['Metric'] for r in times.qubit_summaries['First']} == {
        'T1 (us)', 'T2 (us)', 'Readout duration (ns)',
    }
    assert all(r['Metric'] == 'Duration (ns)' for r in times.gate_summaries['First'])
    error_images = {p: p.read_bytes() for p in saved.output_files.values()}
    saved_times = analysis.compare_times({'First': file}, save_figures=True)
    assert {p.name for p in saved_times.output_files.values()} == {
        'qubit_time_calibration_comparison.png', 'gate_time_calibration_comparison.png',
    }
    assert all(p.stat().st_size > 1000 for p in saved_times.output_files.values())
    assert all(p.read_bytes() == content for p, content in error_images.items())
    assert not plt.get_fignums()
    with pytest.raises(ValueError, match='at least one'):
        analysis.compare_times({})
    for method, stem in ((analysis.compare_errors, 'custom_errors'),
                         (analysis.compare_times, 'custom_times')):
        custom = method({'First': file}, save_figures=True, filename=stem)
        assert {p.name for p in custom.output_files.values()} == {
            f'{stem}_qubit.png', f'{stem}_gate.png',
        }
        assert all(p.stat().st_size > 1000 for p in custom.output_files.values())
        with pytest.raises(ValueError, match='file stem'):
            method({'First': file}, filename='../invalid')
    assert not plt.get_fignums()


def test_missing_gate_errors_and_schematic_layout_render(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    file = tmp_path / 'chip.pkl'
    write_snapshot(file)
    snapshot = load_backend_snapshot(file, tmp_path)
    layout = plot_chip_layout(snapshot, tmp_path)
    assert len(layout.axes) == 3  # Chip plus both calibration colour bars.
    plt.close(layout)
    snapshot['gates'][0]['Error (%)'] = np.nan
    fig = plot_gate_error_comparison({'Missing': snapshot})
    assert any(t.get_text() == 'No error estimates supplied' for t in fig.axes[0].texts)
    plt.close(fig)


def test_time_plots_convert_readout_units_and_preserve_zero_and_missing_times(tmp_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    file = tmp_path / 'chip.pkl'
    write_snapshot(file)
    snapshot = load_backend_snapshot(file, tmp_path)
    qubits = plot_qubit_time_comparison({'Times': snapshot})
    medians = [line.get_xdata()[0] for line in qubits.axes[0].lines if line.get_marker() == 'o']
    assert medians == pytest.approx([200, 2])  # Readout is 2000 ns, displayed as 2 us.
    assert any(t.get_text() == 'N/A' for t in qubits.axes[0].texts)  # T2 is absent.
    assert 'Time (us)' in qubits.axes[0].get_xlabel()
    plt.close(qubits)
    snapshot['gates'].extend([
        {'Gate': 'rz', 'Qubits': (0,), 'Error (%)': None, 'Duration (ns)': 0},
        {'Gate': 'reset', 'Qubits': (0,), 'Error (%)': None, 'Duration (ns)': 2000},
    ])
    gates = plot_gate_time_comparison({'Times': snapshot})
    assert {t.get_text() for t in gates.axes[0].get_yticklabels()} == {'cz', 'rz', 'reset'}
    medians = [line.get_xdata()[0] for line in gates.axes[0].lines if line.get_marker() == 'o']
    assert sorted(medians) == pytest.approx([0, 80, 2000])
    plt.close(gates)
    for record in snapshot['gates']:
        record['Duration (ns)'] = None
    missing = plot_gate_time_comparison({'Missing': snapshot})
    assert any(t.get_text() == 'No duration estimates supplied' for t in missing.axes[0].texts)
    plt.close(missing)
