"""Validate joined placement selections and compilation reuse across panels."""

from types import SimpleNamespace

import pytest

from qgan_v2.analysis import BackendAnalysis
from qgan_v2.analysis.backend_placement import placement_combinations
from qgan_v2.analysis.backend_transpilation import BackendTranspilation


def test_requested_products_and_scalar_parentheses():
    circuits = ('generator-discriminator', 'real-discriminator')
    noisy, columns = placement_combinations((4, 8, 16), circuits, ('noisy'))
    assert len(noisy) == 6 and columns == 3
    assert noisy[:3] == [('noisy', circuits[0], n) for n in (4, 8, 16)]
    real, columns = placement_combinations((4), circuits, ('real-spsa', 'real-psr'))
    assert len(real) == 4 and columns == 2
    assert real[:2] == [('real-spsa', circuits[0], 4), ('real-psr', circuits[0], 4)]
    all_choices, columns = placement_combinations((4, 8, 16), circuits, ('noisy', 'real-spsa', 'real-psr'))
    assert len(set(all_choices)) == 18 and columns == 3
    single, columns = placement_combinations(8, 'real-discriminator', 'noisy')
    assert single == [('noisy', 'real-discriminator', 8)] and columns == 1
    implementations = ('qml-torch', 'runtime-packed-sep', 'runtime-packed-join')
    fake, columns = placement_combinations(4, circuits, 'fake-real', implementations)
    assert len(fake) == 6 and columns == 3
    all_options, columns = placement_combinations(
        (4, 8, 16), circuits, ('noisy', 'real-spsa', 'real-psr', 'fake-real'), implementations)
    assert len(set(all_options)) == 72 and columns == 3


@pytest.mark.parametrize('sizes,circuits,backends', [
    ((), 'real-discriminator', 'noisy'),
    (True, 'real-discriminator', 'noisy'),
    ((4, 4), 'real-discriminator', 'noisy'),
    (32, 'real-discriminator', 'noisy'),
    (4, 'unknown-circuit', 'noisy'),
    (4, 'real-discriminator', 'real'),
])
def test_invalid_selections(sizes, circuits, backends):
    with pytest.raises(ValueError):
        placement_combinations(sizes, circuits, backends)


def test_compilation_is_shared_by_circuits_and_repeated_calls(tmp_path, monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import qgan_v2.analysis.backend_transpilation as reconstruction
    import qgan_v2.analysis.backend_placement as plotting

    config = tmp_path / 'config.yaml'
    config.write_text('test configuration')
    snapshot = {'path': tmp_path / 'backend.pkl'}
    analysis = BackendAnalysis(tmp_path)
    monkeypatch.setattr(analysis, 'load_backend', lambda file: snapshot)
    calls = []
    def compile_snapshot(snapshot, config_file, *, n_qubits, implementation, preset,
                         default_layout=False):
        calls.append((n_qubits, preset))
        return BackendTranspilation([], [SimpleNamespace(), SimpleNamespace()],
                                    [{'template': 'fake'}, {'template': 'real'}], {})
    monkeypatch.setattr(reconstruction, 'reconstruct_backend_transpilation', compile_snapshot)
    monkeypatch.setattr(plotting, 'plot_backend_placements', lambda *args, **kwargs: plt.figure())
    monkeypatch.setattr(plt, 'show', lambda: None)
    args = dict(n_qubits=(4, 8, 16), backend='noisy',
                backend_files={'noisy': 'backend.pkl'}, config_files={'noisy': {4: config}})
    result = analysis.show_transpilation_layouts(**args, save_figures=True, filename='joined')
    assert len(result.placements) == 6
    assert calls == [(n, 'base') for n in (4, 8, 16)]
    for n in (4, 8, 16):
        pair = [p for p in result.placements if p.n_qubits == n]
        assert pair[0].transpilation is pair[1].transpilation
        assert [p.template_index for p in pair] == [0, 1]
    assert set(result.output_files) == {'png'}
    assert all(p.is_file() for p in result.output_files.values())
    assert not (analysis.figure_dir / 'joined.pdf').exists()
    again = analysis.show_transpilation_layouts(**args)
    assert len(again.placements) == 6 and calls == [(n, 'base') for n in (4, 8, 16)]
    assert again.output_files == {}
    variants = analysis.show_transpilation_layouts(**args, preset=('base', 'ang', 'amp'))
    assert len(variants.placements) == 18
    assert calls == [(n, p) for p in ('base', 'ang', 'amp') for n in (4, 8, 16)]
    config.write_text('changed configuration')
    analysis.show_transpilation_layouts(**args)
    assert calls[-3:] == [(n, 'base') for n in (4, 8, 16)] and len(calls) == 12


def test_preset_product_and_unsupported_joined_amplitude():
    circuits = ('generator-discriminator', 'real-discriminator')
    panels, columns = placement_combinations(4, circuits, 'noisy', 'qml-torch', ('base', 'ang', 'amp'))
    assert len(set(panels)) == 6 and columns == 3
    assert [p[-1] for p in panels[:3]] == ['base', 'ang', 'amp']
    panels, columns = placement_combinations(
        (4, 8, 16), circuits, ('noisy', 'fake-real'),
        ('qml-torch', 'runtime-packed-sep'), ('base', 'ang', 'amp'))
    assert len(set(panels)) == 72 and columns == 3
    for preset in ('unknown', (), ('base', 'base')):
        with pytest.raises(ValueError, match='preset'):
            placement_combinations(4, circuits, 'noisy', 'qml-torch', preset)
    panels, _ = placement_combinations(4, circuits, 'noisy', 'runtime-packed-join', ('base', 'amp'))
    assert len(panels) == 4  # Valid selectors; unsupported pairs are skipped by the workflow.
    aliases, _ = placement_combinations(4, ('generator', 'discriminator'), 'noisy', 'runtime-packed-join')
    assert [p[1] for p in aliases] == list(circuits)
    with pytest.raises(ValueError, match='aliases'):
        placement_combinations(4, ('generator', 'generator-discriminator'), 'noisy')


def test_fake_packed_branch_widths_and_logical_labels(monkeypatch):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from pathlib import Path
    import qiskit_ibm_runtime
    from qgan_v2.analysis.backend_placement import placement_layout_job, plot_backend_placements

    def forbid_runtime(*args, **kwargs):
        raise AssertionError('Offline compilation must not open an IBM service/session.')
    monkeypatch.setattr(qiskit_ibm_runtime.QiskitRuntimeService, '__init__', forbid_runtime)
    monkeypatch.setattr(qiskit_ibm_runtime.Session, '__init__', forbid_runtime)
    monkeypatch.setattr(plt, 'show', lambda: None)
    repo = Path(__file__).resolve().parents[2]
    analysis = BackendAnalysis(repo)
    comparison = analysis.show_transpilation_layouts(
        n_qubits=4, backend='fake-real',
        implementation=('qml-torch', 'runtime-packed-sep', 'runtime-packed-join'))
    assert len(comparison.placements) == 6
    assert len(analysis._placement_transpilations) == 3
    for panel in comparison.placements:
        target = panel.snapshot['data']['target']
        assert target.num_qubits == 127
        assert panel.transpilation.provenance['backend_name'] == 'fake_sherbrooke'
        assert panel.transpilation.provenance['backend_file'] is None
        expected_width = 8 if panel.implementation == 'runtime-packed-join' and panel.circuit == 'real-discriminator' else 4
        assert panel.record['logical_qubits'] == expected_width
        assert panel.record['operation_counts'].get('ecr', 0) > 0
        circuit = panel.transpilation.transpiled_circuits[panel.template_index]
        for instruction in circuit.data:
            qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
            assert target.instruction_supported(operation_name=instruction.operation.name, qargs=qubits)
    joined = next(p for p in comparison.placements
                  if p.implementation == 'runtime-packed-join' and p.circuit == 'real-discriminator')
    job = placement_layout_job(joined.transpilation.transpiled_circuits[joined.template_index], joined.record)
    assert job['layout_labels'] == ['real 0', 'fake 0']
    assert [len(g) for g in job['layout_groups']] == [4, 4]
    assert job['logical_label_offsets'] == [0, 4]
    assert len(set(job['group_colors'])) == 2
    fig = plot_backend_placements([joined], 1, title='Joined discriminator')
    labels = {t.get_text() for t in fig.axes[0].texts}
    assert {f'q{i}' for i in range(8)} <= labels
    assert fig.axes[0].get_title() == ''
    assert fig._suptitle.get_text() == 'Joined discriminator'
    plt.close(fig)


def test_each_copy_has_distinct_consistent_colors():
    from qgan_v2.analysis.backend_placement import placement_layout_job, placement_group_color
    fake = [f'fake {i}' for i in range(4)]
    real = [f'real {i}' for i in range(4)]
    assert len({placement_group_color(label) for label in fake + real}) == 8
    record = {'initial_physical_qubits': list(range(16)),
              'layout_groups': [list(range(i, i + 4)) for i in range(0, 16, 4)],
              'layout_labels': ['real 0', 'real 1', 'fake 0', 'fake 1']}
    job = placement_layout_job(SimpleNamespace(), record)
    assert len(set(job['group_colors'])) == 4
    assert job['group_colors'][2] == placement_group_color('fake 0')
    assert job['group_colors'][0] == placement_group_color('real 0')


def test_partial_failures_and_retry_keep_successful_panels(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    import qgan_v2.analysis.backend_transpilation as reconstruction
    import qgan_v2.analysis.backend_placement as plotting

    config = tmp_path / 'config.yaml'
    config.write_text('configuration')
    analysis = BackendAnalysis(tmp_path)
    snapshot = {'path': tmp_path / 'backend.pkl'}
    monkeypatch.setattr(analysis, 'load_backend', lambda file: snapshot)
    fail_angle = True
    calls = []
    def compile_snapshot(snapshot, config_file, *, n_qubits, implementation, preset,
                         default_layout=False):
        calls.append((implementation, preset))
        if preset == 'ang' and fail_angle:
            raise RuntimeError('test compilation failure')
        return BackendTranspilation([], [SimpleNamespace(), SimpleNamespace()], [{}, {}], {})
    monkeypatch.setattr(reconstruction, 'reconstruct_backend_transpilation', compile_snapshot)
    monkeypatch.setattr(plotting, 'plot_backend_placements', lambda *args, **kwargs: plt.figure())
    monkeypatch.setattr(plt, 'show', lambda: None)
    args = dict(n_qubits=4, backend='noisy',
                implementation=('runtime-packed-sep', 'runtime-packed-join'), preset=('base', 'ang', 'amp'),
                backend_files={'noisy': 'backend.pkl'}, config_files={'noisy': {4: config}})
    result = analysis.show_transpilation_layouts(**args)
    assert len(result.placements) == 6 and len(result.skipped_combinations) == 6
    assert len(calls) == 5 and ('runtime-packed-join', 'amp') not in calls
    assert sum('test compilation failure' in p['reason'] for p in result.skipped_combinations) == 4
    fail_angle = False
    result = analysis.show_transpilation_layouts(**args)
    assert len(result.placements) == 10 and len(result.skipped_combinations) == 2
    assert len(calls) == 7  # Failures are retried; successful compilations are reused.
    result = analysis.show_transpilation_layouts(**args, real_circuit_index=99)
    assert len(result.placements) == 5 and len(result.skipped_combinations) == 7
    with pytest.raises(ValueError, match='No placement could be reconstructed'):
        analysis.show_transpilation_layouts(n_qubits=4, backend='noisy',
            preset='amp', implementation='runtime-packed-join', save_figures=True)
    assert not analysis.figure_dir.exists()


def test_supported_pairs_export_with_partial_final_row(tmp_path, monkeypatch):
    from pathlib import Path
    import matplotlib.pyplot as plt
    from qgan_v2.analysis.backend_placement import plot_backend_placements

    monkeypatch.setattr(plt, 'show', lambda: None)
    repo = Path(__file__).resolve().parents[2]
    analysis = BackendAnalysis(repo, figure_dir=tmp_path)
    result = analysis.show_transpilation_layouts(n_qubits=4, backend='fake-real',
        circuit=('generator', 'discriminator'),
        implementation=('runtime-packed-sep', 'runtime-packed-join'),
        preset=('base', 'ang', 'amp'), save_figures=True, filename='supported_pairs')
    assert len(result.placements) == 10 and len(result.skipped_combinations) == 2
    assert all(p['preset'] == 'amp' and p['implementation'] == 'runtime-packed-join'
               for p in result.skipped_combinations)
    assert result.output_files['png'].is_file()
    # Removing a panel must not prevent rendering or leave an empty subplot.
    fig = plot_backend_placements(result.placements[:-1], 2)
    assert len(fig.axes) == 9
    plt.close(fig)
