"""Check physical mappings, routed counts, and offline report round-trips."""

import json

import pytest

qiskit = pytest.importorskip('qiskit')
from qiskit import QuantumCircuit, qpy, transpile
from qiskit.quantum_info import Operator
from qgan_v2.analysis.backend_transpilation import (
    BackendTranspilation, circuit_transpilation_record, save_backend_transpilation,
)


def test_routed_template_mappings_counts_and_roundtrip(tmp_path):
    logical = QuantumCircuit(3)
    logical.h(0)
    logical.cx(0, 2)
    logical.cx(0, 1)
    logical.barrier()
    compiled = transpile(
        logical, basis_gates=['rz', 'sx', 'x', 'cz'],
        coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1]],
        initial_layout=[0, 1, 2], routing_method='basic', optimization_level=0,
    )
    record = circuit_transpilation_record(logical, compiled, 'Routed test')
    assert record['initial_physical_qubits'] == [0, 1, 2]
    assert record['final_physical_qubits'] == compiled.layout.final_index_layout()
    assert record['final_physical_qubits'] != record['initial_physical_qubits']
    assert record['cz_count'] == compiled.count_ops()['cz'] > 2
    assert sum(link['count'] for link in record['cz_links']) == record['cz_count']
    assert 'barrier' not in record['operation_counts']
    assert record['depth'] >= record['two_qubit_depth'] > 0
    assert record['active_physical_qubits'] == [0, 1, 2]
    assert Operator.from_circuit(compiled).equiv(Operator(logical))
    result = BackendTranspilation([logical], [compiled], [record],
                                  {'kind': 'offline_reconstruction'})
    files = save_backend_transpilation(result, tmp_path)
    report = json.loads(files['report'].read_text())
    assert report['circuits'][0] == record
    with files['transpiled_circuits'].open('rb') as stream:
        restored = qpy.load(stream)[0]
    assert circuit_transpilation_record(logical, restored, 'Routed test') == record


def test_idle_wires_are_not_active_and_missing_layout_is_rejected():
    logical = QuantumCircuit(1)
    logical.x(0)
    compiled = transpile(logical, basis_gates=['x'],
                         coupling_map=[[0, 1], [1, 0], [1, 2], [2, 1]], initial_layout=[2])
    record = circuit_transpilation_record(logical, compiled, 'One active wire')
    assert record['physical_width'] == 3
    assert record['active_physical_qubits'] == [2]
    assert record['initial_physical_qubits'] == record['final_physical_qubits'] == [2]
    assert record['cz_count'] == record['two_qubit_depth'] == 0
    with pytest.raises(ValueError, match='physical layout'):
        circuit_transpilation_record(logical, logical, 'No layout')


@pytest.mark.parametrize('backend', ['noisy', 'fake-real'])
@pytest.mark.parametrize('preset,implementation', [
    ('ang', 'qml-torch'), ('ang', 'runtime-packed-sep'), ('ang', 'runtime-packed-join'),
    ('amp', 'qml-torch'), ('amp', 'runtime-packed-sep'),
])
def test_presets_use_training_encoders_and_native_target(backend, preset, implementation, tmp_path, monkeypatch):
    from pathlib import Path
    import numpy as np
    import qiskit_ibm_runtime
    from qiskit.quantum_info import Statevector
    from qgan_v2.analysis import BackendAnalysis
    from qgan_v2.analysis.backend_placement import (
        BACKEND_PLACEMENT_CONFIGS, BACKEND_PLACEMENT_FILES, fake_backend_snapshot,
    )
    from qgan_v2.analysis.backend_transpilation import reconstruct_backend_transpilation
    from qgan_v2.datasets import images

    def forbid_write_or_runtime(*args, **kwargs):
        raise AssertionError('Preset analysis must not write datasets or open IBM services/sessions.')
    monkeypatch.setattr(qiskit_ibm_runtime.QiskitRuntimeService, '__init__', forbid_write_or_runtime)
    monkeypatch.setattr(qiskit_ibm_runtime.Session, '__init__', forbid_write_or_runtime)
    monkeypatch.setattr(images, 'get_prepared_dataset_filename', lambda config: tmp_path / 'missing.npz')
    monkeypatch.setattr(images, 'create_images_dataset_file', forbid_write_or_runtime)
    repo = Path(__file__).resolve().parents[2]
    config = repo / BACKEND_PLACEMENT_CONFIGS[backend][4]
    original = config.read_bytes()
    snapshot = (fake_backend_snapshot() if backend == 'fake-real'
                else BackendAnalysis(repo).load_backend(BACKEND_PLACEMENT_FILES[backend]))
    result = reconstruct_backend_transpilation(snapshot, config, preset=preset, implementation=implementation)
    assert config.read_bytes() == original and not (tmp_path / 'missing.npz').exists()
    provenance = result.provenance
    assert provenance['original_preset'] == 'base'
    assert provenance['requested_preset'] == preset and provenance['preset_overridden']
    assert provenance['configuration_overrides']['experiment.preset'] == preset
    assert provenance['config']['dataset']['parameters']['total_pixels'] == (4 if preset == 'ang' else 16)
    assert provenance['config']['dataset']['id'] == f'generated_gradients-total_pixels{4 if preset == "ang" else 16}'
    batch = 4
    assert provenance['config']['encoding']['batch_size'] == batch
    packed = implementation != 'qml-torch'
    assert result.records[1]['logical_qubits'] == (4 * batch if packed else 4)
    assert len(result.records[1]['layout_groups']) == (4 if packed else 1)
    assert all(len(group) == 4 for group in result.records[1]['layout_groups'])
    assert len(result.records) == (2 if preset == 'ang' else 8)
    if preset == 'ang':
        input_prefix = 'θ_r' if implementation == 'qml-torch' else 'real_'
        real_copies = 2 if implementation == 'runtime-packed-join' else 4 if packed else 1
        assert sum(p.name.startswith(input_prefix) for p in result.logical_circuits[1].parameters) == 4 * real_copies
        if implementation == 'runtime-packed-join':
            assert provenance['configuration_overrides']['encoding.batch_size'] == 4
            assert result.records[1]['layout_labels'] == ['real 0', 'real 1', 'fake 0', 'fake 1']
            generator, discriminator = result.records
            assert generator['training_role'] == 'generator'
            assert discriminator['training_role'] == 'discriminator'
            assert generator['branch_sequences'] == [['randomizer', 'generator', 'discriminator']] * 4
            assert discriminator['branch_sequences'] == (
                [['real_encoding', 'discriminator']] * 2 + [['randomizer', 'generator', 'discriminator']] * 2)
            # The independently built generator has G on all four groups;
            # the joined discriminator has G only on its two fake groups.
            for template, gen_groups in ((result.logical_circuits[0], {0, 1, 2, 3}),
                                         (result.logical_circuits[1], {2, 3})):
                observed = {template.find_bit(q).index // 4 for item in template.data
                            if any(p.name.startswith('θ_g') for p in item.operation.params
                                   if hasattr(p, 'name')) for q in item.qubits}
                assert observed == gen_groups
    else:
        # The first amplitude template encodes the linear gradient, not the base dataset.
        preparation = QuantumCircuit(4)
        preparation.append(result.logical_circuits[1].data[0].operation, range(4))
        pixels = images.create_gradients(16)[0].flatten()
        np.testing.assert_allclose(Statevector(preparation).data, np.sqrt(pixels / pixels.sum()), atol=1e-12)
        if packed:
            assert result.records[1]['real_sample_indices'] == [0, 1, 2, 3]
    target = snapshot['data']['target']
    for compiled in result.transpiled_circuits:
        for item in compiled.data:
            qargs = tuple(compiled.find_bit(q).index for q in item.qubits)
            assert target.instruction_supported(operation_name=item.operation.name, qargs=qargs)


@pytest.mark.parametrize('preset,n_qubits', [('ang', 8), ('amp', 8), ('amp', 16)])
def test_resized_presets_have_matching_dataset_dimensions(preset, n_qubits, monkeypatch):
    from pathlib import Path
    from qgan_v2.analysis.backend_placement import BACKEND_PLACEMENT_CONFIGS, fake_backend_snapshot
    from qgan_v2.analysis.backend_transpilation import reconstruct_backend_transpilation
    import qgan_v2.circuits.factory as factory

    class ConfigChecked(Exception):
        pass
    def inspect_config(config, **kwargs):
        expected = n_qubits if preset == 'ang' else 2 ** n_qubits
        assert config['experiment']['n_qubits'] == n_qubits
        assert config['dataset']['parameters']['total_pixels'] == expected
        assert config['dataset']['id'] == f'generated_gradients-total_pixels{expected}'
        assert kwargs == {'save_file': False, 'save_dataset': False}
        raise ConfigChecked
    monkeypatch.setattr(factory, 'get_circuits', inspect_config)
    repo = Path(__file__).resolve().parents[2]
    with pytest.raises(ConfigChecked):
        reconstruct_backend_transpilation(fake_backend_snapshot(),
            repo / BACKEND_PLACEMENT_CONFIGS['fake-real'][4], preset=preset, n_qubits=n_qubits)
