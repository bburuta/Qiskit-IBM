"""Offline reconstruction of training templates using trusted hardware snapshots.

These circuits are new compilations, not recovered historical Runtime inputs.
No Runtime service, estimator, session, or job is created here.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
from typing import Any


IMPLEMENTATION_OPTIONS = {
    'qml-torch': ('qml_torch', 'separate'),
    'runtime-packed-sep': ('runtime_packed', 'separate'),
    'runtime-packed-join': ('runtime_packed', 'joined'),
}
PRESET_OPTIONS = ('base', 'ang', 'amp')


@dataclass
class BackendTranspilation:
    """Reconstructed circuits, per-template records, and compilation provenance."""

    logical_circuits: list[Any]
    transpiled_circuits: list[Any]
    records: list[dict[str, Any]]
    provenance: dict[str, Any]
    output_files: dict[str, Path] = field(default_factory=dict)


def circuit_transpilation_record(logical: Any, transpiled: Any, name: str) -> dict[str, Any]:
    """Describe a static template; counts exclude barriers, mappings omit ancillas.

    Width includes allocated idle wires. Active qubits are wires touched by any
    non-barrier instruction. Depth excludes barriers; two-qubit depth counts
    only instructions acting on exactly two qubits. Neither is elapsed time.
    Initial/final mappings are indexed by the original logical-qubit index;
    the final mapping incorporates routing, unlike ``layout.final_layout``.
    """
    layout = transpiled.layout
    if layout is None:
        raise ValueError('Expected a transpiled circuit with a physical layout.')
    instructions = [item for item in transpiled.data if item.operation.name != 'barrier']
    active = sorted({transpiled.find_bit(q).index for item in instructions for q in item.qubits})
    pairs = Counter(
        tuple(sorted(transpiled.find_bit(q).index for q in item.qubits))
        for item in instructions if item.operation.name == 'cz'
    )
    return {
        'template': name,
        'logical_qubits': logical.num_qubits,
        'logical_depth': logical.depth(filter_function=lambda item: item.operation.name != 'barrier'),
        'physical_width': transpiled.num_qubits,
        'initial_physical_qubits': list(layout.initial_index_layout(filter_ancillas=True)),
        'final_physical_qubits': list(layout.final_index_layout(filter_ancillas=True)),
        'active_physical_qubits': active,
        'depth': transpiled.depth(filter_function=lambda item: item.operation.name != 'barrier'),
        'two_qubit_depth': transpiled.depth(
            filter_function=lambda item: len(item.qubits) == 2 and item.operation.name != 'barrier'),
        'instruction_count': len(instructions),
        'two_qubit_count': sum(len(item.qubits) == 2 for item in instructions),
        'cz_count': sum(pairs.values()),
        'operation_counts': dict(Counter(item.operation.name for item in instructions)),
        'cz_links': [{'qubits': list(pair), 'count': count} for pair, count in sorted(pairs.items())],
    }


def reconstruct_backend_transpilation(
    snapshot: dict[str, Any], config_file: str | Path, *, n_qubits: int | None = None,
    implementation: str | None = None,
    preset: str | None = None,
    default_layout: bool = False,
) -> BackendTranspilation:
    """Reconstruct QML or packed training compilation using a local target.

    Uses current circuit builders and installed Qiskit. Historical software,
    live target changes, and SABRE search behavior can change the output even
    with the saved seed. Gradients and Runtime measurement/mitigation expansion
    are not included. Missing generated image datasets are built in memory.
    An optional qubit-count override changes only the in-memory configuration;
    provenance records both the original count and the requested variant.
    ``default_layout`` selects Qiskit's default layout pipeline for this new
    compilation while preserving the historical configuration on disk.
    """
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qgan_v2.circuits.factory import get_circuits
    from qgan_v2.config.loader import load_run_config
    from qgan_v2.config.defaults import apply_experiment_preset, generate_dataset_id
    from qgan_v2.config.validation import validate_config
    from qgan_v2.models.qnn import get_composed_circuits

    config_file = Path(config_file).resolve()
    config = load_run_config(config_file)
    original_n_qubits = config['experiment']['n_qubits']
    original_preset = config['experiment']['preset']
    original_config = deepcopy(config)
    original_implementation = dict(config['implementation'])
    overrides = {}
    if implementation is not None:
        if implementation not in IMPLEMENTATION_OPTIONS:
            raise ValueError(f'Choose implementation from {tuple(IMPLEMENTATION_OPTIONS)}.')
        name, packing = IMPLEMENTATION_OPTIONS[implementation]
        for field, value in (('name', name), ('discriminator_packing', packing)):
            if config['implementation'][field] != value:
                overrides[f'implementation.{field}'] = value
                config['implementation'][field] = value
        if name == 'runtime_packed' and config['run']['device'] != 'CPU':
            overrides['run.device'] = 'CPU'
            config['run']['device'] = 'CPU'
    if snapshot.get('source_kind') == 'builtin_fake_backend' and config['experiment']['execution_type'] != 'fake_real':
        overrides['experiment.execution_type'] = 'fake_real'
        config['experiment']['execution_type'] = 'fake_real'
    if n_qubits is not None:
        if not isinstance(n_qubits, int) or isinstance(n_qubits, bool) or n_qubits < 1:
            raise ValueError('n_qubits must be a positive integer.')
        config['experiment']['n_qubits'] = n_qubits
        if n_qubits != original_n_qubits:
            overrides['experiment.n_qubits'] = n_qubits
    if preset is not None:
        if preset not in PRESET_OPTIONS:
            raise ValueError(f'Choose preset from {PRESET_OPTIONS}.')
        config['experiment']['preset'] = preset
    if config['experiment']['preset'] != original_preset:
        apply_experiment_preset(config)
        if config['experiment']['preset'] in ('ang', 'amp'):
            config['encoding']['batch_size'] = 4
    elif config['experiment']['n_qubits'] != original_n_qubits and original_preset in ('ang', 'amp'):
        # Keep custom gradient parameters while resizing the encoded input.
        config['dataset']['parameters']['total_pixels'] = (
            config['experiment']['n_qubits'] if original_preset == 'ang'
            else 2 ** config['experiment']['n_qubits'])
    if (config['dataset']['source'] != original_config['dataset']['source']
            or config['dataset']['parameters'] != original_config['dataset']['parameters']):
        config['dataset']['id'] = generate_dataset_id(config)
    if (config['implementation']['name'] == 'runtime_packed'
            and config['implementation']['discriminator_packing'] == 'joined'):
        if config['encoding']['type'] == 'amplitude':
            raise ValueError('preset="amp" does not support implementation="runtime-packed-join". '
                             'Use "qml-torch" or "runtime-packed-sep".')
        if config['encoding']['type'] == 'angle' and config['encoding']['batch_size'] % 2:
            config['encoding']['batch_size'] += 1
    for section in ('experiment', 'dataset', 'encoding'):
        for field, value in config[section].items():
            if value != original_config[section][field]:
                overrides[f'{section}.{field}'] = deepcopy(value)
    validate_config(config)
    if config['implementation']['name'] not in ('qml_torch', 'runtime_packed'):
        raise ValueError('Choose a qml_torch or runtime_packed config.')
    if config['experiment']['execution_type'] not in ('real', 'noisy', 'fake_real'):
        raise ValueError('Choose a real, fake_real, or hardware-mapped noisy run config.')
    if (config['experiment']['execution_type'] == 'noisy'
            and config['backend']['simulator']['noisy_backend_mapping'] != 'hardware'):
        raise ValueError('A physical-target reconstruction requires noisy_backend_mapping=hardware.')
    target = snapshot['data'].get('target')
    if target is None:
        raise ValueError('The backend snapshot does not contain a transpiler target.')
    if config['experiment']['n_qubits'] > target.num_qubits:
        raise ValueError('The requested circuit is wider than the saved backend target.')
    bundle = get_circuits(config, save_file=False, save_dataset=False)
    settings = dict(config['backend']['transpilation'], seed_transpiler=config['run']['seed'])
    if default_layout:
        settings['layout_method'] = None
        overrides['backend.transpilation.layout_method'] = None
    pm = generate_preset_pass_manager(target=target, **settings)
    jobs = None
    if config['implementation']['name'] == 'qml_torch':
        generator, discriminator, randomizer, real = bundle
        _, fake, real_disc = get_composed_circuits(generator, discriminator, randomizer, real)
        logical = [fake, *real_disc]  # Same order as transpile_train_circuits.
        names = ['Generator–discriminator', *[f'Real-data–discriminator {i}' for i in range(len(real_disc))]]
        compiled = pm.run(logical, num_processes=1)
    else:
        logical, compiled, names, jobs = _reconstruct_packed_templates(config, bundle, pm, target.num_qubits)
    records = [circuit_transpilation_record(before, after, name)
               for before, after, name in zip(logical, compiled, names)]
    for index, record in enumerate(records):
        width = config['experiment']['n_qubits']
        groups = ([record['initial_physical_qubits']] if jobs is None else
                  [record['initial_physical_qubits'][start:start + width]
                   for start in range(0, record['logical_qubits'], width)])
        record['layout_groups'] = groups
        record['training_role'] = 'generator' if index == 0 else 'discriminator'
        record['layout_labels'] = ([f'{"fake" if index == 0 else "real"} 0']
                                   if jobs is None else jobs[index]['layout_labels'])
        record['branch_sequences'] = [
            ['randomizer', 'generator', 'discriminator'] if label.startswith('fake')
            else ['real_encoding', 'discriminator']
            for label in record['layout_labels']]
        if jobs is not None and 'real_sample_indices' in jobs[index]:
            record['real_sample_indices'] = jobs[index]['real_sample_indices']
    packages = {}
    for package in ('qiskit', 'qiskit-aer', 'qiskit-ibm-runtime', 'qiskit-machine-learning'):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    precision = config['backend']['precision']
    provenance = {
        'kind': 'offline_reconstruction',
        'historical_job_circuits_verified': False,
        'reconstructed_at': datetime.now(timezone.utc).isoformat(),
        'backend_file': str(snapshot['path']) if snapshot['path'] is not None else None,
        'backend_source_kind': snapshot.get('source_kind', 'saved_snapshot'),
        'config_file': str(config_file),
        'backend_name': snapshot['data'].get('name', snapshot['config'].get('backend_name')),
        'property_update_date': snapshot['properties'].get('last_update_date').isoformat()
            if snapshot['properties'].get('last_update_date') is not None else None,
        'run_id': config['run']['id'],
        'execution_type': config['experiment']['execution_type'],
        'original_n_qubits': original_n_qubits,
        'requested_n_qubits': config['experiment']['n_qubits'],
        'qubit_count_overridden': config['experiment']['n_qubits'] != original_n_qubits,
        'original_preset': original_preset,
        'requested_preset': config['experiment']['preset'],
        'preset_overridden': config['experiment']['preset'] != original_preset,
        'requested_batch_size': config['encoding']['batch_size'],
        'implementation': config['implementation'],
        'original_implementation': original_implementation,
        'configuration_overrides': overrides,
        'settings': settings,
        'software_versions': packages,
        'requested_precision': precision,
        'nominal_simulator_shots': int(1 / precision ** 2)
            if config['experiment']['execution_type'] == 'noisy' else None,
        'effective_hardware_shots': None,
        'actual_session_duration_seconds': None,
        'scope': 'Unbound training templates before gradient and Runtime measurement/mitigation expansion.',
        'config': config,
    }
    return BackendTranspilation(logical, compiled, records, provenance)


def _reconstruct_packed_templates(config, bundle, pm, target_width):
    """Use the same logical builders and transpiler as packed training jobs."""
    from qgan_v2.models.packed_circuits import (
        create_fake_circuit, create_fixed_real_circuit, create_angle_real_circuit,
        create_direct_disc_circuit, create_angle_disc_circuit, transpile_packed_circuit,
    )
    generator, discriminator, randomizer, real = bundle
    width = config['experiment']['n_qubits']
    batch = config['encoding']['batch_size']
    encoding = config['encoding']['type']
    joined = config['implementation']['discriminator_packing'] == 'joined'
    logical, compiled, names, jobs = [], [], [], []

    def add(circuit, inputs, copies, name, labels):
        if circuit.num_qubits > target_width:
            raise ValueError(f'{name} needs {circuit.num_qubits} physical qubits; target has {target_width}.')
        job = transpile_packed_circuit(circuit, inputs, width, copies, pm)
        job['layout_labels'] = labels
        logical.append(circuit)
        compiled.append(job['circuit'])
        names.append(name)
        jobs.append(job)

    circuit, inputs = create_fake_circuit(randomizer, generator, discriminator, batch)
    generator_name = 'Generator (fake → discriminator)' if joined else 'Generator–discriminator'
    add(circuit, inputs, batch, generator_name, [f'fake {i}' for i in range(batch)])
    for real_index, real_circuit in enumerate(real):
        if joined and encoding == 'direct_circuit':
            circuit, inputs = create_direct_disc_circuit(randomizer, generator, discriminator, real_circuit)
            copies, labels = 2, ['real 0', 'fake 0']
        elif joined and encoding == 'angle':
            circuit, inputs = create_angle_disc_circuit(randomizer, generator, discriminator, real_circuit, batch)
            copies = batch
            labels = [f'real {i}' for i in range(batch // 2)] + [f'fake {i}' for i in range(batch // 2)]
        elif encoding == 'angle':
            circuit, inputs = create_angle_real_circuit(real_circuit, discriminator, batch)
            copies, labels = batch, [f'real {i}' for i in range(batch)]
        else:
            real_indices = ([(real_index + i) % len(real) for i in range(batch)]
                            if encoding == 'amplitude' else [real_index] * batch)
            circuit, inputs = create_fixed_real_circuit(real, real_indices, discriminator)
            copies, labels = batch, [f'real {i}' for i in range(batch)]
        add(circuit, inputs, copies,
            f'Discriminator (real → discriminator + fake → discriminator) {real_index}'
            if joined else f'Real-data–discriminator {real_index}', labels)
        if encoding == 'amplitude':
            # Representative deterministic batch, not recovered training sample order.
            jobs[-1]['real_sample_indices'] = real_indices
    return logical, compiled, names, jobs


def transpilation_summary(result: BackendTranspilation) -> list[dict[str, Any]]:
    """Compact notebook rows; full operation counts and CZ links remain in records."""
    return [{
        'Template': r['template'], 'Logical depth': r['logical_depth'],
        'Depth': r['depth'], '2-qubit depth': r['two_qubit_depth'], 'CZ count': r['cz_count'],
        'Active physical qubits': r['active_physical_qubits'],
        'Initial placement (logical order)': r['initial_physical_qubits'],
        'Final placement (logical order)': r['final_physical_qubits'],
    } for r in result.records]


def save_backend_transpilation(result: BackendTranspilation, directory: str | Path) -> dict[str, Path]:
    """Export a JSON report and QPY circuits in record order, labelled reconstruction."""
    from qiskit import qpy

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = {'report': directory / 'reconstructed_transpilation.json',
             'logical_circuits': directory / 'logical_templates.qpy',
             'transpiled_circuits': directory / 'reconstructed_templates.qpy'}
    files['report'].write_text(json.dumps(
        {'provenance': result.provenance, 'circuits': result.records}, indent=2) + '\n')
    for key, circuits in (('logical_circuits', result.logical_circuits),
                          ('transpiled_circuits', result.transpiled_circuits)):
        with files[key].open('wb') as stream:
            qpy.dump(circuits, stream)
    result.output_files = files
    return files
