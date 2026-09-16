from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

from qgan_v2.circuits.factory import get_circuits
from qgan_v2.visualization.hardware_layout import (
    get_hardware_positions,
    get_hardware_edges,
    get_circuit_edges,
    plot_hardware_layout,
)
from qgan_v2.config.loader import load_run_config
from qgan_v2.datasets.images import get_images_dataset, show_images_dataset
from qgan_v2.execution.backend import (
    create_fake_real_backend,
    load_or_create_real_backend_info,
)
from qgan_v2.models.packed_circuits import (
    prepare_angle_disc_job,
    prepare_angle_real_job,
    prepare_direct_disc_job,
    prepare_fixed_real_job,
    prepare_gen_job,
)
from qgan_v2.models.qnn import compose_circuits, get_observables
from qgan_v2.storage.paths import get_run_path, get_training_data_filename
from qgan_v2.training.batch_torch import generate_random_input
from qgan_v2.training.data import load_training_data_file


#- Load run data -#

# Default visual config values
DEFAULT_VISUAL_CONFIG = {
    'draw_circuits': False,
    'draw_hardware_layout': False,
    'draw_probs': True,
    'draw_images': True,
    'draw_results': True,
}


# Get visual config
def get_visual_config(overrides=None):
    return DEFAULT_VISUAL_CONFIG | (overrides or {})


# Load visualization run
def load_visualization_run(config_file):
    config = load_run_config(config_file)
    run_path = get_run_path(config)
    training_data_file = get_training_data_filename(config)
    generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits = get_circuits(config, save_file=config['circuits']['reset'])

    params = load_training_data_file(training_data_file) if training_data_file.exists() else None
    X = get_images_dataset(config) if config['dataset']['type'] == 'classical' else None

    return {
        'config': config,
        'run_path': run_path,
        'training_data_file': training_data_file,
        'generator_circuit': generator_circuit,
        'discriminator_circuit': discriminator_circuit,
        'randomizer_circuit': randomizer_circuit,
        'real_circuits': real_circuits,
        'X': X,
        'params': params,
    }


# Get last generator model parameters
def get_latest_gen_params(params):
    return params.model_g_state['weight'].detach().cpu().numpy().reshape(-1)


# Reconstruct the immutable pre-training vector for legacy checkpoints whose
# stored NumPy snapshot shared memory with the trained Torch parameter tensor.
def get_initial_gen_params(params):
    stored = np.asarray(params.init_gen_params).reshape(-1)
    config = getattr(params, 'config', None)
    try:
        seed = int(config['run']['seed'])
        init_scale = float(config['training']['init_scale'])
    except (KeyError, TypeError, ValueError):
        return stored.copy()
    rng = np.random.RandomState(seed)
    return rng.uniform(-np.pi, np.pi, size=stored.size) * init_scale


# Get generator parameter snapshots
def get_gen_params(params):
    return [
        ('Initial', get_initial_gen_params(params)),
        ('Last', get_latest_gen_params(params)),
        ('Best', params.best_gen_params),
    ]


# Compose randomized generator and sample random input parameters
def get_randomized_generator(run, random_seed=None):
    config = run['config']
    generator_circuit = run['generator_circuit']
    ran_gen_circuit = compose_circuits(run['randomizer_circuit'], generator_circuit)
    n_random_params = ran_gen_circuit.num_parameters - generator_circuit.num_parameters
    if random_seed is None:
        random_params = generate_random_input(
            1,
            n_random_params,
            config['encoding']['randomness'],
            'cpu',
            torch.float64,
        )
    else:
        # Representative result figures must be reproducible and must not alter
        # the notebook's global Torch random state.
        with torch.random.fork_rng():
            torch.manual_seed(int(random_seed))
            random_params = generate_random_input(
                1,
                n_random_params,
                config['encoding']['randomness'],
                'cpu',
                torch.float64,
            )
    return ran_gen_circuit, random_params.reshape(-1).numpy()


# Assign generator and random input parameters to a randomized generator circuit
def assign_gen_params(ran_gen_circuit, gen_params, random_params):
    return ran_gen_circuit.assign_parameters(np.concatenate((gen_params, random_params)))


_GENERATOR_PARAMETER_SETS = ('initial', 'last', 'best')


def _selected_gen_params(params, parameter_sets):
    available = {
        name.lower(): values
        for name, values in get_gen_params(params)
    }
    selected = tuple(str(name).lower() for name in parameter_sets)
    invalid = [name for name in selected if name not in available]
    if invalid:
        allowed = ", ".join(repr(name) for name in _GENERATOR_PARAMETER_SETS)
        raise ValueError(
            f"parameter_sets may contain only {allowed}; received {invalid}"
        )
    return selected, available


def _target_output(run):
    encoding = run['config']['encoding']['type']
    if encoding == 'angle':
        return np.asarray(run['X'][0], dtype=float)

    target = np.asarray(
        Statevector(run['real_circuits'][0]).probabilities(),
        dtype=float,
    )
    if encoding == 'amplitude':
        target = target.reshape(run['X'].shape[1:3])
    return target


def _generated_output(
    run,
    ran_gen_circuit,
    random_params,
    parameter_values,
    estimator=None,
):
    encoding = run['config']['encoding']['type']
    if encoding == 'angle':
        observables = list(get_observables(run['generator_circuit'].num_qubits)[1])
        ordered_values = np.concatenate((parameter_values, random_params))
        pub = (ran_gen_circuit, observables, ordered_values)
        estimator = estimator or StatevectorEstimator()
        return np.asarray(
            estimator.run([pub]).result()[0].data.evs,
            dtype=float,
        ).reshape(run['X'].shape[1:3])

    generated_circuit = assign_gen_params(
        ran_gen_circuit,
        parameter_values,
        random_params,
    )
    generated = np.asarray(
        Statevector(generated_circuit).probabilities(),
        dtype=float,
    )
    if encoding == 'amplitude':
        generated = generated.reshape(run['X'].shape[1:3])
    return generated


def _generated_outputs(config_file, parameter_sets, random_seed):
    run = load_visualization_run(config_file)
    params = run['params']
    if params is None:
        raise FileNotFoundError(f"No checkpoint found: {run['training_data_file']}")

    selected, available = _selected_gen_params(params, parameter_sets)
    ran_gen_circuit, random_params = get_randomized_generator(
        run,
        random_seed=random_seed,
    )
    estimator = (
        StatevectorEstimator()
        if run['config']['encoding']['type'] == 'angle'
        else None
    )
    generated = [
        _generated_output(
            run,
            ran_gen_circuit,
            random_params,
            available[name],
            estimator,
        )
        for name in selected
    ]
    return run, selected, generated, _target_output(run)


def _parameter_evaluations(params):
    evaluations = {
        name: {'score': np.nan, 'epoch': None}
        for name in _GENERATOR_PARAMETER_SETS
    }
    eval_items = sorted(
        (
            (int(epoch), float(value))
            for epoch, value in params.metrics.eval.items()
            if value is not None and np.isfinite(value)
        ),
        key=lambda item: item[0],
    )
    if eval_items:
        best_epoch, best_score = min(eval_items, key=lambda item: (item[1], item[0]))
        last_epoch, last_score = eval_items[-1]
        evaluations['last'] = {'score': last_score, 'epoch': last_epoch}
        evaluations['best'] = {'score': best_score, 'epoch': best_epoch}
    return evaluations


def _plot_output_panels(run, outputs, target, titles, figsize=None):
    panels = [*outputs, target]
    figsize = figsize or (3.25 * len(panels), 3.2)
    fig, axes = plt.subplots(1, len(panels), figsize=figsize)
    axes = np.atleast_1d(axes)
    if target.ndim == 2:
        vmin = float(min(np.min(values) for values in panels))
        vmax = float(max(np.max(values) for values in panels))
        for ax, values, title in zip(axes, panels, titles):
            ax.imshow(values, cmap='gray', vmin=vmin, vmax=vmax)
            ax.axis('off')
            ax.set_title(title)
        return fig, axes

    n_qubits = run['config']['experiment']['n_qubits']
    for index, (ax, values, title) in enumerate(zip(axes, panels, titles)):
        color = 'C1' if index == len(panels) - 1 else 'C0'
        plot_basis_probabilities(ax, values, n_qubits, title=title, color=color)
    return fig, axes


def plot_basis_probabilities(ax, values, n_qubits, *, title=None, color='C0'):
    """Plot every basis state, including states with zero probability."""
    positions = np.arange(len(values))
    if len(values) <= 256:
        ax.bar(positions, values, width=0.8, color=color)
    else:
        # A single step artist keeps large distributions practical to render.
        ax.stairs(values, np.arange(len(values) + 1) - 0.5, fill=True, color=color)
    ax.set_xlim(-0.5, len(values) - 0.5)
    if len(values) <= 32:
        labels = [format(index, f'0{n_qubits}b') for index in positions]
        ax.set_xticks(positions, labels, rotation=90)
    ax.set_xlabel('basis state')
    ax.set_ylabel('probability')
    if title is not None:
        ax.set_title(title)


def _save_output_figure(fig, save_path):
    if save_path is None:
        return
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches='tight')


#- Circuits visualization -#

# Show circuits
def show_circuits(run, visual_config):
    encoding = run['config']['encoding']['type']
    X = run['X']
    real_circuits = run['real_circuits']
    generator_circuit = run['generator_circuit']
    discriminator_circuit = run['discriminator_circuit']

    if encoding == 'angle':
        real_circuit = real_circuits[0]
        real_circuits = [
            real_circuit.assign_parameters(image.flatten())
            for image in X
        ]

    if visual_config['draw_circuits']:
        fig, axes = plt.subplots(len(real_circuits), 1, figsize=(8, max(3, 2 * len(real_circuits))))
        axes = np.atleast_1d(axes)

        for i, (ax, circuit) in enumerate(zip(axes, real_circuits)):
            circuit.decompose(reps=5).draw('mpl', ax=ax)
            ax.set_title(f'Real circuit {i}')

        plt.tight_layout()
        plt.show()

        generator_fig = generator_circuit.draw('mpl')
        generator_fig.suptitle('Generator circuit')
        plt.show()

        discriminator_fig = discriminator_circuit.draw('mpl')
        discriminator_fig.suptitle('Discriminator circuit')
        plt.show()

    if visual_config['draw_images'] and X is not None:
        show_images_dataset(X)

    if visual_config['draw_probs']:
        fig, axes = plt.subplots(len(real_circuits), 1, figsize=(9, max(3, 2 * len(real_circuits))))
        axes = np.atleast_1d(axes)

        for i, (ax, circuit) in enumerate(zip(axes, real_circuits)):
            plot_histogram(Statevector(circuit).probabilities_dict(), ax=ax, bar_labels=False)
            ax.set_title(f'Real data distribution {i}')

        plt.tight_layout()
        plt.show()


#- Hardware layout visualization -#

# Load the configured real target or the local fake target
def get_hardware_info(config):
    if config['experiment']['execution_type'] == 'fake_real':
        backend = create_fake_real_backend()
        return {
            'target': backend.target,
            'configuration': backend.configuration(),
        }
    return load_or_create_real_backend_info(config)


# Create the pass manager used to reproduce the training layout
def get_hardware_pass_manager(config, hardware_info):
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

    transpilation = config['backend']['transpilation']
    return generate_preset_pass_manager(
        optimization_level=transpilation['optimization_level'],
        target=hardware_info['target'],
        layout_method=transpilation['layout_method'],
        routing_method=transpilation['routing_method'],
        seed_transpiler=config['run']['seed'],
    )


# Prepare the packed jobs displayed for the selected packing mode
def get_hardware_layout_jobs(run, pass_manager):
    config = run['config']
    encoding = config['encoding']['type']
    packing = config['implementation']['discriminator_packing']
    batch_size = config['encoding']['batch_size']
    generator = run['generator_circuit']
    discriminator = run['discriminator_circuit']
    randomizer = run['randomizer_circuit']
    real_circuits = run['real_circuits']

    gen_job = prepare_gen_job(
        randomizer,
        generator,
        discriminator,
        batch_size,
        pass_manager,
    )
    gen_job['layout_labels'] = [
        f'fake {copy_index}'
        for copy_index in range(batch_size)
    ]
    if packing == 'joined' and encoding == 'direct_circuit':
        jobs = {'Generator circuit': gen_job}
        joined_job = prepare_direct_disc_job(
            randomizer,
            generator,
            discriminator,
            real_circuits[0],
            pass_manager,
        )
        joined_job['layout_labels'] = ['real 0', 'fake 0']
        jobs['Discriminator circuit'] = joined_job
    elif packing == 'joined' and encoding == 'angle':
        jobs = {'Generator circuit': gen_job}
        joined_job = prepare_angle_disc_job(
            randomizer,
            generator,
            discriminator,
            real_circuits[0],
            batch_size,
            pass_manager,
        )
        half_batch = batch_size // 2
        joined_job['layout_labels'] = [
            *[
                f'real {copy_index}'
                for copy_index in range(half_batch)
            ],
            *[
                f'fake {copy_index}'
                for copy_index in range(half_batch)
            ],
        ]
        jobs['Discriminator circuit'] = joined_job
    else:
        jobs = {'Fake discriminator / generator circuit': gen_job}
        if encoding == 'angle':
            real_job = prepare_angle_real_job(
                real_circuits[0],
                discriminator,
                batch_size,
                pass_manager,
            )
        else:
            real_indexes = np.arange(batch_size) % len(real_circuits)
            real_job = prepare_fixed_real_job(
                real_circuits,
                real_indexes,
                discriminator,
                pass_manager,
            )
        real_job['layout_labels'] = [
            f'real {copy_index}'
            for copy_index in range(batch_size)
        ]
        jobs['Real discriminator circuit'] = real_job

    return jobs


# Draw packed circuit copies using the shared, offline placement plotter.
def draw_hardware_layout(job, hardware_info, title):
    fig = plot_hardware_layout(job, hardware_info, title)
    plt.show()
    return fig


# Show the generator and discriminator hardware layouts
def show_hardware_layout(run, visual_config):
    if not visual_config['draw_hardware_layout']:
        return
    if run['config']['implementation']['name'] != 'runtime_packed':
        print('Hardware layout visualization is only available for runtime_packed.')
        return

    hardware_info = get_hardware_info(run['config'])
    pass_manager = get_hardware_pass_manager(run['config'], hardware_info)
    jobs = get_hardware_layout_jobs(run, pass_manager)

    for name, job in jobs.items():
        draw_hardware_layout(job, hardware_info, f'{name} hardware layout')


#- Training visualization -#

# Show training progress
def show_training_progress(run, visual_config):
    params = run['params']

    if not visual_config['draw_results']:
        return

    if params is None:
        print('Skipping training plots because training_data.pth is missing.')
        return

    gloss = params.metrics.gloss
    dloss = params.metrics.dloss
    eval_metrics = params.metrics.eval

    if not eval_metrics:
        print('Training data has no evaluation metrics yet.')
        return

    gloss_ax = list(gloss.keys())
    gloss_data = list(gloss.values())
    dloss_ax = list(dloss.keys())
    dloss_data = list(dloss.values())
    eval_ax = list(eval_metrics.keys())
    eval_data = list(eval_metrics.values())
    best_eval = np.min(eval_data)

    fig, (loss, eval_plot) = plt.subplots(2, sharex=True, gridspec_kw={'height_ratios': [0.75, 1]}, figsize=(6, 4))
    fig.suptitle('QGAN training stats')
    eval_plot.set_xlabel('Epochs')
    loss.plot(gloss_ax, gloss_data, label='Generator loss', color='#0094f0ff')
    loss.plot(dloss_ax, dloss_data, label='Discriminator loss', color='C3')
    loss.legend()
    loss.set(ylabel='Loss')
    eval_plot.plot(eval_ax, eval_data, label='Evaluation (zero is best)', color='#ffaf01ff')
    eval_plot.set(ylabel='Evaluation')
    eval_plot.legend()
    fig.tight_layout()
    plt.show()

    print(
        'Training complete:', run['training_data_file'],
        '\nBest evaluation:', best_eval,
        'in epoch', eval_ax[int(np.argmin(eval_data))],
        '\nImprovement:', eval_data[0] - best_eval,
    )


# Show generated probability distributions
def show_generated_probabilities(run, visual_config):
    if not visual_config['draw_results'] or not visual_config['draw_probs']:
        return

    encoding = run['config']['encoding']['type']
    params = run['params']
    X = run['X']
    real_circuit = run['real_circuits'][0]

    if params is None:
        return

    gen_param_sets = get_gen_params(params)
    ran_gen_circuit, random_params = get_randomized_generator(run)

    generated_circuits = [
        assign_gen_params(ran_gen_circuit, param_values, random_params)
        for _, param_values in gen_param_sets
    ]

    if encoding == 'angle':
        real_circuit = real_circuit.assign_parameters(X[0].flatten())

    prob_dicts = [
        *[
            Statevector(circuit).probabilities_dict()
            for circuit in generated_circuits
        ],
        Statevector(real_circuit).probabilities_dict(),
    ]
    titles = [f'{name} generated distribution' for name, _ in gen_param_sets] + ['Real distribution']

    fig, axes = plt.subplots(4, 1, sharey=False, sharex=True, figsize=(9, 6))
    for ax, title, probs in zip(axes, titles, prob_dicts):
        ax.set_title(title)
        plot_histogram(probs, ax=ax, bar_labels=False)

    axes[0].set_ylabel('Probabilities')
    fig.tight_layout()
    plt.show()


#- Image results visualization -#

# Show result images
def show_result_images(images, titles):
    fig, axes = plt.subplots(1, len(images), figsize=(8, 2))
    axes = np.atleast_1d(axes)

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


def plot_generated_output(
    config_file,
    *,
    parameter_set='best',
    random_seed=0,
    num_outputs=1,
    save_path=None,
):
    """Create a generated-versus-target figure with one or more outputs.

    ``parameter_set`` may be ``initial``, ``last``, or ``best``.  The returned
    manifest links the visual to its run id and, when recorded, its evaluation
    score. The initial parameters predate the first evaluation measurement.
    When ``random_seed`` is an integer, additional outputs use consecutive
    seeds. When it is ``None``, every output draws a fresh random input.
    Statevector reconstruction is intentionally used for the qualitative image;
    the run's noisy/real execution type remains recorded in the title.
    """

    parameter_set = str(parameter_set).lower()
    if parameter_set not in _GENERATOR_PARAMETER_SETS:
        raise ValueError("parameter_set must be 'initial', 'last', or 'best'")
    if (
        isinstance(num_outputs, (bool, np.bool_))
        or not isinstance(num_outputs, (int, np.integer))
        or num_outputs < 1
    ):
        raise ValueError("num_outputs must be a positive integer")
    num_outputs = int(num_outputs)

    run = load_visualization_run(config_file)
    params = run['params']
    if params is None:
        raise FileNotFoundError(f"No checkpoint found: {run['training_data_file']}")
    _, available = _selected_gen_params(params, (parameter_set,))
    input_seeds = (
        tuple(None for _ in range(num_outputs))
        if random_seed is None
        else tuple(int(random_seed) + index for index in range(num_outputs))
    )
    estimator = (
        StatevectorEstimator()
        if run['config']['encoding']['type'] == 'angle'
        else None
    )
    generated_outputs = []
    for input_seed in input_seeds:
        ran_gen_circuit, random_params = get_randomized_generator(
            run,
            random_seed=input_seed,
        )
        generated_outputs.append(
            _generated_output(
                run,
                ran_gen_circuit,
                random_params,
                available[parameter_set],
                estimator,
            )
        )
    target = _target_output(run)
    generated_titles = (
        (f'Generated ({parameter_set})',)
        if num_outputs == 1
        else tuple(
            f'Generated {index + 1} ({parameter_set})'
            for index in range(num_outputs)
        )
    )
    fig, _ = _plot_output_panels(
        run,
        generated_outputs,
        target,
        (*generated_titles, 'Target'),
        figsize=(max(8, 3.25 * (num_outputs + 1)), 3.2),
    )
    config = run['config']
    evaluation = _parameter_evaluations(run['params'])[parameter_set]
    evaluation_epoch = evaluation['epoch']
    evaluation_score = evaluation['score']
    run_id = config['run']['id']
    if parameter_set == 'initial':
        evaluation_label = "initial parameters; evaluation not recorded"
    else:
        evaluation_label = (
            f"{parameter_set} evaluation={evaluation_score:.5g} "
            f"at epoch={evaluation_epoch}"
        )
    fig.suptitle(f"{run_id}\n{evaluation_label}")
    fig.tight_layout()
    _save_output_figure(fig, save_path)

    return fig, {
        'run_id': run_id,
        'config_file': str(Path(config_file)),
        'execution_type': config['experiment']['execution_type'],
        'parameter_set': parameter_set,
        'random_seed': random_seed,
        'input_seeds': input_seeds,
        'num_outputs': num_outputs,
        'evaluation_score': evaluation_score,
        'evaluation_epoch': evaluation_epoch,
    }


def plot_generated_output_sequence(
    config_file,
    *,
    parameter_sets=('initial', 'last', 'best'),
    random_seed=0,
    save_path=None,
):
    """Plot several generator checkpoints and their target in one row.

    The default layout is ``Initial | Last | Best | Target``. A single random
    input draw is shared by all generator checkpoints so visual differences are
    attributable to the learned parameters rather than input resampling.
    """

    run, parameter_sets, generated_outputs, target = _generated_outputs(
        config_file,
        parameter_sets,
        random_seed,
    )
    titles = [name.capitalize() for name in parameter_sets] + ['Target']
    fig, _ = _plot_output_panels(
        run,
        generated_outputs,
        target,
        titles,
    )
    evaluations = _parameter_evaluations(run['params'])

    config = run['config']
    run_id = config['run']['id']
    fig.suptitle(run_id)
    fig.tight_layout()
    _save_output_figure(fig, save_path)

    return fig, {
        'run_id': run_id,
        'config_file': str(Path(config_file)),
        'execution_type': config['experiment']['execution_type'],
        'parameter_sets': parameter_sets,
        'random_seed': random_seed,
        'evaluations': {
            name: evaluations.get(name, {'score': np.nan, 'epoch': None})
            for name in parameter_sets
        },
    }


def _show_result_images(run, visual_config):
    if not visual_config['draw_results'] or not visual_config['draw_images']:
        return

    params = run['params']
    if params is None:
        print('Skipping result images because training_data.pth is missing.')
        return

    ran_gen_circuit, random_params = get_randomized_generator(run)
    gen_param_sets = get_gen_params(params)
    estimator = (
        StatevectorEstimator()
        if run['config']['encoding']['type'] == 'angle'
        else None
    )
    images = [
        *[
            _generated_output(
                run,
                ran_gen_circuit,
                random_params,
                param_values,
                estimator,
            )
            for _, param_values in gen_param_sets
        ],
        _target_output(run),
    ]
    titles = [name for name, _ in gen_param_sets] + ['Real']

    show_result_images(images, titles)


# Show angle result images
def show_angle_result_images(run, visual_config):
    _show_result_images(run, visual_config)


# Show amplitude result images
def show_amp_result_images(run, visual_config):
    _show_result_images(run, visual_config)


#- Main visualization -#

# Show visualization run
def show_visualization_run(run, visual_config=None):
    visual_config = get_visual_config(visual_config)
    encoding = run['config']['encoding']['type']

    print('Run path:', run['run_path'])
    print('Circuits:', 3 + len(run['real_circuits']), '(generator, discriminator, randomizer, real circuits)')

    if run['params'] is None:
        print('Training data not found:', run['training_data_file'])

    show_circuits(run, visual_config)
    show_hardware_layout(run, visual_config)
    show_training_progress(run, visual_config)
    show_generated_probabilities(run, visual_config)

    if encoding == 'angle':
        show_angle_result_images(run, visual_config)

    elif encoding == 'amplitude':
        show_amp_result_images(run, visual_config)


# Run visualization
def run_visualization(config_file, visual_config=None):
    visual_config = get_visual_config(visual_config)

    run = load_visualization_run(config_file)
    show_visualization_run(run, visual_config)

    return run
