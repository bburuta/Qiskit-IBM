import matplotlib.pyplot as plt
import numpy as np
import torch
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram

from qgan_v4.circuits.factory import get_circuits
from qgan_v4.config.loader import load_run_config
from qgan_v4.datasets.images import get_images_dataset, show_images_dataset
from qgan_v4.execution.backend import (
    create_fake_real_backend,
    load_or_create_real_backend_info,
)
from qgan_v4.models.packed_circuits import (
    prepare_angle_disc_job,
    prepare_angle_real_job,
    prepare_direct_disc_job,
    prepare_fixed_real_job,
    prepare_gen_job,
)
from qgan_v4.models.qnn import compose_circuits, get_observables
from qgan_v4.storage.paths import get_run_path, get_training_data_filename
from qgan_v4.training.batch_torch import generate_random_input
from qgan_v4.training.data import load_training_data_file


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


# Get generator parameter snapshots
def get_gen_params(params):
    return [
        ('Initial', params.init_gen_params),
        ('Last', get_latest_gen_params(params)),
        ('Best', params.best_gen_params),
    ]


# Compose randomized generator and sample random input parameters
def get_randomized_generator(run):
    config = run['config']
    generator_circuit = run['generator_circuit']
    ran_gen_circuit = compose_circuits(run['randomizer_circuit'], generator_circuit)
    n_random_params = ran_gen_circuit.num_parameters - generator_circuit.num_parameters
    random_params = generate_random_input(1, n_random_params, config['encoding']['randomness'], 'cpu', torch.float64)
    return ran_gen_circuit, random_params.reshape(-1).numpy()


# Assign generator and random input parameters to a randomized generator circuit
def assign_gen_params(ran_gen_circuit, gen_params, random_params):
    return ran_gen_circuit.assign_parameters(np.concatenate((gen_params, random_params)))


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
        tuple(sorted(qubit._index for qubit in instruction.qubits))
        for instruction in circuit.data
        if len(instruction.qubits) == 2
    })


# Draw packed circuit copies on the physical hardware layout
def draw_hardware_layout(job, hardware_info, title):
    positions = get_hardware_positions(hardware_info)
    hardware_edges = get_hardware_edges(hardware_info)
    circuit_edges = get_circuit_edges(job['circuit'])
    layout_groups = job['layout_groups']
    layout_labels = job.get(
        'layout_labels',
        [f'copy {copy_index}' for copy_index in range(len(layout_groups))],
    )
    selected_qubits = set().union(*(set(group) for group in layout_groups))
    colors = plt.get_cmap('tab20')(
        np.linspace(0, 1, max(len(layout_groups), 1))
    )

    fig, ax = plt.subplots(figsize=(14, 7))

    for a, b in hardware_edges:
        ax.plot(
            [positions[a][0], positions[b][0]],
            [positions[a][1], positions[b][1]],
            color='0.86',
            linewidth=0.7,
            zorder=1,
        )

    idle_qubits = [qubit for qubit in positions if qubit not in selected_qubits]
    ax.scatter(
        [positions[qubit][0] for qubit in idle_qubits],
        [positions[qubit][1] for qubit in idle_qubits],
        s=28,
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
            linewidth=2,
            alpha=0.45,
            zorder=3,
        )

    for copy_index, group in enumerate(layout_groups):
        ax.scatter(
            [positions[qubit][0] for qubit in group],
            [positions[qubit][1] for qubit in group],
            s=120,
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
                f'{qubit}\nq{local_index}',
                ha='center',
                va='center',
                fontsize=7,
                zorder=5,
            )

    ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    fig.tight_layout()
    plt.show()


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


# Show angle result images
def show_angle_result_images(run, visual_config):
    if not visual_config['draw_results'] or not visual_config['draw_images']:
        return

    params = run['params']
    X = run['X']
    generator_circuit = run['generator_circuit']

    if params is None:
        print('Skipping result images because training_data.pth is missing.')
        return

    ran_gen_circuit, random_params = get_randomized_generator(run)

    observables = list(get_observables(generator_circuit.num_qubits)[1])

    estimator = StatevectorEstimator()
    dims = X.shape[1:3]

    def generate_sample(gen_params):
        ordered_values = np.concatenate((gen_params, random_params))
        pub = (ran_gen_circuit, observables, ordered_values)
        result = estimator.run([pub]).result()[0]
        return np.asarray(result.data.evs, dtype=float).reshape(dims)

    gen_param_sets = get_gen_params(params)
    images = [
        *[
            generate_sample(param_values)
            for _, param_values in gen_param_sets
        ],
        X[0],
    ]
    titles = [name for name, _ in gen_param_sets] + ['Real']

    show_result_images(images, titles)


# Show amplitude result images
def show_amp_result_images(run, visual_config):
    if not visual_config['draw_results'] or not visual_config['draw_images']:
        return

    params = run['params']
    X = run['X']
    real_circuit = run['real_circuits'][0]

    if params is None:
        print('Skipping result images because training_data.pth is missing.')
        return

    ran_gen_circuit, random_params = get_randomized_generator(run)
    dims = X.shape[1:3]

    gen_param_sets = get_gen_params(params)
    images = [
        *[
            Statevector(assign_gen_params(ran_gen_circuit, param_values, random_params)).probabilities().reshape(dims)
            for _, param_values in gen_param_sets
        ],
        np.asarray(Statevector(real_circuit).probabilities()).reshape(dims),
    ]
    titles = [name for name, _ in gen_param_sets] + ['Real']

    show_result_images(images, titles)


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
