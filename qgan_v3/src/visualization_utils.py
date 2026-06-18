import numpy as np
import torch
from qiskit.quantum_info import Statevector

from batch_torch import generate_torch_random_input
from circuits import get_circuits
from config_manager.config_manager import get_run_path
from datasets.images_dataset import get_images_dataset, show_images_dataset
from main import load_run_config, run_train
from qnn import compose_circuits, get_observables
from train import get_training_data_filename



#- Load run data -#

# Load visualization run
def load_visualization_run(config_file):
    config = load_run_config(config_file)
    run_path = get_run_path(config)
    training_data_file = get_training_data_filename(run_path)
    generator_circuit, discriminator_circuit, randomizer_circuit, real_circuits = get_circuits(config, save_file=True)

    params = None
    if training_data_file.exists():
        params = torch.load(training_data_file, weights_only=False, map_location='cpu')

    X = None
    if config['dataset']['type'] == 'classical':
        X = get_images_dataset(config)

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



#- Circuits visualization -#

# Show circuits
def show_circuits(run, visual_config):
    import matplotlib.pyplot as plt
    from IPython.display import display
    from qiskit.visualization import plot_histogram

    config = run['config']
    encoding = config['encoding']['type']
    params = run['params']
    X = run['X']
    real_circuits = run['real_circuits']
    generator_circuit = run['generator_circuit']
    discriminator_circuit = run['discriminator_circuit']

    if encoding == 'angle' and X is not None:
        real_circuit = real_circuits[0]
        real_circuits = []
        for image in X:
            real_circuits.append(real_circuit.assign_parameters(image.flatten()))

    if visual_config['draw_circuits']:
        fig, axes = plt.subplots(len(real_circuits), 1, figsize=(8, max(3, 2 * len(real_circuits))))
        if len(real_circuits) == 1:
            axes = [axes]

        for i, circuit in enumerate(real_circuits):
            circuit.decompose(reps=5).draw('mpl', ax=axes[i])
            axes[i].set_title(f'Real circuit {i}')

        plt.tight_layout()
        plt.show()

        display(generator_circuit.draw('mpl'))
        display(discriminator_circuit.draw('mpl'))

    if visual_config.get('draw_images', False) and X is not None:
        show_images_dataset(X)

    if visual_config['draw_probs']:
        fig, axes = plt.subplots(len(real_circuits), 1, figsize=(9, max(3, 2 * len(real_circuits))))
        if len(real_circuits) == 1:
            axes = [axes]

        for i, circuit in enumerate(real_circuits):
            plot_histogram(Statevector(circuit).probabilities_dict(), ax=axes[i], bar_labels=False)
            axes[i].set_title(f'Real data distribution {i}')

        plt.tight_layout()
        plt.show()

        if params is not None:
            init_gen_circuit = generator_circuit.assign_parameters(params['init_gen_params'])
            fig, ax = plt.subplots(1, 1, figsize=(9, 3))
            ax.set_title('Initial generated distribution')
            plot_histogram(Statevector(init_gen_circuit).probabilities_dict(), ax=ax)
            plt.show()

    if params is not None:
        print(params['init_disc_params'])



#- Training visualization -#

# Show training progress
def show_training_progress(run, visual_config):
    import matplotlib.pyplot as plt

    params = run['params']

    if not visual_config['draw_results']:
        return

    if params is None:
        print('Skipping training plots because training_data.pth is missing.')
        return

    gloss = params['metrics'].get('gloss', {})
    dloss = params['metrics'].get('dloss', {})
    eval_metrics = params['metrics'].get('eval', {})

    if not eval_metrics:
        print('Training data has no evaluation metrics yet.')
        return

    gloss_ax = list(gloss.keys())
    gloss_data = list(gloss.values())
    dloss_ax = list(dloss.keys())
    dloss_data = list(dloss.values())
    eval_ax = list(eval_metrics.keys())
    eval_data = list(eval_metrics.values())

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
        '\nBest evaluation:', np.min(eval_data),
        'in epoch', eval_ax[int(np.argmin(eval_data))],
        '\nImprovement:', eval_data[0] - np.min(eval_data),
    )


# Show generated probability distributions
def show_generated_probabilities(run, visual_config):
    import matplotlib.pyplot as plt
    from qiskit.visualization import plot_histogram

    if not visual_config['draw_results'] or not visual_config['draw_probs']:
        return

    config = run['config']
    encoding = config['encoding']['type']
    params = run['params']
    X = run['X']
    real_circuit = run['real_circuits'][0]
    generator_circuit = run['generator_circuit']
    randomizer_circuit = run['randomizer_circuit']

    if params is None:
        return

    ran_gen_circuit = compose_circuits(randomizer_circuit, generator_circuit)
    n_random_params = ran_gen_circuit.num_parameters - generator_circuit.num_parameters
    random_params = generate_torch_random_input(1, n_random_params, config['encoding']['randomness'], 'cpu', torch.float64).reshape(-1).numpy()

    init_circuit = ran_gen_circuit.assign_parameters(np.concatenate((params['init_gen_params'], random_params)))
    last_circuit = ran_gen_circuit.assign_parameters(np.concatenate((params['gen_params'].detach().numpy(), random_params)))
    best_circuit = ran_gen_circuit.assign_parameters(np.concatenate((params['best_gen_params'], random_params)))

    if encoding == 'angle':
        real_circuit = real_circuit.assign_parameters(X[0].flatten())

    prob_dicts = [
        Statevector(init_circuit).probabilities_dict(),
        Statevector(last_circuit).probabilities_dict(),
        Statevector(best_circuit).probabilities_dict(),
        Statevector(real_circuit).probabilities_dict(),
    ]
    titles = ['Initial distribution', 'Last generated distribution', 'Best generated distribution', 'Real distribution']

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
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(images), figsize=(8, 2))
    if len(images) == 1:
        axes = [axes]

    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap='gray')
        ax.axis('off')
        ax.set_title(title)

    fig.tight_layout()
    plt.show()


# Show angle result images
def show_angle_result_images(run, visual_config):
    from qiskit.primitives import StatevectorEstimator

    if not visual_config['draw_results'] or not visual_config['draw_images']:
        return

    config = run['config']
    params = run['params']
    X = run['X']
    generator_circuit = run['generator_circuit']
    randomizer_circuit = run['randomizer_circuit']

    if params is None:
        print('Skipping result images because training_data.pth is missing.')
        return

    ran_gen_circuit = compose_circuits(randomizer_circuit, generator_circuit)
    n_random_params = ran_gen_circuit.num_parameters - generator_circuit.num_parameters
    random_params = generate_torch_random_input(1, n_random_params, config['encoding']['randomness'], 'cpu', torch.float64).reshape(-1).numpy()

    _, observables = get_observables(generator_circuit.num_qubits)
    observables = [observables[i] for i in range(len(observables))]

    estimator = StatevectorEstimator()
    dims = X.shape[1:3]

    def generate_sample(gen_params):
        if torch.is_tensor(gen_params):
            gen_params = gen_params.detach().numpy()

        ordered_values = np.concatenate((gen_params, random_params))
        pub = (ran_gen_circuit, observables, ordered_values)
        result = estimator.run([pub]).result()[0]
        return np.asarray(result.data.evs, dtype=float).reshape(dims)

    images = [
        generate_sample(params['init_gen_params']),
        generate_sample(params['gen_params']),
        generate_sample(params['best_gen_params']),
        X[0],
    ]
    titles = ['Initial', 'Last', 'Best', 'Real']

    show_result_images(images, titles)


# Show amplitude result images
def show_amp_result_images(run, visual_config):
    if not visual_config['draw_results'] or not visual_config['draw_images']:
        return

    config = run['config']
    params = run['params']
    X = run['X']
    real_circuit = run['real_circuits'][0]
    generator_circuit = run['generator_circuit']
    randomizer_circuit = run['randomizer_circuit']

    if params is None:
        print('Skipping result images because training_data.pth is missing.')
        return

    ran_gen_circuit = compose_circuits(randomizer_circuit, generator_circuit)
    n_random_params = ran_gen_circuit.num_parameters - generator_circuit.num_parameters
    random_params = generate_torch_random_input(1, n_random_params, config['encoding']['randomness'], 'cpu', torch.float64).reshape(-1).numpy()
    dims = X.shape[1:3]

    images = [
        Statevector(ran_gen_circuit.assign_parameters(np.concatenate((params['init_gen_params'], random_params)))).probabilities().reshape(dims),
        Statevector(ran_gen_circuit.assign_parameters(np.concatenate((params['gen_params'].detach().numpy(), random_params)))).probabilities().reshape(dims),
        Statevector(ran_gen_circuit.assign_parameters(np.concatenate((params['best_gen_params'], random_params)))).probabilities().reshape(dims),
        np.asarray(Statevector(real_circuit).probabilities()).reshape(dims),
    ]
    titles = ['Initial', 'Last', 'Best', 'Real']

    show_result_images(images, titles)



#- Main visualization -#

# Show visualization run
def show_visualization_run(run, visual_config):
    encoding = run['config']['encoding']['type']

    print('Run path:', run['run_path'])
    print('Circuits:', 3 + len(run['real_circuits']), '(generator, discriminator, randomizer, real circuits)')

    if run['params'] is None:
        print('Training data not found:', run['training_data_file'])

    show_circuits(run, visual_config)
    show_training_progress(run, visual_config)
    show_generated_probabilities(run, visual_config)

    if encoding == 'angle':
        show_angle_result_images(run, visual_config)

    elif encoding == 'amplitude':
        show_amp_result_images(run, visual_config)


# Run visualization
def run_visualization(config_file, visual_config):
    if visual_config['train']:
        run_train(config_file)

    run = load_visualization_run(config_file)
    show_visualization_run(run, visual_config)

    return run
