import time

from qgan_v4.config.loader import prepare_run_config
from qgan_v4.implementations.registry import get_implementation
from qgan_v4.storage.paths import get_run_path
from qgan_v4.training.data import save_best_gen_params, save_checkpoint
from qgan_v4.training.interrupter import Interrupter


#- Training display -#

# Print training progress row
def print_progress(epoch, gen_loss, disc_loss, current_eval, best_eval, elapsed_time):
    values = (epoch, gen_loss, disc_loss, current_eval, best_eval, elapsed_time)
    for value in values: print(f"{value:.3g} ".rjust(20), end="|")
    print()


# Print training summary
def print_summary(state):
    print(
        "Training complete:",
        "\n   Data path:", get_run_path(state.config),
        "\n   Best eval:", state.metrics.best_eval(),
        "\n   Total time:", sum(state.metrics.times.values()),
    )


#- Training loop -#

# Train qGAN model
def train(config, interrupter=None):
    config = prepare_run_config(config)
    impl = get_implementation(config)
    state = impl.setup(config)
    interrupter = interrupter or Interrupter()

    if config["training"]["print_every"]:
        names = ("Epoch", "Generator cost", "Discriminator cost", "Eval", "Best eval", "Time")
        for name in names: print(f"{name} ".rjust(20), end="|")
        print()

    epoch = state.current_epoch - 1
    best_eval = state.metrics.best_eval()
    prev_times = sum(state.metrics.times.values())
    start_time = time.time()

    try:
        for epoch in range(state.current_epoch, config["training"]["max_iterations"]):
            disc_loss = float("nan")
            gen_loss = float("nan")

            # Train discriminator
            for _ in range(config["training"]["disc_iterations"]):
                disc_loss = impl.train_discriminator_step(state)
                state.metrics.dloss[epoch] = disc_loss

            # Train generator
            for _ in range(config["training"]["gen_iterations"]):
                gen_loss = impl.train_generator_step(state)
                state.metrics.gloss[epoch] = gen_loss

            # Evaluate generator
            current_eval = impl.evaluate(state)
            state.metrics.eval[epoch] = current_eval

            # Save best generator parameters
            if current_eval < best_eval:
                best_eval = current_eval
                save_best_gen_params(state)

            # Update epoch time
            state.metrics.times[epoch] = time.time() - start_time
            start_time = time.time()

            # Print epoch progress
            print_every = config["training"]["print_every"]
            if print_every and epoch % print_every == 0:
                now_times = sum(state.metrics.times.values())
                print_progress(epoch, gen_loss, disc_loss, current_eval, best_eval, now_times - prev_times)
                prev_times = now_times

            # Stop gracefully after interruption
            if interrupter.kill_now:
                print("Interrupter: Graceful exit triggered. Breaking loop.")
                break

    finally:
        # Save checkpoint and close implementation
        try:
            save_checkpoint(state, epoch)
        finally:
            impl.close(state)
        print_summary(state)

    return state
