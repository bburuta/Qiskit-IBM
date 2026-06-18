import torch

from datasets.images_dataset import get_image_shape
from encoding import images_to_prob
from datasets.quantum_dataset import circuits_to_probs
from batch_torch import generate_torch_random_input, generate_torch_real_input_index



#- Evaluation methods -#

# Evaluation method: KL-Div of generated and real sample
def evaluate_kl(gen_dist, target):
    return torch.nn.functional.kl_div(
        input=gen_dist.clamp_min(1e-10).log(),
        target=target,
        reduction='batchmean'
    ).item()


# Evaluate specific gradient (top-left to bottom-right) for small images
def evaluate_gradient(gen_dists, target=None, image_shape=None):
    if image_shape is None:
        image_shape = get_image_shape(gen_dists.shape[1])

    img_h, img_w = image_shape
    batch = gen_dists.reshape(-1, img_h, img_w)

    h_diff = batch[:, :, 1:] - batch[:, :, :-1]
    v_diff = batch[:, 1:, :] - batch[:, :-1, :]

    h_penalty = (-h_diff.clamp(max=0)).mean()
    v_penalty = (-v_diff.clamp(max=0)).mean()

    penalty = 0.5 * (h_penalty + v_penalty)
    return penalty.item()


# Get evaluation function
def get_evaluation_function(eval_method, image_shape=None):
    if eval_method == 'kl':
        evaluate = evaluate_kl
        compute_targets = True

    elif eval_method == 'gradient':
        def evaluate(gen_dist, target=None):
            return evaluate_gradient(gen_dist, target=target, image_shape=image_shape)
        compute_targets = False

    else:
        raise ValueError(f"Unknown evaluation method: {eval_method}")
    
    return evaluate, compute_targets


#- Evaluation data -#

# Get target probability distributions
def get_target_probs(dataset_type, contrast, device, dtype, real_circuits=None, X=None):
    if dataset_type == 'quantum':
        if real_circuits is None:
            raise ValueError("real_circuits must be provided for quantum datasets.")
        
        probs = circuits_to_probs(real_circuits)
        target_probs = torch.as_tensor(probs, device=device, dtype=dtype)

    elif dataset_type == 'classical':
        if X is None:
            raise ValueError("X must be provided for classical datasets.")
        
        X = torch.as_tensor(X, device=device, dtype=dtype)
        
        target_probs = images_to_prob(X, contrast)

    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    return target_probs


#- Batch evaluation -#

# Evaluate batch of generated samples with real samples
def batch_evaluation(eval_method, randomness, eval_g, batch_size, num_random_params, device, dtype, target_probs=None, image_shape=None):
    evaluate, compute_targets = get_evaluation_function(eval_method, image_shape=image_shape)

    with torch.no_grad():
        # Get fake samples
        random_input = generate_torch_random_input(batch_size, num_random_params, randomness, device, dtype)
        fake_outputs = eval_g(random_input)

        # Get real samples
        if compute_targets:
            if target_probs is None:
                raise ValueError("target_probs must be provided for KL evaluation.")

            real_indexes = generate_torch_real_input_index(batch_size, target_probs.shape[0], device)
            targets = target_probs[real_indexes]
        else:
            targets = None

        current_eval = evaluate(fake_outputs, targets)

    return current_eval
