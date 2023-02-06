import pickle
import torch
import os

def save(args, metrics, model=None, seed=None):
    if 'results_dir' in args.keys():
        results_dir = args['results_dir'] + f"{args['session_id']}/" + f"LD{args['latent_dim']}_ND{args['num_inducing']}_AG{args['affine_grad']}" + (f"_HD{args['gpfa_h_dim']}/" if args['model']=='aea-sgpvae' else '/') + f"{args['model']}"
    else:
        results_dir = '_results/' + f"{args['session_id']}/" + f"LD{args['latent_dim']}_ND{args['num_inducing']}_AG{args['affine_grad']}" + (f"_HD{args['gpfa_h_dim']}/" if args['model']=='aea-sgpvae' else '/') + f"{args['model']}"

    if seed is None:
        if os.path.isdir(results_dir):
            i = 1
            while os.path.isdir(results_dir + '_' + str(i)):
                i += 1

            results_dir += '_' + str(i)
    else:
        results_dir = results_dir + str(seed)

    os.makedirs(results_dir, exist_ok=True)

    # Pickle args and metrics.
    with open(results_dir + '/args.pkl', 'wb') as f:
        pickle.dump(args, f)

    with open(results_dir + '/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    # Save args and results in text format.
    with open(results_dir + '/results.txt', 'w') as f:
        f.write('Args: \n')
        if isinstance(args, list):
            for d in args:
                f.write(str(d) + '\n')
        else:
            f.write(str(args) + '\n')

        f.write('\nMetrics: \n')
        for (key, value) in metrics.items():
            f.write('{}: {}\n'.format(key, value))
    
    if model is not None:
        torch.save(model, f'{results_dir}/trained_model.pt')