from spg import distributions,  jax_utils, run, data_utils

import jax.numpy as jnp
import jax

jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


def benchmark_mixtures(data, magic_df, num_mix=2):
    mask = data >= .1
    data = data[mask] - .1
    #scale = data.std()
    #data = data/scale
    
    t_prime = data_utils.get_tprime_for_times(data.index, magic_df['ssp245'])    
    data = data.values / data.values.std()
    eps = 1e-12

    # mixing_dist = npd.Categorical(probs=jnp.ones(3) / 3.)
    # component_dist = npd.Normal(loc=jnp.zeros(3), scale=jnp.ones(3))
    # mixture = npd.MixtureSameFamily(mixing_dist, component_dist)
    # mixture.sample(jax.random.PRNGKey(42)).shape
    # mixture.fit(data)
    # num_mix = 5

    for dist in [distributions.TFGammaMixCond]:
        dist = dist(num_mix=num_mix)
        print(f'----------------- Fitting distribution {dist.name} -----------------')
        dist.fit(data, cond=t_prime)#, weighting=(data**3+1))
        print(dist.get_params())

        print(f'Log prob: {dist.log_prob(data,  cond=t_prime)}')

        # a = np.linspace(0., 40, 1000)
        sample = dist.sample(len(data), cond=t_prime)

        # mask = sample < 400
        # sample = sample[mask]
        # print(f'Total number of data points {(~mask).sum()}') 
        run.plot_qq(data, sample, output_path=f'qq_{dist.name}_mix_{str(num_mix).zfill(2)}_v2.png')

if __name__ == '__main__':
    data = data_utils.load_data()
    magic = data_utils.load_nc('christchurch')
    benchmark_mixtures(data, magic)