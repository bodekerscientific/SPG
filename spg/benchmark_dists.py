from spg import distributions,  jax_utils, run, data_utils
from pathlib import Path
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import numpy as np


jax.config.update('jax_platform_name', 'cpu')
jax.config.update("jax_enable_x64", True)


def benchmark_mixtures(df, magic_df, num_mix=2, loc='dunedin'):
    #mask = df >= .1
    #df = df[mask] - .1
    #scale = data.std()
    #data = data/scale

    t_prime = data_utils.get_tprime_for_times(df.index, magic_df['ssp245'])
    all_data = df.values
    eps = 1e-12
    rng = jax.random.PRNGKey(42)
    # mixing_dist = npd.Categorical(probs=jnp.ones(3) / 3.)
    # component_dist = npd.Normal(loc=jnp.zeros(3), scale=jnp.ones(3))
    # mixture = npd.MixtureSameFamily(mixing_dist, component_dist)
    # mixture.sample(jax.random.PRNGKey(42)).shape
    # mixture.fit(data)
    # num_mix = 5

    years = jnp.arange(1910, 2021, 10)
    all_q = []
    all_wd = []
    target_q = np.array([0, 0.95, 0.975, 0.99, 0.995, 0.999])
    all_params = []
    for n_start, n_end in zip(years[:-1], years[1:]):
        mask = (df.index.year >= n_start) & (df.index.year < n_end)
        data = all_data[mask]
        
    #     wd = (data >= .1).sum()
    #     all_wd.append(wd)
        
    #     q_sum = []
    #     for q in np.quantile(data, target_q):
    #         q_sum.append(data[data > q].mean())
    #     all_q.append(np.array(q_sum))
        
    # all_q = np.stack(all_q, axis=1)
    
    # for n in range(all_q.shape[0]):
    #     plt.figure(figsize=(12, 8))
    #     plt.plot(years[:-1], all_q[n])
    #     plt.savefig(f'precip_{loc}_{target_q[n]}.png')
    #     plt.close()

    # plt.plot(years[:-1], all_wd)
    # plt.savefig(f'wd_{loc}.png')
    # plt.close()
    
        param_init = None
        for dist in [distributions.TFGamma]:
            dist = dist(param_init=param_init)
            print(
                f'----------------- Fitting distribution {dist.name} -----------------')
            dist.fit(data)  # , weighting=(data**3+1))
            print(dist.get_params())

            print(f'Log prob: {dist.log_prob(data,  cond=t_prime)}')

            # a = np.linspace(0., 40, 1000)
            sample = dist.sample(len(data), rng)

            # mask = sample < 400
            # sample = sample[mask]
            # print(f'Total number of data points {(~mask).sum()}')
            run.plot_qq(
                data, sample, output_path=f'qq_{loc}_{dist.name}_mix_{str(num_mix).zfill(2)}_{n_start}_v2.png')
            param_init = dist._params
            
            all_params.append(dist.get_params())
        
    all_params = jnp.array(all_params)
        
    for n in range(2):
        plt.figure(figsize=(12, 8))
        plt.plot(years[:-1], all_params[:, n])
        plt.savefig(f'param_change+{loc}_{n}.png')


if __name__ == '__main__':
    #data = data_utils.load_data()

    input_path = Path('/mnt/temp/projects/otago_uni_marsden/data_keep/spg/station_data_hourly/')
    #data = data_utils.load_nc(input_path / 'christchurch.nc')
    data = data_utils.load_data()
    
    benchmark_mixtures(data, data_utils.load_magic())
