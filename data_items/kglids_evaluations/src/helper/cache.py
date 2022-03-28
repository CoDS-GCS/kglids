import pickle
import os


def cache_score(dumping_file: dict, k: int, top_k: list, save_as: str):
    with open('../cache/' + save_as + '_k-{}'.format(k)+ '.pkl', 'wb') as handle:
        pickle.dump(dumping_file, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('cached {}.pkl successfully!'.format(save_as + '_k-{}'.format(k)))
    if k != top_k[0]:
        previous_k_value = top_k[top_k.index(k) - 1]
        rm_obj = '../cache/' + save_as + '_k-{}'.format(previous_k_value) + '.pkl'

        if os.path.exists(rm_obj):
            os.remove(rm_obj)
