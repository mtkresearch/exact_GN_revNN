#from ray import tune
from hyperopt import hp

def reassign(search_space, param, value):
    if param in search_space:
        search_space[param] = value
        return True
    else:
        for k, v in search_space.items():
            if isinstance(v, dict):
                result = reassign(v, param, value)
                if result:
                    return True
    return False


def set_tune_params(search_space, tune_params):
    for param, space in tune_params.items():
        if space['type'] == 'choice':
            #reassign(search_space, param, tune.choice(space['values']))
            reassign(search_space, param, hp.choice(param, space['values']))
        elif space['type'] == 'uniform':
            #reassign(search_space, param, tune.uniform(space['min'], space['max']))
            reassign(search_space, param, hp.uniform(param, space['min'], space['max']))
        elif space['type'] == 'loguniform':
            ln10 = 2.3026
            reassign(search_space, param, hp.loguniform(param, space['min'] * ln10, space['max'] * ln10))
        elif space['type'] == 'randint':
            reassign(search_space, param, hp.randint(param, space['min'], space['max']))
