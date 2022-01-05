from modules.rl.rl_policy import RandomPolicy, ShortestPathPolicy
from modules.rl.rl_utils import EvaluatePolicy, GetFullCoverageSample
from modules.rl.rl_custom_worlds import CreateWorlds

run_world_names = [
    'Manhattan3x3_PauseFreezeWorld',
    'Manhattan3x3_PauseDynamicWorld',
    'Manhattan5x5_DuplicateSetA',
    'Manhattan5x5_DuplicateSetB',
    'Manhattan5x5_FixedEscapeInit',
    'Manhattan5x5_VariableEscapeInit',
    'MetroU3_e17tborder_FixedEscapeInit',
    'MetroU3_e17t31_FixedEscapeInit', 
    'MetroU3_e17t0_FixedEscapeInit', 
    'MetroU3_e17tborder_VariableEscapeInit'
]
worlds = CreateWorlds(run_world_names, make_reflexive=True, state_repr='et', state_enc='nodes')

for env, world_name in zip(worlds, run_world_names):
    policies=[
        RandomPolicy(env),
        ShortestPathPolicy(env, weights = 'equal'),
        ShortestPathPolicy(env, weights = 'min_indegree'),
    ]
    policy_names=[
        'random',
        #'shortest_path',
        #'mindeg_path'
    ]
    for policy_name, policy in zip(policy_names, policies):
        logdir = 'results/testing/Heuristics/'+world_name+'/'+policy_name
        m=1
        if policy_name == 'random':
            if len(env.world_pool) < 10:
                m=10000
            elif len(env.world_pool) < 10000:
                m=10
            else:
                m=2
        _, returns, _ = EvaluatePolicy(env,policy, env.world_pool*m, print_runs=False, save_plots=False, logdir=logdir)
        plotlist = GetFullCoverageSample(returns, env.world_pool*m, bins=5, n=5)
        EvaluatePolicy(env, policy, plotlist, print_runs=True, save_plots=True, logdir=logdir)