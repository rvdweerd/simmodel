from stable_baselines3.common.callbacks import BaseCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy

class TensorboardLoggingCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardLoggingCallback, self).__init__(verbose)
    def _on_step(self):
        self.logger.record('loss',1)
         



class TestCallBack(BaseCallback):
    def __init__(self, verbose=0, logdir=''):
        super(TestCallBack, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.best_res_det = -1e6
        self.best_res_nondet = -1e6
        self.logdir = logdir
    def _on_step(self):
        pass#print('on_step  calls',self.n_calls) 
    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        res_det = evaluate_policy(self.model, self.training_env, n_eval_episodes=32, reward_threshold=-100, warn=False, return_episode_rewards=False, deterministic=True)
        print('Test result (deterministic): avg rew:', res_det[0], 'std:', res_det[1])
        if res_det[0] >= self.best_res_det:
            self.best_res_det = res_det[0]
            print('...New best det results, saving model')
            self.model.save(self.logdir+"/model_best")
            OF_det    = open(self.logdir+'/model_best_save_history.txt', 'a')
            OF_det.write('timestep:'+str(self.num_timesteps)+', avg det res:'+str(res_det[0])+'\n')
            OF_det.close()

        res_nondet = evaluate_policy(self.model, self.training_env, n_eval_episodes=32, reward_threshold=-100, warn=False, return_episode_rewards=False, deterministic=False)
        print('Test result (non-deterministic): avg rew:', res_nondet[0], 'std:', res_nondet[1])
        if res_nondet[0] >= self.best_res_nondet:
            self.best_res_nondet = res_nondet[0]
            print('...New best nondet results, saving model')
            self.model.save(self.logdir+"/model_nondet_best")
            OF_nondet = open(self.logdir+'/model_nondet_best_save_history.txt', 'a')
            OF_nondet.write('timestep:'+str(self.num_timesteps)+', avg nondet res:'+str(res_nondet[0])+'\n')
            OF_nondet.close()


class SimpleCallback(BaseCallback):
    """
    a simple callback that can only be called twice

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, verbose=0):
        super(SimpleCallback, self).__init__(verbose)
        self._called = False
    
    def _on_step(self):
      if not self._called:
        print("callback - first call")
        self._called = True
        return True # returns True, training continues.
      print("callback - second call")
      return False # returns False, training stops. 