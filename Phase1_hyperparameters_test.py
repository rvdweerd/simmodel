import torch

# DQN
def GetHyperParams_DQN(world):
    HP={}
    # MetroU3_e17t0_FixedEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          ------------        ------------------   --------------------   ----------------------
    hp['dims_hidden_layers']= {'et':[256,256],   'etUt':[256,256],      'ete0U0':[256,256],     'etUte0U0':[256,256], }
    hp['batch_size']        = {'et':64,          'etUt':64,             'ete0U0':64,            'etUte0U0':64,        }
    hp['mem_size']          = {'et':25000,       'etUt':25000,          'ete0U0':25000,         'etUte0U0':25000,     }
    hp['learning_rate']     = {'et':2e-4,        'etUt':1e-3,           'ete0U0':1e-3,          'etUte0U0':2e-4,      }
    hp['num_episodes']      = {'et':35000,       'etUt':35000,          'ete0U0':35000,         'etUte0U0':35000,     }
    hp['eps_0']             = {'et':1.,          'etUt':1.,             'ete0U0':1.,            'etUte0U0':1.,        }
    hp['eps_min']           = {'et':0.05,        'etUt':0.05,           'ete0U0':0.05,          'etUte0U0':0.05,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,            'ete0U0':0.8,           'etUte0U0':0.8,       }  
    HP['MetroU3_e17t0_FixedEscapeInit'] = hp    
    
    # MetroU3_e17t31_FixedEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          --------------        ----------------   --------------------   ----------------------
    hp['dims_hidden_layers']= {'et':[256,256],   'etUt':[256,256],      'ete0U0':[256,256],    'etUte0U0':[256,256], }
    hp['batch_size']        = {'et':64,          'etUt':64,             'ete0U0':64,            'etUte0U0':64,        }
    hp['mem_size']          = {'et':25000,       'etUt':25000,          'ete0U0':25000,         'etUte0U0':25000,     }
    hp['learning_rate']     = {'et':2e-4,        'etUt':2e-4,           'ete0U0':2e-4,          'etUte0U0':2e-4,      }
    hp['num_episodes']      = {'et':30000,       'etUt':30000,          'ete0U0':30000,         'etUte0U0':30000,     }
    hp['eps_0']             = {'et':1.,          'etUt':1.,             'ete0U0':1.,            'etUte0U0':1.,        }
    hp['eps_min']           = {'et':0.05,        'etUt':0.05,           'ete0U0':0.05,          'etUte0U0':0.05,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,            'ete0U0':0.8,           'etUte0U0':0.8,       }  
    HP['MetroU3_e17t31_FixedEscapeInit'] = hp    
    
    return HP[world]

