import torch

# SB3 PPO
def GetHyperParams_SB3PPO(world):
    HP={}
    act=torch.nn.Tanh
    #act=torch.nn.ReLU

    # Manhattan3x3_PauseFreezeWorld
    hp={}
    #                      sr='et'        sr='etUt'       sr='ete0U0'       sr='etUte0U0'
    #                     -----------   -------------   ---------------   -----------------
    hp['actor']       = {'et':[64,64], 'etUt':[64,64], 'ete0U0':[64,64], 'etUte0U0':[64,64] }
    hp['critic']      = {'et':[64,64], 'etUt':[64,64], 'ete0U0':[64,64], 'etUte0U0':[64,64] }
    hp['activation']  = {'et':act    , 'etUt':act    , 'ete0U0':act    , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5      , 'etUt':5      , 'ete0U0':5      , 'etUte0U0':5       }
    hp['total_steps'] = {'et':25000  , 'etUt':25000  , 'ete0U0':25000  , 'etUte0U0':25000   }
    hp['eval_determ'] = {'et':False  , 'etUt':True   , 'ete0U0':False  , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':2000   , 'etUt':1      , 'ete0U0':2000   , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['Manhattan3x3_PauseFreezeWorld'] = hp
    # Manhattan3x3_PauseDynamicWorld
    hp={}
    #                      sr='et'        sr='etUt'       sr='ete0U0'       sr='etUte0U0'
    #                     -----------   -------------   ---------------   -----------------
    hp['actor']       = {'et':[64,64], 'etUt':[64,64], 'ete0U0':[64,64], 'etUte0U0':[64,64] }
    hp['critic']      = {'et':[64,64], 'etUt':[64,64], 'ete0U0':[64,64], 'etUte0U0':[64,64] }
    hp['activation']  = {'et':act    , 'etUt':act    , 'ete0U0':act    , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5      , 'etUt':5      , 'ete0U0':5      , 'etUte0U0':5       }
    hp['total_steps'] = {'et':20000  , 'etUt':20000  , 'ete0U0':20000  , 'etUte0U0':20000   }
    hp['eval_determ'] = {'et':False  , 'etUt':True   , 'ete0U0':False  , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':2000   , 'etUt':1      , 'ete0U0':2000   , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['Manhattan3x3_PauseDynamicWorld'] = hp

    # Manhattan5x5_FixedEscapeInit
    hp={}
    #                      sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                     -----------        ------------------   --------------------   ----------------------
    hp['actor']       = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['critic']      = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['activation']  = {'et':act         , 'etUt':act         , 'ete0U0':act         , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5           , 'etUt':5           , 'ete0U0':5           , 'etUte0U0':5       }
    hp['total_steps'] = {'et':200000      , 'etUt':350000      , 'ete0U0':350000      , 'etUte0U0':350000  }
    hp['eval_determ'] = {'et':False       , 'etUt':True        , 'ete0U0':False       , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':4           , 'etUt':1           , 'ete0U0':4           , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['Manhattan5x5_FixedEscapeInit'] = hp
    # Manhattan5x5_VariableEscapeInit
    hp={}
    #                      sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                     -----------        ------------------   --------------------   ----------------------
    hp['actor']       = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['critic']      = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['activation']  = {'et':act         , 'etUt':act         , 'ete0U0':act         , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5           , 'etUt':5           , 'ete0U0':5           , 'etUte0U0':5       }
    hp['total_steps'] = {'et':200000      , 'etUt':200000      , 'ete0U0':200000      , 'etUte0U0':200000  }
    hp['eval_determ'] = {'et':False       , 'etUt':True        , 'ete0U0':False       , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':2           , 'etUt':1           , 'ete0U0':2           , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['Manhattan5x5_VariableEscapeInit'] = hp
    # Manhattan5x5_DuplicateSetA
    hp={}
    #                      sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                     -----------        ------------------   --------------------   ----------------------
    hp['actor']       = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['critic']      = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['activation']  = {'et':act         , 'etUt':act         , 'ete0U0':act         , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5           , 'etUt':5           , 'ete0U0':5           , 'etUte0U0':5       }
    hp['total_steps'] = {'et':50000       , 'etUt':50000       , 'ete0U0':50000       , 'etUte0U0':50000  }
    hp['eval_determ'] = {'et':False       , 'etUt':True        , 'ete0U0':False       , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':1000        , 'etUt':1           , 'ete0U0':1000        , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['Manhattan5x5_DuplicateSetA'] = hp
    # Manhattan5x5_DuplicateSetB
    hp={}
    #                      sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                     -----------        ------------------   --------------------   ----------------------
    hp['actor']       = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['critic']      = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['activation']  = {'et':act         , 'etUt':act         , 'ete0U0':act         , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5           , 'etUt':5           , 'ete0U0':5           , 'etUte0U0':5       }
    hp['total_steps'] = {'et':50000       , 'etUt':50000       , 'ete0U0':50000       , 'etUte0U0':50000  }
    hp['eval_determ'] = {'et':False       , 'etUt':True        , 'ete0U0':False       , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':1000        , 'etUt':1           , 'ete0U0':1000        , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['Manhattan5x5_DuplicateSetB'] = hp


    # MetroU3_e17tborder_FixedEscapeInit
    hp={}
    #                      sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                     -----------        ------------------   --------------------   ----------------------
    hp['actor']       = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['critic']      = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['activation']  = {'et':act         , 'etUt':act         , 'ete0U0':act         , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5           , 'etUt':5           , 'ete0U0':5           , 'etUte0U0':5       }
    hp['total_steps'] = {'et':150000      , 'etUt':150000      , 'ete0U0':150000      , 'etUte0U0':150000  }
    hp['eval_determ'] = {'et':False       , 'etUt':True        , 'ete0U0':False       , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':4           , 'etUt':1           , 'ete0U0':4           , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['MetroU3_e17tborder_FixedEscapeInit'] = hp
     # MetroU3_e17tborder_VariableEscapeInit
    hp={}
    #                      sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                     -----------        ------------------   --------------------   ----------------------
    hp['actor']       = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['critic']      = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['activation']  = {'et':act         , 'etUt':act         , 'ete0U0':act         , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5           , 'etUt':5           , 'ete0U0':5           , 'etUte0U0':5       }
    hp['total_steps'] = {'et':150000      , 'etUt':150000      , 'ete0U0':150000      , 'etUte0U0':150000  }
    hp['eval_determ'] = {'et':False       , 'etUt':True        , 'ete0U0':False       , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':2           , 'etUt':1           , 'ete0U0':2           , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['MetroU3_e17tborder_VariableEscapeInit'] = hp
    # MetroU3_e17t31_FixedEscapeInit
    hp={}
    #                      sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                     -----------        ------------------   --------------------   ----------------------
    hp['actor']       = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['critic']      = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['activation']  = {'et':act         , 'etUt':act         , 'ete0U0':act         , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5           , 'etUt':5           , 'ete0U0':5           , 'etUte0U0':5       }
    hp['total_steps'] = {'et':150000      , 'etUt':150000      , 'ete0U0':150000      , 'etUte0U0':150000  }
    hp['eval_determ'] = {'et':False       , 'etUt':True        , 'ete0U0':False       , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':4           , 'etUt':1           , 'ete0U0':4           , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['MetroU3_e17t31_FixedEscapeInit'] = hp
    # MetroU3_e17t0_FixedEscapeInit
    hp={}
    #                      sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                     -----------        ------------------   --------------------   ----------------------
    hp['actor']       = {'et':[256,256,64], 'etUt':[256,256,64], 'ete0U0':[256,256,64], 'etUte0U0':[256,256,64] }
    hp['critic']      = {'et':[256,256,64], 'etUt':[256,256,64], 'ete0U0':[256,256,64], 'etUte0U0':[256,256,64] }
    hp['activation']  = {'et':act         , 'etUt':act         , 'ete0U0':act         , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5           , 'etUt':5           , 'ete0U0':5           , 'etUte0U0':5       }
    hp['total_steps'] = {'et':150000      , 'etUt':150000      , 'ete0U0':150000      , 'etUte0U0':150000  }
    hp['eval_determ'] = {'et':False       , 'etUt':True        , 'ete0U0':False       , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':4           , 'etUt':1           , 'ete0U0':4           , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['MetroU3_e17t0_FixedEscapeInit'] = hp
  
    return HP[world]    