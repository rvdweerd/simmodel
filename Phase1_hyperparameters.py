import torch

# DQN
def GetHyperParams_DQN(world):
    HP={}
    # Manhattan3x3_PauseFreezeWorld
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        ------------------   --------------------   ----------------------
    hp['dims_hidden_layers']= {'et':[128,128],     'etUt':[128,128],   'ete0U0':[128,128],    'etUte0U0':[128,128], }
    hp['batch_size']        = {'et':64,            'etUt':64,          'ete0U0':64,           'etUte0U0':64,        }
    hp['mem_size']          = {'et':4000,          'etUt':4000,        'ete0U0':4000,         'etUte0U0':4000,      }
    hp['learning_rate']     = {'et':5e-4,          'etUt':5e-4,        'ete0U0':5e-4,         'etUte0U0':5e-4,      }
    hp['num_episodes']      = {'et':1500,          'etUt':1500,        'ete0U0':1500,         'etUte0U0':2500,      }
    hp['eps_0']             = {'et':1.,            'etUt':1.,          'ete0U0':1.,           'etUte0U0':1.,        }
    hp['eps_min']           = {'et':.1,            'etUt':.1,          'ete0U0':.1,           'etUte0U0':.1,        }
    hp['cutoff_factor']     = {'et':0.9,           'etUt':0.9,         'ete0U0':0.9,          'etUte0U0':0.9,       }  
    HP['Manhattan3x3_PauseFreezeWorld'] = hp
    # Manhattan3x3_PauseDynamicWorld
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        ------------------   --------------------   ----------------------
    hp['dims_hidden_layers']= {'et':[128],         'etUt':[128],       'ete0U0':[128],            'etUte0U0':[128], }
    hp['batch_size']        = {'et':32,            'etUt':32,          'ete0U0':32,               'etUte0U0':32,    }
    hp['mem_size']          = {'et':300,           'etUt':300,         'ete0U0':300,              'etUte0U0':300,   }
    hp['learning_rate']     = {'et':1e-5,          'etUt':1e-5,        'ete0U0':1e-5,             'etUte0U0':1e-5,  }
    hp['num_episodes']      = {'et':3500,          'etUt':3500,        'ete0U0':3500,             'etUte0U0':3500,  }
    hp['eps_0']             = {'et':1.,            'etUt':1.,          'ete0U0':1.,               'etUte0U0':1.,    }
    hp['eps_min']           = {'et':0.1,           'etUt':0.1,         'ete0U0':0.1,              'etUte0U0':0.1,   }
    hp['cutoff_factor']     = {'et':0.8,           'etUt':0.8,         'ete0U0':0.8,              'etUte0U0':0.8,   }  
    HP['Manhattan3x3_PauseDynamicWorld'] = hp  
    # Manhattan5x5_FixedEscapeInit
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        -----------------     -------------------     ---------------------
    hp={}
    hp['dims_hidden_layers']= {'et':[256,128],   'etUt':[256,128],      'ete0U0':[256,128],     'etUte0U0':[256,128], }
    hp['batch_size']        = {'et':64,          'etUt':64,             'ete0U0':64,            'etUte0U0':64,        }
    hp['mem_size']          = {'et':15000,       'etUt':15000,          'ete0U0':15000,         'etUte0U0':15000,     }
    hp['learning_rate']     = {'et':1e-4,        'etUt':1e-4,           'ete0U0':1e-4,          'etUte0U0':1e-4,      }
    hp['num_episodes']      = {'et':25000,       'etUt':25000,          'ete0U0':25000,         'etUte0U0':25000,     }
    hp['eps_0']             = {'et':1.,          'etUt':1.,             'ete0U0':1.,            'etUte0U0':1.,        }
    hp['eps_min']           = {'et':0.05,        'etUt':0.05,           'ete0U0':0.05,          'etUte0U0':0.05,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,            'ete0U0':0.8,           'etUte0U0':0.8,       }  
    HP['Manhattan5x5_FixedEscapeInit'] = hp
    
    # Manhattan5x5_VariableEscapeInit
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        -----------------     -------------------     ---------------------
    hp={}
    hp['dims_hidden_layers']= {'et':[256,128],   'etUt':[256,128],      'ete0U0':[256,128],     'etUte0U0':[256,128], }
    hp['batch_size']        = {'et':64,          'etUt':64,             'ete0U0':64,            'etUte0U0':64,        }
    hp['mem_size']          = {'et':15000,       'etUt':15000,          'ete0U0':15000,         'etUte0U0':15000,     }
    hp['learning_rate']     = {'et':1e-4,        'etUt':1e-4,           'ete0U0':1e-4,          'etUte0U0':1e-4,      }
    hp['num_episodes']      = {'et':25000,       'etUt':25000,          'ete0U0':25000,         'etUte0U0':25000,     }
    hp['eps_0']             = {'et':1.,          'etUt':1.,             'ete0U0':1.,            'etUte0U0':1.,        }
    hp['eps_min']           = {'et':0.05,        'etUt':0.05,           'ete0U0':0.05,          'etUte0U0':0.05,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,            'ete0U0':0.8,           'etUte0U0':0.8,       }  
    HP['Manhattan5x5_VariableEscapeInit'] = hp

    # Manhattan5x5_DuplicateSetA
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          ---FIXED---        ----FIXED-------   ----FIXED-----------   ------FIXED----------
    hp['dims_hidden_layers']= {'et':[128],   'etUt':[128],      'ete0U0':[256,128],     'etUte0U0':[256,128], }
    hp['batch_size']        = {'et':64,          'etUt':64,             'ete0U0':64,            'etUte0U0':64,        }
    hp['mem_size']          = {'et':1000,       'etUt':1000,          'ete0U0':15000,         'etUte0U0':15000,     }
    hp['learning_rate']     = {'et':1e-3,        'etUt':1e-3,           'ete0U0':1e-4,          'etUte0U0':1e-4,      }
    hp['num_episodes']      = {'et':200,       'etUt':200,          'ete0U0':25000,         'etUte0U0':25000,     }
    hp['eps_0']             = {'et':1.,          'etUt':1.,             'ete0U0':1.,            'etUte0U0':1.,        }
    hp['eps_min']           = {'et':0.1,        'etUt':0.1,           'ete0U0':0.05,          'etUte0U0':0.05,      }
    hp['cutoff_factor']     = {'et':0.9,         'etUt':0.9,            'ete0U0':0.8,           'etUte0U0':0.8,       }  
    HP['Manhattan5x5_DuplicateSetA'] = hp
    # Manhattan5x5_DuplicateSetB
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        ------------------   --------------------   ----------------------
    hp['dims_hidden_layers']= {'et':[128],   'etUt':[128],      'ete0U0':[128],     'etUte0U0':[256,128], }
    hp['batch_size']        = {'et':64,          'etUt':64,             'ete0U0':64,            'etUte0U0':64,        }
    hp['mem_size']          = {'et':1000,       'etUt':1000,          'ete0U0':1000,         'etUte0U0':15000,     }
    hp['learning_rate']     = {'et':1e-3,        'etUt':1e-3,           'ete0U0':1e-3,          'etUte0U0':1e-4,      }
    hp['num_episodes']      = {'et':200,       'etUt':200,          'ete0U0':200,         'etUte0U0':25000,     }
    hp['eps_0']             = {'et':1.,          'etUt':1.,             'ete0U0':1.,            'etUte0U0':1.,        }
    hp['eps_min']           = {'et':0.1,        'etUt':0.1,           'ete0U0':0.1,          'etUte0U0':0.05,      }
    hp['cutoff_factor']     = {'et':0.9,         'etUt':0.9,            'ete0U0':0.9,           'etUte0U0':0.8,       }  
    HP['Manhattan5x5_DuplicateSetB'] = hp

    # MetroU3_e17tborder_FixedEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          --FIXED-----        ----FIXED----------   --------------------   ----------------------
    hp['dims_hidden_layers']= {'et':[256,128],   'etUt':[256,128],      'ete0U0':[256,128],     'etUte0U0':[256,128], }
    hp['batch_size']        = {'et':64,          'etUt':64,             'ete0U0':64,            'etUte0U0':64,        }
    hp['mem_size']          = {'et':15000,       'etUt':15000,          'ete0U0':15000,         'etUte0U0':15000,     }
    hp['learning_rate']     = {'et':1e-4,        'etUt':1e-4,           'ete0U0':1e-4,          'etUte0U0':1e-4,      }
    hp['num_episodes']      = {'et':25000,       'etUt':25000,          'ete0U0':25000,         'etUte0U0':25000,     }
    hp['eps_0']             = {'et':1.,          'etUt':1.,             'ete0U0':1.,            'etUte0U0':1.,        }
    hp['eps_min']           = {'et':0.05,        'etUt':0.05,           'ete0U0':0.05,          'etUte0U0':0.05,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,            'ete0U0':0.8,           'etUte0U0':0.8,       }  
    HP['MetroU3_e17tborder_FixedEscapeInit'] = hp
    # MetroU3_e17tborder_VariableEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          --FIXED-----        ----FIXED----------   --------------------   ----------------------
    hp['dims_hidden_layers']= {'et':[256,128],   'etUt':[256,128],      'ete0U0':[256,128],     'etUte0U0':[256,128], }
    hp['batch_size']        = {'et':64,          'etUt':64,             'ete0U0':64,            'etUte0U0':64,        }
    hp['mem_size']          = {'et':15000,       'etUt':15000,          'ete0U0':15000,         'etUte0U0':15000,     }
    hp['learning_rate']     = {'et':1e-4,        'etUt':1e-4,           'ete0U0':1e-4,          'etUte0U0':1e-4,      }
    hp['num_episodes']      = {'et':25000,       'etUt':25000,          'ete0U0':25000,         'etUte0U0':25000,     }
    hp['eps_0']             = {'et':1.,          'etUt':1.,             'ete0U0':1.,            'etUte0U0':1.,        }
    hp['eps_min']           = {'et':0.05,        'etUt':0.05,           'ete0U0':0.05,          'etUte0U0':0.05,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,            'ete0U0':0.8,           'etUte0U0':0.8,       }  
    HP['MetroU3_e17tborder_VariableEscapeInit'] = hp
    # MetroU3_e17t0_FixedEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          ------------        ------------------   --------------------   ----------------------
    hp['dims_hidden_layers']= {'et':[256,128],    'etUt':[256,256],      'ete0U0':[256,128],     'etUte0U0':[256,256], }
    hp['batch_size']        = {'et':64,           'etUt':64,             'ete0U0':64,            'etUte0U0':64,        }
    hp['mem_size']          = {'et':15000,        'etUt':25000,          'ete0U0':15000,         'etUte0U0':25000,     }
    hp['learning_rate']     = {'et':1e-4,         'etUt':2e-4,           'ete0U0':1e-4,          'etUte0U0':2e-4,      }
    hp['num_episodes']      = {'et':25000,        'etUt':35000,          'ete0U0':25000,         'etUte0U0':35000,     }
    hp['eps_0']             = {'et':1.,           'etUt':1.,             'ete0U0':1.,            'etUte0U0':1.,        }
    hp['eps_min']           = {'et':0.05,         'etUt':0.05,           'ete0U0':0.05,          'etUte0U0':0.05,      }
    hp['cutoff_factor']     = {'et':0.8,          'etUt':0.8,            'ete0U0':0.8,           'etUte0U0':0.8,       }  
    HP['MetroU3_e17t0_FixedEscapeInit'] = hp    
    # MetroU3_e17t31_FixedEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          --------------        ----------------   --------------------   ----------------------
    hp['dims_hidden_layers']= {'et':[256,128],   'etUt':[256,256],      'ete0U0':[256,128],     'etUte0U0':[256,256], }
    hp['batch_size']        = {'et':64,          'etUt':64,             'ete0U0':64,            'etUte0U0':64,        }
    hp['mem_size']          = {'et':15000,       'etUt':25000,          'ete0U0':15000,         'etUte0U0':25000,     }
    hp['learning_rate']     = {'et':1e-4,        'etUt':2e-4,           'ete0U0':1e-4,          'etUte0U0':2e-4,      }
    hp['num_episodes']      = {'et':25000,       'etUt':30000,          'ete0U0':25000,         'etUte0U0':30000,     }
    hp['eps_0']             = {'et':1.,          'etUt':1.,             'ete0U0':1.,            'etUte0U0':1.,        }
    hp['eps_min']           = {'et':0.05,        'etUt':0.05,           'ete0U0':0.05,          'etUte0U0':0.05,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,            'ete0U0':0.8,           'etUte0U0':0.8,       }  
    HP['MetroU3_e17t31_FixedEscapeInit'] = hp    
    
    return HP[world]

# DRQN
def GetHyperParams_DRQN(world):
    HP={}
    # Manhattan3x3_PauseFreezeWorld
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        ------------------   --------------------   ----------------------
    hp['lstm_hidden_size']  = {'et':18,            'etUt':18,          'ete0U0':18,               'etUte0U0':18,    }
    hp['dims_hidden_layers']= {'et':[128],         'etUt':[128],       'ete0U0':[128],            'etUte0U0':[128], }
    hp['batch_size']        = {'et':64,            'etUt':64,          'ete0U0':64,               'etUte0U0':64,    }
    hp['mem_size']          = {'et':300,           'etUt':300,         'ete0U0':300,              'etUte0U0':300,   }
    hp['learning_rate']     = {'et':5e-6,          'etUt':5e-6,        'ete0U0':5e-6,             'etUte0U0':5e-6,  }
    hp['num_episodes']      = {'et':3500,          'etUt':3500,        'ete0U0':3500,             'etUte0U0':3500,  }
    hp['eps_0']             = {'et':1.,            'etUt':1.,          'ete0U0':1.,               'etUte0U0':1.,    }
    hp['eps_min']           = {'et':0.1,           'etUt':0.1,         'ete0U0':0.1,              'etUte0U0':0.1,   }
    hp['cutoff_factor']     = {'et':0.8,           'etUt':0.8,         'ete0U0':0.8,              'etUte0U0':0.8,   }  
    hp['target_update_freq']= {'et':1,             'etUt':1,           'ete0U0':1,                'etUte0U0':1,     }      
    HP['Manhattan3x3_PauseFreezeWorld'] = hp    
    # Manhattan3x3_PauseDynamicWorld
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        ------------------   --------------------   ----------------------
    hp['lstm_hidden_size']  = {'et':18,            'etUt':18,          'ete0U0':18,               'etUte0U0':18,    }
    hp['dims_hidden_layers']= {'et':[128],         'etUt':[128],       'ete0U0':[128],            'etUte0U0':[128], }
    hp['batch_size']        = {'et':64,            'etUt':64,          'ete0U0':64,               'etUte0U0':64,    }
    hp['mem_size']          = {'et':300,           'etUt':300,         'ete0U0':300,              'etUte0U0':300,   }
    hp['learning_rate']     = {'et':5e-6,          'etUt':5e-6,        'ete0U0':5e-6,             'etUte0U0':5e-6,  }
    hp['num_episodes']      = {'et':3500,          'etUt':3500,        'ete0U0':3500,             'etUte0U0':3500,  }
    hp['eps_0']             = {'et':1.,            'etUt':1.,          'ete0U0':1.,               'etUte0U0':1.,    }
    hp['eps_min']           = {'et':0.1,           'etUt':0.1,         'ete0U0':0.1,              'etUte0U0':0.1,   }
    hp['cutoff_factor']     = {'et':0.8,           'etUt':0.8,         'ete0U0':0.8,              'etUte0U0':0.8,   }  
    hp['target_update_freq']= {'et':1,             'etUt':1,           'ete0U0':1,                'etUte0U0':1,     }      
    HP['Manhattan3x3_PauseDynamicWorld'] = hp    
    # Manhattan5x5_FixedEscapeInit
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        -----------------     -------------------     ---------------------
    hp={}
    hp['lstm_hidden_size']  = {'et':64,          'etUt':64,           'ete0U0':64,           'etUte0U0':64,       }
    hp['dims_hidden_layers']= {'et':[256,256],   'etUt':[1028],       'ete0U0':[256,256],    'etUte0U0':[256,256],}
    hp['batch_size']        = {'et':256,         'etUt':256,          'ete0U0':256,          'etUte0U0':256,      }
    hp['mem_size']          = {'et':1500,        'etUt':500,          'ete0U0':1500,         'etUte0U0':1500,     }
    hp['learning_rate']     = {'et':1e-6,        'etUt':5e-6,         'ete0U0':1e-6,         'etUte0U0':1e-6,     }
    hp['num_episodes']      = {'et':50000,       'etUt':25000,        'ete0U0':50000,        'etUte0U0':50000,    }
    hp['eps_0']             = {'et':1.,          'etUt':1.,           'ete0U0':1.,           'etUte0U0':1.,       }
    hp['eps_min']           = {'et':0.1,         'etUt':0.1,          'ete0U0':0.1,          'etUte0U0':0.1,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,          'ete0U0':0.8,          'etUte0U0':0.8,      }  
    hp['target_update_freq']= {'et':100,         'etUt':100,          'ete0U0':100,          'etUte0U0':100,      }  
    HP['Manhattan5x5_FixedEscapeInit'] = hp
    # MetroU3_e17tborder_FixedEscapeInit
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        -----------------     -------------------     ---------------------
    hp={}
    hp['lstm_hidden_size']  = {'et':64,          'etUt':64,           'ete0U0':64,           'etUte0U0':64,       }
    hp['dims_hidden_layers']= {'et':[256,256],   'etUt':[1028],       'ete0U0':[256,256],       'etUte0U0':[256,256],}
    hp['batch_size']        = {'et':256,         'etUt':256,          'ete0U0':256,          'etUte0U0':256,      }
    hp['mem_size']          = {'et':1500,        'etUt':500,          'ete0U0':1500,          'etUte0U0':1500,     }
    hp['learning_rate']     = {'et':1e-6,        'etUt':5e-6,         'ete0U0':1e-6,         'etUte0U0':1e-6,     }
    hp['num_episodes']      = {'et':50000,       'etUt':25000,        'ete0U0':50000,        'etUte0U0':50000,    }
    hp['eps_0']             = {'et':1.,          'etUt':1.,           'ete0U0':1.,           'etUte0U0':1.,       }
    hp['eps_min']           = {'et':0.1,         'etUt':0.1,          'ete0U0':0.1,          'etUte0U0':0.1,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,          'ete0U0':0.8,          'etUte0U0':0.8,      }  
    hp['target_update_freq']= {'et':100,         'etUt':100,          'ete0U0':100,          'etUte0U0':100,      }  
    HP['MetroU3_e17tborder_FixedEscapeInit'] = hp
    # MetroU3_e17t31_FixedEscapeInit
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        -----------------     -------------------     ---------------------
    hp={}
    hp['lstm_hidden_size']  = {'et':64,          'etUt':64,           'ete0U0':64,           'etUte0U0':64,       }
    hp['dims_hidden_layers']= {'et':[256,256],   'etUt':[1028],       'ete0U0':[256,256],       'etUte0U0':[256,256],}
    hp['batch_size']        = {'et':256,         'etUt':256,          'ete0U0':256,          'etUte0U0':256,      }
    hp['mem_size']          = {'et':1500,        'etUt':500,          'ete0U0':1500,          'etUte0U0':1500,     }
    hp['learning_rate']     = {'et':1e-6,        'etUt':5e-6,         'ete0U0':1e-6,         'etUte0U0':1e-6,     }
    hp['num_episodes']      = {'et':50000,       'etUt':25000,        'ete0U0':50000,        'etUte0U0':50000,    }
    hp['eps_0']             = {'et':1.,          'etUt':1.,           'ete0U0':1.,           'etUte0U0':1.,       }
    hp['eps_min']           = {'et':0.1,         'etUt':0.1,          'ete0U0':0.1,          'etUte0U0':0.1,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,          'ete0U0':0.8,          'etUte0U0':0.8,      }  
    hp['target_update_freq']= {'et':100,         'etUt':100,          'ete0U0':100,          'etUte0U0':100,      }  
    HP['MetroU3_e17t31_FixedEscapeInit'] = hp
    # MetroU3_e17t0_FixedEscapeInit
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        -----------------     -------------------     ---------------------
    hp={}
    hp['lstm_hidden_size']  = {'et':64,          'etUt':64,           'ete0U0':64,           'etUte0U0':64,       }
    hp['dims_hidden_layers']= {'et':[256,256],   'etUt':[1028],       'ete0U0':[256,256],       'etUte0U0':[256,256],}
    hp['batch_size']        = {'et':256,         'etUt':256,          'ete0U0':256,          'etUte0U0':256,      }
    hp['mem_size']          = {'et':1500,        'etUt':500,          'ete0U0':1500,          'etUte0U0':1500,     }
    hp['learning_rate']     = {'et':1e-6,        'etUt':5e-6,         'ete0U0':1e-6,         'etUte0U0':1e-6,     }
    hp['num_episodes']      = {'et':50000,       'etUt':25000,        'ete0U0':50000,        'etUte0U0':50000,    }
    hp['eps_0']             = {'et':1.,          'etUt':1.,           'ete0U0':1.,           'etUte0U0':1.,       }
    hp['eps_min']           = {'et':0.1,         'etUt':0.1,          'ete0U0':0.1,          'etUte0U0':0.1,      }
    hp['cutoff_factor']     = {'et':0.8,         'etUt':0.8,          'ete0U0':0.8,          'etUte0U0':0.8,      }  
    hp['target_update_freq']= {'et':100,         'etUt':100,          'ete0U0':100,          'etUte0U0':100,      }  
    HP['MetroU3_e17t0_FixedEscapeInit'] = hp

    return HP[world]


# PPO-RNN
def GetHyperParams_PPO_RNN(world):
    HP={}
    # Manhattan3x3_PauseFreezeWorld
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        ------------------   --------------------   ----------------------
    hp['HIDDEN_SIZE']       = {'et':128,          'etUt':128,          'ete0U0':128,          'etUte0U0':128 }
    hp['LIN_SIZE']          = {'et':[128,64], 'etUt':[128,64], 'ete0U0':[128,64], 'etUte0U0':[128,64] }
    hp['BATCH_SIZE']        = {'et':80,           'etUt':80,           'ete0U0':80,           'etUte0U0':80 }
    hp['ACTOR_LEARNING_RATE']={'et':1e-4,           'etUt':1e-4,           'ete0U0':1e-4,           'etUte0U0':1e-4 }
    hp['CRITIC_LEARNING_RATE']={'et':1e-4,           'etUt':1e-4,           'ete0U0':1e-4,           'etUte0U0':1e-4 }
    hp['BATCH_SIZE']        = {'et':80,           'etUt':80,           'ete0U0':80,           'etUte0U0':80 }
    hp['RECURRENT_SEQ_LEN'] = {'et':2,            'etUt':2,            'ete0U0':2,            'etUte0U0':2 }
    hp['ROLLOUT_STEPS']     = {'et':40,           'etUt':40,           'ete0U0':40,           'etUte0U0':40 }
    hp['PARALLEL_ROLLOUTS'] = {'et':6,            'etUt':6,            'ete0U0':6,            'etUte0U0':6 }
    hp['PATIENCE']          = {'et':200,          'etUt':200,          'ete0U0':200,          'etUte0U0':200 }
    hp['EVAL_DETERMINISTIC']= {'et':True,         'etUt':True,         'ete0U0':True,         'etUte0U0':True }    
    hp['SAMPLE_MULTIPLIER'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }    
    HP['Manhattan3x3_PauseFreezeWorld'] = hp
    # Manhattan3x3_PauseDynamicWorld
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        ------------------   --------------------   ----------------------
    hp['HIDDEN_SIZE']       = {'et':128,          'etUt':128,          'ete0U0':128,          'etUte0U0':128 }
    hp['LIN_SIZE']          = {'et':[128,64], 'etUt':[128,64], 'ete0U0':[128,64], 'etUte0U0':[128,64] } #[128,64]
    hp['BATCH_SIZE']        = {'et':80,           'etUt':80,           'ete0U0':80,           'etUte0U0':80 }
    hp['ACTOR_LEARNING_RATE']={'et':1e-4,           'etUt':1e-4,           'ete0U0':1e-4,           'etUte0U0':1e-4 }
    hp['CRITIC_LEARNING_RATE']={'et':1e-4,           'etUt':1e-4,           'ete0U0':1e-4,           'etUte0U0':1e-4 }
    hp['RECURRENT_SEQ_LEN'] = {'et':2,            'etUt':2,            'ete0U0':2,            'etUte0U0':2 }
    hp['ROLLOUT_STEPS']     = {'et':40,           'etUt':40,           'ete0U0':40,           'etUte0U0':40 }
    hp['PARALLEL_ROLLOUTS'] = {'et':6,            'etUt':6,            'ete0U0':6,            'etUte0U0':6 }
    hp['PATIENCE']          = {'et':200,          'etUt':200,          'ete0U0':200,          'etUte0U0':200 }
    hp['EVAL_DETERMINISTIC']= {'et':True,         'etUt':True,         'ete0U0':True,         'etUte0U0':True }    
    hp['SAMPLE_MULTIPLIER'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }    
    HP['Manhattan3x3_PauseDynamicWorld'] = hp
   
    # Manhattan5x5_FixedEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        ----FIXED----------   -----FIXED-----------   ----------------------
    hp['HIDDEN_SIZE']       = {'et':128,          'etUt':128,          'ete0U0':128,          'etUte0U0':128 }
    hp['LIN_SIZE']          = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['BATCH_SIZE']        = {'et':64,           'etUt':64,           'ete0U0':64,           'etUte0U0':64 }
    hp['ACTOR_LEARNING_RATE']={'et':5e-4,           'etUt':5e-4,           'ete0U0':5e-4,           'etUte0U0':1e-4 }
    hp['CRITIC_LEARNING_RATE']={'et':5e-4,           'etUt':5e-4,           'ete0U0':5e-4,           'etUte0U0':1e-4 }
    hp['RECURRENT_SEQ_LEN'] = {'et':2,            'etUt':2,            'ete0U0':2,            'etUte0U0':2 }
    hp['ROLLOUT_STEPS']     = {'et':400,          'etUt':400,          'ete0U0':400,          'etUte0U0':400 }
    hp['PARALLEL_ROLLOUTS'] = {'et':6,            'etUt':6,            'ete0U0':6,            'etUte0U0':6 }
    hp['PATIENCE']          = {'et':300,          'etUt':300,          'ete0U0':300,          'etUte0U0':300 }
    hp['EVAL_DETERMINISTIC']= {'et':True,         'etUt':True,         'ete0U0':True,         'etUte0U0':True }    
    hp['SAMPLE_MULTIPLIER'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }    
    HP['Manhattan5x5_FixedEscapeInit'] = hp
    # Manhattan5x5_VariableEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          ---FIXED-----        ----FIXED-------   -----FIXED---------   ----------------------
    hp['HIDDEN_SIZE']       = {'et':128,          'etUt':128,          'ete0U0':128,          'etUte0U0':128 }
    hp['LIN_SIZE']          = {'et':[256,256,64], 'etUt':[256,256,64], 'ete0U0':[256,256,64], 'etUte0U0':[256,256,64] }
    hp['BATCH_SIZE']        = {'et':64,           'etUt':64,           'ete0U0':64,           'etUte0U0':64 }
    hp['ACTOR_LEARNING_RATE']={'et':1e-4,           'etUt':5e-4,           'ete0U0':5e-4,           'etUte0U0':5e-4 }
    hp['CRITIC_LEARNING_RATE']={'et':1e-4,           'etUt':5e-4,           'ete0U0':5e-4,           'etUte0U0':5e-4 }
    hp['RECURRENT_SEQ_LEN'] = {'et':2,            'etUt':2,            'ete0U0':2,            'etUte0U0':2 }
    hp['ROLLOUT_STEPS']     = {'et':400,          'etUt':400,          'ete0U0':400,          'etUte0U0':400 }
    hp['PARALLEL_ROLLOUTS'] = {'et':6,            'etUt':6,            'ete0U0':6,            'etUte0U0':6 }
    hp['PATIENCE']          = {'et':300,          'etUt':300,          'ete0U0':300,          'etUte0U0':300 }
    hp['EVAL_DETERMINISTIC']= {'et':True,         'etUt':True,         'ete0U0':True,         'etUte0U0':True }    
    hp['SAMPLE_MULTIPLIER'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }    
    HP['Manhattan5x5_VariableEscapeInit'] = hp
    # Manhattan5x5_DuplicateSetA
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          ---FIXED---        ----FIXED-------   ----FIXED-----------   ------FIXED----------
    hp['HIDDEN_SIZE']       = {'et':128,          'etUt':128,          'ete0U0':128,          'etUte0U0':128 }
    hp['LIN_SIZE']          = {'et':[128,64], 'etUt':[128,64], 'ete0U0':[128,64], 'etUte0U0':[128,64] }
    hp['BATCH_SIZE']        = {'et':80,           'etUt':80,           'ete0U0':80,           'etUte0U0':80 }
    hp['ACTOR_LEARNING_RATE']={'et':5e-4,           'etUt':1e-4,           'ete0U0':5e-4,           'etUte0U0':5e-4 }
    hp['CRITIC_LEARNING_RATE']={'et':5e-4,           'etUt':1e-4,           'ete0U0':5e-4,           'etUte0U0':5e-4 }
    hp['BATCH_SIZE']        = {'et':80,           'etUt':80,           'ete0U0':80,           'etUte0U0':80 }
    hp['RECURRENT_SEQ_LEN'] = {'et':2,            'etUt':2,            'ete0U0':2,            'etUte0U0':2 }
    hp['ROLLOUT_STEPS']     = {'et':40,           'etUt':40,           'ete0U0':40,           'etUte0U0':40 }
    hp['PARALLEL_ROLLOUTS'] = {'et':6,            'etUt':6,            'ete0U0':6,            'etUte0U0':6 }
    hp['PATIENCE']          = {'et':200,          'etUt':200,          'ete0U0':200,          'etUte0U0':200 }
    hp['EVAL_DETERMINISTIC']= {'et':True,         'etUt':True,         'ete0U0':True,         'etUte0U0':True }    
    hp['SAMPLE_MULTIPLIER'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }    
    HP['Manhattan5x5_DuplicateSetA'] = hp
    # Manhattan5x5_DuplicateSetB
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          -----------        ------------------   --------------------   ----------------------
    hp['HIDDEN_SIZE']       = {'et':128,          'etUt':128,          'ete0U0':128,          'etUte0U0':128 }
    hp['LIN_SIZE']          = {'et':[128,64],     'etUt':[128,64],     'ete0U0':[128,64],     'etUte0U0':[128,64] }
    hp['BATCH_SIZE']        = {'et':64,           'etUt':64,           'ete0U0':64,           'etUte0U0':64 }
    hp['ACTOR_LEARNING_RATE']={'et':5e-4,         'etUt':5e-4,         'ete0U0':5e-4,         'etUte0U0':5e-4 }
    hp['CRITIC_LEARNING_RATE']={'et':5e-4,        'etUt':5e-4,         'ete0U0':5e-4,         'etUte0U0':5e-4 }
    hp['BATCH_SIZE']        = {'et':80,           'etUt':80,           'ete0U0':80,           'etUte0U0':80 }
    hp['RECURRENT_SEQ_LEN'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }
    hp['ROLLOUT_STEPS']     = {'et':200,           'etUt':200,           'ete0U0':200,           'etUte0U0':200 }
    hp['PARALLEL_ROLLOUTS'] = {'et':6,            'etUt':6,            'ete0U0':6,            'etUte0U0':6 }
    hp['PATIENCE']          = {'et':200,          'etUt':200,          'ete0U0':200,          'etUte0U0':200 }
    hp['EVAL_DETERMINISTIC']= {'et':False,         'etUt':False,         'ete0U0':False,         'etUte0U0':False }    
    hp['SAMPLE_MULTIPLIER'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }    
    HP['Manhattan5x5_DuplicateSetB'] = hp

    # MetroU3_e17tborder_FixedEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          --FIXED-----        ----FIXED----------   --------------------   ----------------------
    hp['HIDDEN_SIZE']       = {'et':128,          'etUt':128,          'ete0U0':128,          'etUte0U0':128 }
    hp['LIN_SIZE']          = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['BATCH_SIZE']        = {'et':64,           'etUt':64,           'ete0U0':64,           'etUte0U0':64 }
    hp['ACTOR_LEARNING_RATE']={'et':1e-4,           'etUt':5e-4,           'ete0U0':5e-4,           'etUte0U0':1e-4 }
    hp['CRITIC_LEARNING_RATE']={'et':1e-4,           'etUt':5e-4,           'ete0U0':5e-4,           'etUte0U0':1e-4 }
    hp['RECURRENT_SEQ_LEN'] = {'et':2,            'etUt':2,            'ete0U0':2,            'etUte0U0':2 }
    hp['ROLLOUT_STEPS']     = {'et':400,          'etUt':400,          'ete0U0':400,          'etUte0U0':400 }
    hp['PARALLEL_ROLLOUTS'] = {'et':6,            'etUt':6,            'ete0U0':6,            'etUte0U0':6 }
    hp['PATIENCE']          = {'et':300,          'etUt':300,          'ete0U0':300,          'etUte0U0':300 }
    hp['EVAL_DETERMINISTIC']= {'et':True,         'etUt':True,         'ete0U0':True,         'etUte0U0':True }    
    hp['SAMPLE_MULTIPLIER'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }    
    HP['MetroU3_e17tborder_FixedEscapeInit'] = hp
    # MetroU3_e17tborder_VariableEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          --FIXED-----        ----FIXED----------   --------------------   ----------------------
    hp['HIDDEN_SIZE']       = {'et':128,          'etUt':128,          'ete0U0':128,          'etUte0U0':128 }
    hp['LIN_SIZE']          = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['BATCH_SIZE']        = {'et':64,           'etUt':64,           'ete0U0':64,           'etUte0U0':64 }
    hp['ACTOR_LEARNING_RATE']={'et':1e-4,         'etUt':5e-4,         'ete0U0':5e-4,         'etUte0U0':5e-4 }
    hp['CRITIC_LEARNING_RATE']={'et':1e-4,        'etUt':5e-4,         'ete0U0':5e-4,         'etUte0U0':5e-4 }
    hp['RECURRENT_SEQ_LEN'] = {'et':2,            'etUt':2,            'ete0U0':2,            'etUte0U0':2 }
    hp['ROLLOUT_STEPS']     = {'et':400,          'etUt':400,          'ete0U0':400,          'etUte0U0':400 }
    hp['PARALLEL_ROLLOUTS'] = {'et':6,            'etUt':6,            'ete0U0':6,            'etUte0U0':6 }
    hp['PATIENCE']          = {'et':300,          'etUt':300,          'ete0U0':300,          'etUte0U0':300 }
    hp['EVAL_DETERMINISTIC']= {'et':False,        'etUt':True,         'ete0U0':True,         'etUte0U0':True }    
    hp['SAMPLE_MULTIPLIER'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }    
    HP['MetroU3_e17tborder_VariableEscapeInit'] = hp
    # MetroU3_e17t0_FixedEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          ------------        ------------------   --------------------   ----------------------
    hp['HIDDEN_SIZE']       = {'et':128,          'etUt':128,          'ete0U0':256,          'etUte0U0':128 }
    hp['LIN_SIZE']          = {'et':[256,128,64], 'etUt':[128,128,64], 'ete0U0':[1028,1028,128], 'etUte0U0':[128,128,64] }
    hp['BATCH_SIZE']        = {'et':64,           'etUt':64,           'ete0U0':128,          'etUte0U0':64 }
    hp['ACTOR_LEARNING_RATE']={'et':1e-3,         'etUt':5e-4,         'ete0U0':7e-4,         'etUte0U0':5e-4 }
    hp['CRITIC_LEARNING_RATE']={'et':1e-3,        'etUt':5e-4,         'ete0U0':7e-4,         'etUte0U0':5e-4 }
    hp['RECURRENT_SEQ_LEN'] = {'et':2,            'etUt':2,            'ete0U0':2,            'etUte0U0':2 }
    hp['ROLLOUT_STEPS']     = {'et':400,          'etUt':400,          'ete0U0':400,          'etUte0U0':400 }
    hp['PARALLEL_ROLLOUTS'] = {'et':6,            'etUt':6,            'ete0U0':6,            'etUte0U0':6 }
    hp['PATIENCE']          = {'et':300,          'etUt':300,          'ete0U0':300,          'etUte0U0':300 }
    hp['EVAL_DETERMINISTIC']= {'et':True,         'etUt':True,         'ete0U0':True,         'etUte0U0':True }    
    hp['SAMPLE_MULTIPLIER'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }    
    HP['MetroU3_e17t0_FixedEscapeInit'] = hp    
    # MetroU3_e17t31_FixedEscapeInit
    hp={}
    #                            sr='et'              sr='etUt'           sr='ete0U0'             sr='etUte0U0'
    #                          --------------        ----------------   --------------------   ----------------------
    hp['HIDDEN_SIZE']       = {'et':128,          'etUt':256,          'ete0U0':128,          'etUte0U0':128 }
    hp['LIN_SIZE']          = {'et':[256,256,64], 'etUt':[256,256,256], 'ete0U0':[256,256,64], 'etUte0U0':[256,256,64] }
    hp['BATCH_SIZE']        = {'et':64,           'etUt':64,           'ete0U0':64,           'etUte0U0':64 }
    hp['ACTOR_LEARNING_RATE']={'et':2e-4,         'etUt':2.5e-4,           'ete0U0':5e-4,           'etUte0U0':5e-4 }
    hp['CRITIC_LEARNING_RATE']={'et':2e-4,        'etUt':2.5e-4,           'ete0U0':5e-4,           'etUte0U0':5e-4 }
    hp['RECURRENT_SEQ_LEN'] = {'et':2,            'etUt':2,            'ete0U0':2,            'etUte0U0':2}
    hp['ROLLOUT_STEPS']     = {'et':200,          'etUt':200,          'ete0U0':200,          'etUte0U0':200 }
    hp['PARALLEL_ROLLOUTS'] = {'et':6,            'etUt':6,            'ete0U0':6,            'etUte0U0':6 }
    hp['PATIENCE']          = {'et':400,          'etUt':200,          'ete0U0':400,          'etUte0U0':400 }
    hp['EVAL_DETERMINISTIC']= {'et':True,         'etUt':True,         'ete0U0':True,         'etUte0U0':True }    
    hp['SAMPLE_MULTIPLIER'] = {'et':1,            'etUt':1,            'ete0U0':1,            'etUte0U0':1 }    
    HP['MetroU3_e17t31_FixedEscapeInit'] = hp    

    
    
    
    return HP[world]

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
    hp['actor']       = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['critic']      = {'et':[128,128,64], 'etUt':[128,128,64], 'ete0U0':[128,128,64], 'etUte0U0':[128,128,64] }
    hp['activation']  = {'et':act         , 'etUt':act         , 'ete0U0':act         , 'etUte0U0':act     }
    hp['num_seeds']   = {'et':5           , 'etUt':5           , 'ete0U0':5           , 'etUte0U0':5       }
    hp['total_steps'] = {'et':150000      , 'etUt':150000      , 'ete0U0':150000      , 'etUte0U0':150000  }
    hp['eval_determ'] = {'et':False       , 'etUt':True        , 'ete0U0':False       , 'etUte0U0':True    }
    hp['sampling_m'] =  {'et':4           , 'etUt':1           , 'ete0U0':4           , 'etUte0U0':1       } # if stochastic policy, sample more 
    HP['MetroU3_e17t0_FixedEscapeInit'] = hp
  
    return HP[world]    