import simdata_utils as su

configs = su.GetConfigs() # dict with pre-set configs: "Manhattan5","Manhattan11","CircGraph"
#conf=configs['Manhattan5']
conf=configs['Manhattan11']
#conf=configs['CircGraph']
#conf=configs['TKGraph']
conf['direction_north']=False

sp = su.DefineSimParameters(conf)
dirname = su.make_result_directory(sp)
register, databank, iratios = su.LoadDatafile(dirname)

dataframe=4
start_escape_node = databank[dataframe]['start_escape_route']
unit_paths = databank[dataframe]['paths']

print('unit_paths',unit_paths,'intercept_rate',iratios[dataframe])
su.PlotAgentsOnGraph(sp,[start_escape_node],unit_paths,[i for i in range(sp.L)])

k=0