import os


def default_settings():
	settings = {}

	settings['method'] = 'gd'
	settings['averaging'] = 'uni'
	settings['batch_size'] = 100
	settings['num_epochs'] = 1_000
	settings['beta_2'] = 0.5
	settings['visible_gpu'] = 0
	settings['run_seed'] = 0

	return settings


def build_string(settings):
	command = 'python ijcnn_driver.py'
	command+=' '
	for key,value in settings.items():
		command += '-'+key
		command += ' '
		command += str(value)
		command += ' '
	return command



#################################################################################
# Deterministic methods
det_methods = ['gd','newton']

for method in det_methods:
	settings = default_settings()
	settings['method'] = method 
	print(build_string(settings))
	os.system(build_string(settings))

#################################################################################
# Stochastic methods
stoch_methods = ['stoch_gd','stoch_newton']

for method in stoch_methods:
	settings = default_settings()
	settings['method'] = method 
	print(build_string(settings))
	os.system(build_string(settings))

#################################################################################
# Hessian averaging methods

# uniform averaging
settings = default_settings()
settings['method'] = 'fan'
settings['averaging'] = 'uni'
print(build_string(settings))
os.system(build_string(settings))

# exponential averaging

beta_2s = [0.1,0.25,0.5,0.9,0.99]

for beta_2 in beta_2s:
	settings = default_settings()
	settings['method'] = 'fan'
	settings['averaging'] = 'exp'
	settings['beta_2'] = beta_2
	print(build_string(settings))
	os.system(build_string(settings))



