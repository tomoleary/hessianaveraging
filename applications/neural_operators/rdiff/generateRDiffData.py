# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import math
import numpy as np
import matplotlib.pyplot as plt
import dolfin as dl
import sys, os
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
sys.path.append(os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

sys.path.append('../../')

from dr_model import nonlinear_diffusion_reaction_model, nonlinear_diffusion_reaction_settings


formulation = 'pointwise'

nsamples = 10000
n_samples_pod = 100
pod_rank = 200

################################################################################
# Set up the model

settings = nonlinear_diffusion_reaction_settings()

model = nonlinear_diffusion_reaction_model(settings)


################################################################################
# Generate training data
mesh = model.problem.Vh[hp.STATE].mesh()

if formulation=='full_state':
    q_trial = dl.TrialFunction(model.problem.Vh[hp.STATE])
    q_test = dl.TestFunction(model.problem.Vh[hp.STATE])

    M = dl.PETScMatrix(mesh.mpi_comm())
    dl.assemble(dl.inner(q_trial,q_test)*dl.dx, tensor=M)

    B = hf.StateSpaceIdentityOperator(M, use_mass_matrix=False)

    output_basis = None
    compute_derivatives = (1,0)

elif formulation=='pointwise':
    B = model.misfit.B

    dQ = settings['ntargets']
    output_basis = np.eye(dQ)
    compute_derivatives = (1,0)
else:
    raise


observable = hf.LinearStateObservable(model.problem,B)
prior = model.prior

dataGenerator = hf.DataGenerator(observable,prior)



data_dir = 'data/rdiff_test/'
data_dir+= formulation
data_dir+='/'

if formulation=='full_state':
    dataGenerator.two_step_generate(nsamples, n_samples_pod = n_samples_pod, derivatives = compute_derivatives,\
                                                    pod_rank = pod_rank, data_dir = data_dir)
elif formulation=='pointwise':
    dataGenerator.generate(nsamples, derivatives = compute_derivatives,output_basis = output_basis, data_dir = data_dir)
else:
    raise




