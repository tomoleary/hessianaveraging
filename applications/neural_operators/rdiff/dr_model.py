# This file is part of the hessianaveraging package. For more information see
# https://github.com/tomoleary/hessianaveraging/
#
# hessianaveraging is free software; you can redistribute it and/or modify
# it under the terms of the Apache license. 
#
# Author: Tom O'Leary-Roseberry
# Contact: tom.olearyroseberry@utexas.edu

import os, sys
import ufl, math
import dolfin as dl
import numpy as np
sys.path.append(os.environ.get('HIPPYLIB_PATH'))
import hippylib as hp
import logging
logging.getLogger('FFC').setLevel(logging.WARNING)
logging.getLogger('UFL').setLevel(logging.WARNING)
dl.set_log_active(False)
import matplotlib.pyplot as plt
plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=20)
import matplotlib
matplotlib.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

STATE, PARAMETER, ADJOINT = 0, 1, 2

def nonlinear_diffusion_reaction_settings(settings ={}):
    settings["seed"] = 0 #Random seed
    settings["nx"] = 40 # Number of cells in each direction

    #Prior statistics
    settings["sigma"] = 3 # pointwise variance
    settings["rho"] = 0.12 # spatial correlation length

    #Anisotropy of prior samples
    settings["theta0"] = 1.0
    settings["theta1"] = 1.0
    settings["alpha"] = 0.25*math.pi

    #Likelihood specs
    settings["ntargets"] = 50
    settings["rel_noise"] = 0.02

    #Printing and saving
    settings["verbose"] = True
    settings["output_path"] = "./result/"

    #MCMC settings
    settings["number_of_samples"] = 4000
    settings["output_frequency"] = 50
    settings["step_size"] = 0.1
    settings["burn_in"] = 500
    settings["k"] = 50
    settings["p"] = 20

    # inverse problem settings
    settings['mtrue_type'] = 'logperm'
    return settings

def u_boundary(x, on_boundary):
    return on_boundary and (x[1] < dl.DOLFIN_EPS or x[1] > 1.0 - dl.DOLFIN_EPS)

def pde_varf(u, m, p):
    return ufl.exp(m) * ufl.inner(ufl.grad(u), ufl.grad(p)) * ufl.dx + u * u * u * p * ufl.dx - dl.Constant(0.0) * p * ufl.dx

def nonlinear_diffusion_reaction_model(settings):

    output_path = settings["output_path"]
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    np.random.seed(seed=settings["seed"])

    ndim = 2
    nx = settings["nx"]
    mesh = dl.UnitSquareMesh(nx, nx)
    Vh2 = dl.FunctionSpace(mesh, 'Lagrange', 2)
    Vh1 = dl.FunctionSpace(mesh, 'Lagrange', 1)
    Vh = [Vh2, Vh1, Vh2]

    if settings["verbose"]:
        print("Number of dofs: STATE={0}, PARAMETER={1}, ADJOINT={2}".format(
            Vh[STATE].dim(), Vh[PARAMETER].dim(), Vh[ADJOINT].dim()))

    u_bdr = dl.Expression("x[1]", degree=1)
    u_bdr0 = dl.Constant(0.0)
    bc = dl.DirichletBC(Vh[STATE], u_bdr, u_boundary)
    bc0 = dl.DirichletBC(Vh[STATE], u_bdr0, u_boundary)

    pde = hp.PDEVariationalProblem(Vh, pde_varf, bc, bc0, is_fwd_linear=False)

    sigma = settings["sigma"]
    rho = settings["rho"]
    delta = 1.0/(sigma*rho)
    gamma = delta*rho**2

    theta0 = settings["theta0"]
    theta1 = settings["theta1"]
    alpha = settings["alpha"]

    anis_diff = dl.CompiledExpression(hp.ExpressionModule.AnisTensor2D(), degree=1)
    anis_diff.set(theta0, theta1, alpha)

    prior = hp.BiLaplacianPrior(Vh[PARAMETER], gamma, delta, anis_diff, robin_bc=True)

    ntargets = settings["ntargets"]
    rel_noise = settings["rel_noise"]

    # Targets only on the bottom
    targets_x = np.linspace(0.1, 0.9, ntargets)
    targets_y = np.linspace(0.1, 0.5, ntargets)
    if True:
        # targets on a diametric line
        targets = np.zeros([ntargets, ndim])
        targets[:, 0] = targets_x
        targets[:, 1] = targets_y

    else:
        # on a grid
        targets = []
        for xi in x_targets:
            for yi in y_targets:
                targets.append((xi,yi))
        targets = np.array(targets)
        ntargets = targets.shape[0]

    if settings["verbose"]:
        print("Number of observation points: {0}".format(ntargets))
    misfit = hp.PointwiseStateObservation(Vh[STATE], targets)

    misfit.targets = targets
    model = hp.Model(pde, prior, misfit)

    return model

