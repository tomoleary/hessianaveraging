import os, sys
import dolfin as dl
import math
import ufl
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla 
import scipy.linalg 

import matplotlib.pyplot as plt
import math
import time

sys.path.append( os.environ.get('HIPPYLIB_PATH', "...") )
import hippylib as hp

sys.path.append( os.environ.get('HIPPYFLOW_PATH'))
import hippyflow as hf

sys.path.append('../')

from dr_model import nonlinear_diffusion_reaction_model, nonlinear_diffusion_reaction_settings

################################################################################
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('-data_dir', '--data_dir', type=str, default='../data/rdiff_data/', help="Where to save")
parser.add_argument('-basis_type', '--basis_type', type=str, default='as', help="pod as or kle")
parser.add_argument('-rank', '--rank', type=int, default=400, help="Active subspace rank")
parser.add_argument('-oversample', '--oversample', type=int, default=10, help="Active subspace oversample")
parser.add_argument('-ndata', '--ndata', type=int, default=800, help="Number of samples")

args = parser.parse_args()

data_dir = args.data_dir

################################################################################
# Parameters

rank = args.rank
oversample = args.oversample

################################################################################
# Set up the model

################################################################################
# Set up the model


settings = nonlinear_diffusion_reaction_settings()

model = nonlinear_diffusion_reaction_model(settings)

prior = model.prior


Vh = model.problem.Vh


assert dl.MPI.comm_world.size == 1, print('Not thought out in other cases yet')


if args.basis_type.lower() == 'pod':
	data_dir = args.data_dir
	all_data = np.load(data_dir+'mq_data.npz')

	u_data = all_data['q_data'][:args.ndata]

	POD = hf.PODProjectorFromData(Vh)

	d_POD, phi, Mphi, u_shift = POD.construct_subspace(u_data,rank)

	check_orth = True
	if check_orth:
		PsistarPsi = Mphi.T@phi
		orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
		print('||Psi^*Psi - I|| = ',orth_error)
		assert orth_error < 1e-5

	np.save('POD_basis',phi)
	np.save('POD_d',d_POD)
	np.save('POD_projector',Mphi)
	np.save('POD_shift',u_shift)

	fig, ax = plt.subplots()
	ax.semilogy(np.arange(len(d_POD)), d_POD)

	ax.set(xlabel='index', ylabel='eigenvalue',
		   title='POD spectrum')
	ax.grid()

	fig.savefig("POD_eigenvalues.pdf")

elif args.basis_type.lower() == 'kle':
	KLE = hf.KLEProjector(prior)
	KLE.parameters['rank'] = rank
	KLE.parameters['oversampling'] = oversample
	KLE.parameters['save_and_plot'] = False

	d_KLE, kle_basis, kle_projector = KLE.construct_input_subspace()

	input_basis = hf.mv_to_dense(kle_basis)
	input_projector = hf.mv_to_dense(kle_projector)


	check_orth = True
	if check_orth:
		PsistarPsi = input_projector.T@input_basis
		orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
		print('||Psi^*Psi - I|| = ',orth_error)
		assert orth_error < 1e-5

	np.save('KLE_basis',input_basis)
	np.save('KLE_d',d_KLE)
	np.save('KLE_projector',input_projector)

elif args.basis_type.lower() == 'as':
	################################################################################
	# Load the data

	data_dir = args.data_dir
	all_data = np.load(data_dir+'mq_data.npz')
	all_data = np.load(data_dir+'mq_data.npz')
	JTPhi_data = np.load(data_dir+'JstarPhi_data.npz')

	m_data = all_data['m_data'][:args.ndata]
	q_data = all_data['q_data'][:args.ndata]
	PhiTJ_data = np.transpose(JTPhi_data['JstarPhi_data'], (0,2,1))[:args.ndata]

	print('m_data.shape = ',m_data.shape)
	print('q_data.shape = ',q_data.shape)
	print('PhiTJ_data.shape = ',PhiTJ_data.shape)


	################################################################################
	# Instance JTJ operator 
	print('Loading JTJ')
	JTJ_operator = hf.MeanJTJfromDataOperator(PhiTJ_data,prior)
	# Set up the Gaussian random
	m_vector = dl.Vector()
	JTJ_operator.init_vector_lambda(m_vector,0)
	Omega = hp.MultiVector(m_vector,rank+oversample)
	hp.parRandom.normal(1.,Omega)

	t0 = time.time()
	print('Beginning doublePassG')
	if hasattr(prior, "R"):
		d_GN, V_GN = hp.doublePassG(JTJ_operator,\
			prior.R, prior.Rsolver, Omega,rank,s=1)
	else:
		d_GN, V_GN = hp.doublePassG(JTJ_operator,\
			prior.Hlr, prior.Hlr, Omega,rank,s=1)

	print('doublePassG took ',time.time() - t0,'s')

	input_basis = hf.mv_to_dense(V_GN)

	# Compute the projector RV_r from the basis
	RV_GN = hp.MultiVector(V_GN[0],V_GN.nvec())
	RV_GN.zero()
	hp.MatMvMult(prior.R,V_GN,RV_GN)

	input_projector = hf.mv_to_dense(RV_GN)

	check_orth = True
	if check_orth:
		PsistarPsi = input_projector.T@input_basis
		orth_error = np.linalg.norm(PsistarPsi - np.eye(PsistarPsi.shape[0]))
		print('||Psi^*Psi - I|| = ',orth_error)
		assert orth_error < 1e-5

	np.save('AS_input_basis',input_basis)
	np.save('AS_d_GN',d_GN)
	np.save('AS_input_projector',input_projector)

	fig, ax = plt.subplots()
	ax.semilogy(np.arange(len(d_GN)), d_GN)

	ax.set(xlabel='index', ylabel='eigenvalue',
		   title='GEVP JJT spectrum')
	ax.grid()

	fig.savefig("JJT_eigenvalues.pdf")

else: 
	raise






