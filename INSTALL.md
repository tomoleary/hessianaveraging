		      ___           ___           ___           ___     
		     /__/\         /  /\         /  /\         /  /\    
		     \  \:\       /  /:/_       /  /:/_       /  /:/_   
		      \__\:\     /  /:/ /\     /  /:/ /\     /  /:/ /\  
		  ___ /  /::\   /  /:/ /:/_   /  /:/ /::\   /  /:/ /::\ 
		 /__/\  /:/\:\ /__/:/ /:/ /\ /__/:/ /:/\:\ /__/:/ /:/\:\
		 \  \:\/:/__\/ \  \:\/:/ /:/ \  \:\/:/~/:/ \  \:\/:/~/:/
		  \  \::/       \  \::/ /:/   \  \::/ /:/   \  \::/ /:/ 
		   \  \:\        \  \:\/:/     \__\/ /:/     \__\/ /:/  
		    \  \:\        \  \::/        /__/:/        /__/:/   
		     \__\/         \__\/         \__\/         \__\/    
		               ___                        ___           
		              /  /\          ___         /  /\          
		             /  /::\        /__/\       /  /:/_         
		            /  /:/\:\       \  \:\     /  /:/ /\        
		           /  /:/~/::\       \  \:\   /  /:/_/::\       
		          /__/:/ /:/\:\  ___  \__\:\ /__/:/__\/\:\      
		          \  \:\/:/__\/ /__/\ |  |:| \  \:\ /~~/:/      
		           \  \::/      \  \:\|  |:|  \  \:\  /:/       
		            \  \:\       \  \:\__|:|   \  \:\/:/        
		             \  \:\       \__\::::/     \  \::/         
		              \__\/           ~~~~       \__\/                               
					
					
		Hessian-Averaged Newton Methods for Stochastic Optimization



## Installation instruction and useage

If using NVIDIA GPUs, make sure that any relevant CUDA paths are set first

### conda 

	conda env create -f environment.yml

If the environment fails to build from these instructions, I suggest trying the following sequence, as it worked as recently as July 9, 2024 on a machine with one NVIDIA L40S GPU. 

```bash
conda create -n hessavg python=3.9.16
pip install --upgrade pip
pip install --upgrade "jax[cuda11_pip]==0.4.24" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install torch  torchvision matplotlib flax==0.6.9 optax==0.1.7 scipy==1.11.4 orbax-checkpoint==0.4.8
```

## Dependencies

The code that accompanies the manuscript was developed in 2022-2023 and builds against older versions of `flax`, `optax` and related repos such as `orbax.checkpoint`. The first version of this repo contains the code that was used to generate the experiments used in the manuscript. 

In the future, I plan to update the code to work with more modern neural network codes in `jax`. 