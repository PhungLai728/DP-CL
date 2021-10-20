# DP-CL(Continual Learning with Differential Privacy)

This is the official implementation of the [Continual Learning with Differential Privacy](https://arxiv.org/pdf/2110.05223.pdf).

If you use this code or our results in your research, please cite as appropriate:

```
@article{desai2021continual,
  title={Continual Learning with Differential Privacy},
  author={Pradnya, Desai and Lai, Phung and Phan, NhatHai and Thai, My},
  journal={International Conference on Neural Information Processing},
  year={2021}
}
```


## Software Requirements

Python 3.7 is used for the current codebase.

Tensorflow 2.5

## Model configurations
In the permuted MNIST dataset, we use a fully connected network with two hidden layers of 256 hidden neurons. Given a stream of 17 tasks, the model is optimized via stochastic gradient descent with a learning rate \alpha = 0.1. In computing g_{ref}, the batch-size is set to 100 for each training task and 50 for the mini-memory block. The number of runs for each experiment is 3. The noise scale \sigma = 1 and the gradient clipping bound \beta = 0.1. In the Split CIFAR dataset, a reduced ResNet-18 with 3 times less feature maps across all the layers. The network has a final linear classifier for prediction in the  Split CIFAR dataset. The batch-size is set to 10 in each training task. Other hyperparameters, e.g., learning rate, noise scale, gradient clipping bound, etc., are the same as in the permuted MNIST dataset experiment. The number of runs for each experiment is 5.

## Experiments
The repository comes with instructions to reproduce the results in the paper or to train the model from scratch:

To reproduce the results:
+ Clone or download the folder from this repository.
+ Please find dataset on [Google Drive folder](https://drive.google.com/drive/folders/1RP22MIEFwH4jlo4Hrh7aTUKoHSJMbzhk?usp=sharing). 

+ Go to folder `DP-CL/` and Run `./replicate_results_xx.sh xx 3` where xx is the name of dataset and task that you'd like to run. 
For example:
`./replicate_results_mnist.sh MNIST 3` for MNIST,
`./replicate_results_cifar100.sh CIFAR 3` for CIFAR-100,
`./replicate_results_cifar10.sh CIFAR 3` for CIFAR-10.


## Potential issues 
If you have any issues while running the code or further information, please send email directly to the first authors of this paper (`pnd26@njit.edu` or `tl353@njit.edu`). 
