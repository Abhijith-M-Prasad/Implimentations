# AutoGrad
This is python package for automatic differentiation implimented for educational purposes hence is not optimized for training large production grade deep learning models. This project is inspired from the work of Andrej Karpathy called [micrograd](https://github.com/karpathy/micrograd), however, the intent is not the same.
- micrograd focuses on giving intutions of backpropagation while autograd focuses on taking away some abstractions of frameworks, like PyTorch hence allow user to be more mindful of what is going on in the implementation.  
Ex: zeroing gradients, updating parameters etc or even backpropagation can be implemented explicitly. 
- micrograd only supports single values only while autograd supports tensors of any dimension.