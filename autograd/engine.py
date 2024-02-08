# imports
import numpy as np

class Tensor:
    """ represents matrix as a collection of node in computation graph """

    def __init__(self, data:[list|np.ndarray], dtype=np.float32, requires_grad=True):
        """
        Initializes an instance of the tensor with the given data and data type.

        Parameters:
            data (list, ndarray): A list or ndarray of data values.
            dtype (str, optional): The data type of the values. Defaults to 'float32'.

        Returns:
            None
        """
        self.data = data if isinstance(data, np.ndarray) else np.array(data, dtype=dtype)
        self.grad = None
        self.requires_grad = requires_grad
        if requires_grad:
            self.grad = Tensor(np.zeros_like(self.data), requires_grad=False)
        self._children = set()
        self._backward = lambda: None
        self.shape = self.data.shape

    def __getitem__(self, index):
        """
        Get an item from the tensor.

        Parameters:
            index (int): The index of the item to get.

        Returns:
            Tensor: The item at the given index.
        """
        return Tensor(self.data[index])
    
    def __setitem__(self, index, value):
        """
        Set an item in the tensor.

        Parameters:
            index (int): The index of the item to set.
            value (Tensor): The value to set the item to.

        Returns:
            None
        """
        self.data[index] = value

    def __len__(self):
        """
        Get the length of the tensor.

        Returns:
            int: The length of the tensor.
        """
        return len(self.data)

    def __repr__(self):
        """
        Return a string representation of the object.
        """
        r = self.data.__repr__()
        return f"Tensor{r[5:-1]}, dtype={self.data.dtype}, grad={self.requires_grad})"
    
    def __type__(self):
        """
        Return the type of the object.
        """
        return Tensor
    
    def backward(self, gradient=None):
        """
        Perform backward propagation to compute gradients.
        """
        topo = []
        visited = set()
        def build_graph(v):
            if v not in visited:
                topo.append(v)
                visited.add(v)
                for child in v._children:
                    build_graph(child)
        topo.reverse()
        build_graph(self)
        for node in topo:
            try:
                node._backward()
            except:
                pass
            #node.check_grad_shape() 
            
        
    def check_grad_shape(self):
        """
        Check if the shape of the gradient matches the shape of the tensor.
        """
        if self.grad.shape != self.shape:
            raise ValueError(f"Gradient shape {self.grad.shape} does not match tensor shape {self.data.shape}")
    
    def __add__(self, other:list):
        """
        Add two tensors together and return the result.

        Parameters:
            other (list or Tensor): The tensor to be added to self.

        Returns:
            Tensor: The sum of self and other.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data)
        out._children = {self, other}

        def _backward(gradient=None):
            """
            Add the gradient of `out` to the `grad` attribute of `self` and `other`.
            """
            gradient = out.grad if gradient is None else gradient
            t1 = gradient.shape != self.shape
            t2 = gradient.shape != other.shape
            if t1:
                other.grad.data += gradient.data
                # handle broadcasting with grad distributing property of add
                fill = np.sum(gradient.data)/self.numel()
                self.grad.data += ones_like(self).data * fill
            elif t2:
                self.grad.data += gradient.data
                # handle broadcasting with grad distributing property of add
                fill = np.sum(gradient.data)/other.numel()
                other.grad.data += ones_like(other).data * fill

        out._backward = _backward

        return out
    
    def __radd__(self, other:list):
        """
        Method to implement the right-side addition with a list, by calling the __add__ method.
        """
        return self.__add__(other)
    
    def __mul__(self, other:list):
        """
        Multiply the tensor element-wise with another tensor or a list-like object.

        Parameters:
            other (Tensor or list-like): The tensor or list-like object to be multiplied element-wise with the current tensor.

        Returns:
            Tensor: A new tensor object containing the element-wise product of the current tensor and the other tensor or list-like object.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data)
        out._children = {self, other}

        def _backward(gradient=None):
            """
            A function that performs backward propagation to compute gradients.
            """
            gradient = out.grad if gradient is None else gradient
            # Handle broadcasting if the gradient shape doesn't match the input shape
            t1 = gradient.shape != self.shape
            t2 = gradient.shape != other.shape
            self.grad.data = self.grad.data + gradient.data * other.data
            other.grad.data = other.grad.data + gradient.data * self.data

            if t1:
                ax = find_broadcasting_axes(self.data, gradient.data)
                print(f"t1={t1}, t2={t2}, ax={ax}")
                self.grad.data = np.sum(self.grad.data, axis=ax)
            elif t2:
                ax = find_broadcasting_axes(other.data, gradient.data)
                other.grad.data = np.sum(other.grad.data, axis=ax)
        
        out._backward = _backward

        return out
    
    def __rmul__(self, other:list):
        """
        Method to implement the right-side multiplication with a list, by calling the __mul__ method.
        """
        return self.__mul__(other)
    
    def __sub__(self, other:list):
        """
        Subtract the tensor element-wise with another tensor or a list-like object.

        Parameters:
            other (Tensor or list-like): The tensor or list-like object to be subtracted element-wise with the current tensor.

        Returns:
            Tensor: A new tensor object containing the element-wise subtraction of the current tensor and the other tensor or list-like object.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self + (-other)
    
    def __rsub__(self, other:list):
        """
        Method to implement the right-side subtraction with a list, by calling the __sub__ method.
        """
        return self.__sub__(other)
    
    def __truediv__(self, other:list):
        """
        Divide the tensor element-wise with another tensor or a list-like object.

        Parameters:
            other (Tensor or list-like): The tensor or list-like object to be divided element-wise with the current tensor.

        Returns:
            Tensor: A new tensor object containing the element-wise division of the current tensor and the other tensor or list-like object.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        return self * (other ** -1)

    def __rtruediv__(self, other:list):
        """
        Method to implement the right-side division with a list, by calling the __truediv__ method.
        """
        return self.__truediv__(other)
    
    def __pow__(self, other:list):
        """
        Raise the tensor element-wise to the power of another tensor or a list-like object.

        Parameters:
            other (Tensor or list-like): The tensor or list-like object to be raised to the power of the current tensor.

        Returns:
            Tensor: A new tensor object containing the element-wise power of the current tensor and the other tensor or list-like object.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data ** other.data)
        out._children = {self, other}

        def _backward(gradient=None):
            """
            A function that performs backward propagation to compute gradients.
            """
            gradient = out.grad if gradient is None else gradient
            self.grad.data += other.data * (self.data ** (other.data - 1)) * out.grad.data
        
        out._backward = _backward

        return out
    
    def __rpow__(self, other:list):
        """
        Method to implement the right-side power with a list, by calling the __pow__ method.
        """
        return self.__pow__(other)
    
    def __neg__(self):
        """
        Return the negative of the tensor.
        """
        return self * -1

    def transpose(self):
        """
        Transpose the data and create a new Tensor. Define a backward function
        to calculate gradient with respect to the input.
        """
        out = Tensor(self.data.T)
        out._children = {self}

        def _backward(gradient=None):
            """
            A function that performs backward propagation to compute gradients.
            """
            gradient = out.grad if gradient is None else gradient
            self.grad.data += gradient.data.T
        
        out._backward = _backward

        return out

    @property
    def T(self):
        """
        Return the transpose of the object.
        """
        return self.transpose()
    
    @property
    def mT(self):
        """
        Return the matrix transpose of the object.
        """
        l = len(self.data.shape)
        ls = list(range(l-2))
        ls.append(l-1)
        ls.append(l-2)
        out = Tensor(self.data.transpose(*ls))
        out._children = {self}

        def _backward(gradient=None):
            """
            A function that performs backward propagation to compute gradients.
            """
            gradient = out.grad if gradient is None else gradient
            self.grad.data += gradient.data.transpose(*ls)

        out._backward = _backward
        return out


    def __matmul__(self, other:list):
        """
        Perform matrix multiplication with another Tensor object.

        Parameters:
            other (Tensor): Another Tensor object or a list to be converted to a Tensor.

        Returns:
            Tensor: A new Tensor object resulting from the matrix multiplication.
        """
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data)
        out._children = {self, other}

        def _backward(gradient=None):
            """
            A function that performs backward propagation to compute gradients.
            """
            gradient = out.grad if gradient is None else gradient
            t1 = len(gradient.data.shape) != len(self.data.shape)
            t2 = len(gradient.data.shape) != len(other.data.shape)
            if t1:
                s = np.sum(gradient.data @ other.mT.data, axis=tuple(range(len(gradient.data.shape)-len(self.data.shape))))
                self.grad.data = self.grad.data + s
                other.grad.data = other.grad.data + self.mT.data @ gradient.data
            elif t2:
                s = np.sum(self.mT.data @ gradient.data, axis=tuple(range(len(gradient.data.shape)-len(other.data.shape))))
                self.grad.data = self.grad.data + gradient.data @ other.mT.data 
                other.grad.data = other.grad.data + s
            else:
                self.grad.data = self.grad.data + gradient.data @ other.mT.data
                other.grad.data = other.grad.data + self.mT.data @ gradient.data
        
        out._backward = _backward

        return out
    
    def __rmatmul__(self, other:list):
        """
        Method to implement the right-side matrix multiplication with a list, by calling the __matmul__ method.
        """
        return self.__matmul__(other)
    
    def reshape(self, shape):
        """
        Reshape the data and create a new Tensor. Define a backward function
        to calculate gradient with respect to the input.
        """
        out = Tensor(self.data.reshape(shape))
        out._children = {self}

        def _backward(gradient=None):
            """
            A function that performs backward propagation to compute gradients.
            """
            gradient = out.grad if gradient is None else gradient
            self.grad.data += gradient.data.reshape(self.data.shape)
        
        out._backward = _backward
        return out
    
    def numel(self):
        """
        Return the number of elements in the tensor.
        """
        return self.data.size


#--------------------------------------------------------------------------------------------------------
# Random

class Random:
    def __init__(self, seed=None):
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
    def randn(self, *args):
        """
        Return a sample (or samples) from the “standard normal” distribution.
        """
        return Tensor(np.random.randn(*args))

    def randint(self, low, high=None, size=None, dtype=int):
        """
        Return random integers from low (inclusive) to high (exclusive).
        """
        return Tensor(np.random.randint(low, high, size, dtype))

    def uniform(self, low=0.0, high=1.0, size=None):
        """
        Draw samples from a uniform distribution.
        """
        return Tensor(np.random.uniform(low, high, size))

    def normal(self, loc=0.0, scale=1.0, size=None):
        """
        Draw random samples from a normal (Gaussian) distribution.
        """
        return Tensor(np.random.normal(loc, scale, size))

    def choice(self, a, size=None, replace=True, p=None):
        """
        Generates a random sample from a given 1-D array.
        """
        return Tensor(np.random.choice(a, size, replace, p))
    
    def shuffle(self, a):
        """
        Shuffle array or sequence in-place.
        """
        return Tensor(np.random.shuffle(a))
    
    def permutation(self, a):
        """
        Return a random permutation of the elements of a.
        """
        return Tensor(np.random.permutation(a))
    
    # randn_like
    def randn_like(self, tensor):
        """
        Return a sample (or samples) from the “standard normal” distribution with the same
        shape as the given tensor.
        """
        shape = tensor.data.shape
        return Tensor(np.random.randn(shape[0], shape[1])) 
    
    # uniform_like
    def uniform_like(self, tensor, low=0.0, high=1.0):
        """
        Draw samples from a uniform distribution with the same shape as the given tensor.
        """
        return Tensor(np.random.uniform(low, high, tensor.data.shape))
    
    # normal_like
    def normal_like(self, tensor, loc=0.0, scale=1.0):
        """
        Draw random samples from a normal (Gaussian) distribution with the same shape as the given tensor.
        """
        return Tensor(np.random.normal(loc, scale, tensor.data.shape))
    
    # choice_like
    def choice_like(self, tensor, a):
        """
        Generates a random sample from a given 1-D array with the same shape as the given tensor.
        """
        return Tensor(np.random.choice(a, tensor.data.shape))
    
random = Random()
@property
def seed(seed):
    """
    Property decorator for the seed function.
    """
    random.seed = seed

#-------------------------------------------------------------------------------------------------
# Initialization
def ones(shape):
    """
    Create a new tensor filled with ones with the given shape. 

    Parameters:
    shape (tuple): The shape of the new tensor.

    Returns:
    Tensor: A new tensor filled with ones.
    """
    return Tensor(np.ones(shape))

def ones_like(tensor):
    """
    Create a new tensor filled with ones with the same shape as the given tensor.

    Parameters:
    tensor (Tensor): The tensor to create the new tensor with.

    Returns:
    Tensor: A new tensor filled with ones with the same shape as the given tensor.
    """
    return Tensor(np.ones_like(tensor.data))

def zeros(shape):
    """
    Create a new tensor filled with zeros with the given shape.

    Parameters:
    shape (tuple): The shape of the new tensor.

    Returns:
    Tensor: A new tensor filled with zeros.
    """
    return Tensor(np.zeros(shape))

def zeros_like(tensor):
    """
    Create a new tensor filled with zeros with the same shape as the given tensor.

    Parameters:
    tensor (Tensor): The tensor to create the new tensor with.

    Returns:
    Tensor: A new tensor filled with zeros with the same shape as the given tensor.
    """
    return Tensor(np.zeros_like(tensor.data))

def arange(start, stop=None, step=1, dtype=None):
    """
    Return evenly spaced values within a given interval.

    Parameters:
    start (int): The start of the interval.
    stop (int): The end of the interval.
    step (int): The step between each value.
    dtype (type): The data type of the returned array.

    Returns:
    Tensor: An array of evenly spaced values within the interval.
    """
    return Tensor(np.arange(start, stop, step, dtype))

#--------------------------------------------------------------------------------------------------------
# Operations
def allclose(x, y, rtol=1e-5, atol=1e-5):
    return np.allclose(x.data, y.data, rtol=rtol, atol=atol)

def sum(tensor, axis=None, keepdims=False):
    """
    Sum all the elements in the tensor over the specified axis or over all axes if axis is None.

    Parameters:
    tensor (Tensor): The tensor to sum.
    axis (int, optional): The axis to sum over. Defaults to None.
    keepdims (bool, optional): Whether to keep the dimensions or not. Defaults to False.

    Returns:
    Tensor: The sum of the elements in the tensor.
    """
    def _backward(gradient=None):
        """
        A function that performs backward propagation to compute gradients.
        """
        gradient = out.grad if gradient is None else gradient
        if axis is not None:
            el = tensor.shape[axis]
            tensor.grad.data = np.repeat(np.expand_dims(gradient.data, axis), el, axis=axis)
        else:
            tensor.grad.data += gradient.data

    out = Tensor(np.sum(tensor.data, axis=axis, keepdims=keepdims))
    out._children = {tensor}
    out._backward = _backward
    return out

def mean(tensor, axis=None, keepdims=False):
    """
    Calculate the mean of all the elements in the tensor over the specified axis or over all axes if axis is None.

    Parameters:
    tensor (Tensor): The tensor to calculate the mean.
    axis (int, optional): The axis to sum over. Defaults to None.
    keepdims (bool, optional): Whether to keep the dimensions or not. Defaults to False.

    Returns:
    Tensor: The mean of the elements in the tensor.
    """
    def _backward(gradient=None):
        """
        A function that performs backward propagation to compute gradients.
        """
        gradient = out.grad if gradient is None else gradient
        if axis is not None:
            el = tensor.shape[axis]
            grad = gradient.data/el
            tensor.grad.data = np.repeat(np.expand_dims(grad.data, axis), el, axis=axis)
        else:
            tensor.grad += gradient/tensor.numel()

    out = Tensor(np.mean(tensor.data, axis=axis, keepdims=keepdims))
    out._children = {tensor}
    out._backward = _backward
    return out

def exp(tensor):
    """
    Calculate the exponential of all the elements in the tensor.

    Parameters:
    tensor (Tensor): The tensor to calculate the exponential.

    Returns:
    Tensor: The exponential of the elements in the tensor.
    """
    out = Tensor(np.exp(1))**tensor
    def _backward(gradient=None):
        """
        A function that performs backward propagation to compute gradients.
        """
        gradient = out.grad if gradient is None else gradient
        tensor.grad.data += gradient.data * out.data
    out._backward = _backward
    out._children = {tensor}
    return out

def log(tensor):
    """
    Calculate the natural logarithm of all the elements in the tensor.

    Parameters:
    tensor (Tensor): The tensor to calculate the natural logarithm.

    Returns:
    Tensor: The natural logarithm of the elements in the tensor.
    """
    out = Tensor(np.log(tensor.data))
    out._children = {tensor}
    def _backward(gradient=None):
        """
        A function that performs backward propagation to compute gradients.
        """
        gradient = out.grad if gradient is None else gradient
        tensor.grad.data = tensor.grad.data + gradient.data / tensor.data
    out._backward = _backward
    return out

def where(condition, x, y):
    """
    Return the elements of x or y depending on the condition.

    Parameters:
    condition (Tensor): The condition tensor.
    x (Tensor): The tensor to return if the condition is true.
    y (Tensor): The tensor to return if the condition is false.

    Returns:
    Tensor: The elements of x or y depending on the condition.
    """
    return Tensor(np.where(condition.data, x.data, y.data))

def max(tensor, axis=None):
    """
    Find the maximum value in the tensor over the specified axis or over all axes if axis is None.

    Parameters:
    tensor (Tensor): The tensor to find the maximum value in.
    axis (int, optional): The axis to find the maximum value over. Defaults to None.

    Returns:
    Tensor: The maximum value in the tensor.
    """
    ans = np.max(tensor.data, axis=axis)
    out = Tensor(ans)
    out._children = {tensor}
    def _backward(gradient=None):
        """
        A function that performs backward propagation to compute gradients.
        """
        gradient = out.grad if gradient is None else gradient
        fill = np.zeros_like(tensor.data)
        l = np.where(tensor.data == ans)
        fill[l] = 1
        tensor.grad.data = tensor.grad.data + gradient.data * fill

    out._backward = _backward
            
    return out

#--------------------------------------------------------------------------------------------------------
# Utilities
def find_broadcasting_axes(arr1, arr2):
    # Reverse the shapes of the arrays
    shape1 = arr1.shape[::-1]
    shape2 = arr2.shape[::-1]
    
    # Initialize the list to store broadcasting axes
    broadcasting_axes = []
    l1 = len(shape1)
    l2 = len(shape2)

    # Find the maximum number of dimensions
    max_dims = np.max([l1, l2]).item()

    # Iterate over the dimensions
    for i in range(max_dims):
        dim1 = shape1[i] if i < l1 else 1
        dim2 = shape2[i] if i < l2 else 1

        # Check if the dimensions are compatible
        if dim1 != dim2 and dim1 != 1 and dim2 != 1:
            raise ValueError("Arrays are not broadcastable.")
        
        # Determine the broadcasting axes
        if dim1 == 1:
            broadcasting_axes.append(i - 1)
        #elif dim2 == 1:
        #    broadcasting_axes.append(i)

    return tuple(broadcasting_axes)
