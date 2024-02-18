import engine as e
import torch

def main():
    """ runs the test cases """

    errors = 0 # error counter

    print("started test cases")

    # Create a tensor
    try:
        t = e.random.randn(3, 3)
        print("\t\033[1;32;40mpassed\033[0m test for creating a tensor ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to create a tensor ")

    # getitem from a tensor
    try:
        t = e.random.randn(3, 3)
        a = t[0]
        if e.allclose(a.data, t.data[0]): 
            print("\t\033[1;32;40mpassed\033[0m test for getitem from a tensor ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to getitem from a tensor ")

    # setitem in a tensor
    try:
        t = e.random.randn(3, 3)
        t[0, 0] = 0
        if t.data[0, 0] == 0: 
            print("\t\033[1;32;40mpassed\033[0m test for setitem in a tensor ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to setitem in a tensor \nt[0,0]=", t.data[0,0])

    # Add two tensors
    try:
        a = e.random.randn(3)
        b = e.random.randn(3, 3)
        t = a + b
        if e.allclose(t.data, a.data + b.data):
            print("\t\033[1;32;40mpassed\033[0m test for addition ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to add two tensors ")

    # addition engine vs torch
    try:
        x1 = e.random.randn(4)
        y1 = torch.tensor(x1.data, requires_grad=True)
        x2 = e.random.randn(3, 4)
        y2 = torch.tensor(x2.data, requires_grad=True)
        x3 = x1 + x2
        y3 = y1 + y2
        y3.retain_grad()
        x3.grad = e.ones_like(x3)
        x3.backward()
        y3.backward(torch.ones_like(y3))
        if e.allclose(x1.grad, y1.grad) and e.allclose(x2.grad, y2.grad):
            print("\t\033[1;32;40mpassed\033[0m addition engine vs torch")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m addition engine vs torch:\nx1.grad=", x1.grad, "\ny1.grad=", y1.grad, "\nx2.grad=", x2.grad, "\ny2.grad=", y2.grad)
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m addition engine vs torch")

    # right side addition
    try:
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        b = e.random.randn(3, 3)
        t = a + b
        print("\t\033[1;32;40mpassed\033[0m test for right side addition ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to right side addition ")
    
    # Multiply two tensors
    try:
        a = e.random.randn(3, 3)
        b = e.random.randn(3, 3)
        t = b * a
        if e.allclose(t.data, a.data * b.data):
            print("\t\033[1;32;40mpassed\033[0m test for multiplication ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to multiply two tensors:  \nt=", t, "\na=", a, "\nb=", b)

    # multiplication engine vs torch
    try:
        x1 = e.random.randn(4)
        y1 = torch.tensor(x1.data, requires_grad=True)
        x2 = e.random.randn(3, 4)
        y2 = torch.tensor(x2.data, requires_grad=True)
        x3 = x1 * x2
        y3 = y1 * y2
        y3.retain_grad()
        x3.grad = e.ones_like(x3)
        x3.backward()
        y3.backward(torch.ones_like(y3))
        if e.allclose(x1.grad, y1.grad) and e.allclose(x2.grad, y2.grad):
            print("\t\033[1;32;40mpassed\033[0m multiplication engine vs torch")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m multiplication engine vs torch:\nx1.grad=", x1.grad, "\ny1.grad=", y1.grad, "\nx2.grad=", x2.grad, "\ny2.grad=", y2.grad)
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m multiplication engine vs torch")
    # right side multiplication
    try:
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        b = e.random.randn(3, 3)
        t = b * a
        print("\t\033[1;32;40mpassed\033[0m test for right side multiplication ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to right side multiplication ")

    # matrix multiplication
    try:
        a = e.random.randn(4, 3)
        b = e.random.randn(3, 6)
        t = a @ b
        if e.allclose(t.data, a.data @ b.data):
            print("\t\033[1;32;40mpassed\033[0m test for matrix multiplication ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to matrix multiplication:  \nt=", t, "\na=", a, "\nb=", b)

    # backward matrix multiplication engine vs torch
    try:
        x1 = e.random.randn(4, 3)
        x2 = e.random.randn(3, 6)
        y1 = torch.tensor(x1.data, requires_grad=True, dtype=torch.float32)
        y2 = torch.tensor(x2.data, requires_grad=True, dtype=torch.float32)
        x3 = x1 @ x2
        y3 = y1 @ y2
        x3.grad = e.ones_like(x3)
        x3.backward()
        y3.backward(torch.ones_like(y3))
        if e.allclose(x1.grad, e.Tensor(y1.grad.numpy())) and e.allclose(x2.grad, e.Tensor(y2.grad.numpy())):
            print("\t\033[1;32;40mpassed\033[0m backward matrix multiplication engine vs torch")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m backward matrix multiplication engine vs torch :\nx1.grad=", x1.grad, "\ny1.grad=", y1.grad, "\nx2.grad=", x2.grad, "\ny2.grad=", y2.grad)
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m backward matrix multiplication engine vs torch")        

    # right side matrix multiplication
    """try:
        a = 3
        b = e.random.randn(3, 3)
        t = a @ b
        print("\tpassed\033[0m test for right side matrix multiplication")
    except:
        errors += 1
        print("\tFailed\033[0m to right side matrix multiplication")
    """
    # power
    try:
        a = e.random.randn(3, 3)
        t = a ** 2
        if e.allclose(t.data, a.data ** 2):
            print("\t\033[1;32;40mpassed\033[0m test for power ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to power:  \nt=", t, "\na=", a)

    # backpropogation for power
    try:
        t.grad = e.random.randn(3, 3)
        t.backward()
        if e.allclose(t.grad * 2 * a, a.grad):
            print("\t\033[1;32;40mpassed\033[0m backpropogation for power ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate power: \nt.grad=", t.grad, "\na.grad=", a.grad)
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to backpropogate power ")

    # transpose
    try:
        a = e.random.randn(3, 3)
        t = a.T
        if e.allclose(t.data, a.data.T):
            print("\t\033[1;32;40mpassed\033[0m test for transpose ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to transpose:  \nt=", t, "\na=", a)

    # backpropogation for transpose
    try:
        t.grad = e.random.randn(3, 3)
        t.backward()
        if e.allclose(t.grad.T, a.grad):
            print("\t\033[1;32;40mpassed\033[0m backpropogation for transpose ")
        else:
            errors += 1
            print("\t\033[1;32;40mFailed\033[0m to backpropogate transpose: \nt.grad=", t.grad, "\na.grad=", a.grad)
    except:
        errors += 1
        print("\t\033[1;32;40mFailed\033[0m to backpropogate transpose ")

    # transpose in engine vs torch
    try:
        x1 = e.random.randn(4, 3)
        y1 = torch.tensor(x1.data, requires_grad=True, dtype=torch.float32)
        x2 = x1.T 
        y2 = y1.T 
        x2.grad = e.ones_like(x2)
        x2.backward()
        y2.backward(torch.ones_like(y2))
        if e.allclose(x1.grad, y1.grad):
            print("\t\033[1;32;40mpassed\033[0m transpose in engine vs torch")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m transpose in engine vs torch")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m transpose in engine vs torch")

    # reshape
    try:
        a = e.random.randn(3, 3)
        t = a.reshape(9)
        if e.allclose(t.data, a.data.reshape(9)):
            print("\t\033[1;32;40mpassed\033[0m test for reshape ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to reshape:  \nt=", t, "\na=", a)

    # backpropogation for reshape
    try:
        t.grad = e.random.randn(9)
        t.backward()
        if e.allclose(t.grad.reshape((3, 3)), a.grad):
            print("\t\033[1;32;40mpassed\033[0m backpropogation for reshape ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate reshape: \nt.grad=", t.grad, "\na.grad=", a.grad)
    except:
        errors += 1
        print("\t\033[1;32;40mFailed\033[0m to backpropogate reshape ")

    # sum 
    try:
        a = e.random.randn(3, 3)
        t = e.sum(a)
        if e.allclose(t.data, a.data.sum()):
            print("\t\033[1;32;40mpassed\033[0m test for sum ")
    except:
        errors += 1
        print("\tFailed\033[0m to sum \nt=", t, "\na=", a)

    # backpropogation for sum
    try:
        t.grad = t
        t.backward()
        if e.allclose(t.grad, a.grad):
            print("\t\033[1;32;40mpassed\033[0m backpropogation for sum ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate sum: \nt.grad=", t.grad, "\na.grad=", a.grad)
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to backpropogate sum ")

    # sum vs torch.sum
    try:
        b = torch.tensor(a.data, requires_grad=True)
        c = torch.sum(b)
        #c.grad = torch.tensor(t.grad.data)
        c.backward(torch.tensor(t.grad.data))
        if e.allclose(a.grad, e.Tensor(b.grad.numpy())):
            print("\t\033[1;32;40mpassed\033[0m test for sum vs torch.sum ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to sum vs torch.sum:  \n a.grad=", a.grad, "\nb.grad.numpy()=", e.Tensor(b.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to sum vs torch.sum ")

    # test sum vs torch.sum over axis
    try:
        a = e.random.randn(3, 3)
        b = torch.tensor(a.data, requires_grad=True)
        c = torch.sum(b, dim=0)
        t = e.sum(a, axis=0)
        t.grad = e.ones_like(t)
        t.backward()
        c.backward(torch.ones_like(c))
        if e.allclose(a.grad, e.Tensor(b.grad.numpy())):
            print("\t\033[1;32;40mpassed\033[0m test for sum vs torch.sum over axis ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to sum vs torch.sum over axis:  \n a.grad=", a.grad, "\nb.grad.numpy()=", e.Tensor(b.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to sum vs torch.sum over axis ")

    # test sum vs torch.sum backward over axis
    try:
        a = e.random.randn(3, 3)
        b = torch.tensor(a.data, requires_grad=True)
        c = torch.sum(b, dim=0)
        t = e.sum(a, axis=0)
        t.grad = e.ones_like(t)
        t.backward()
        c.backward(torch.ones_like(c))
        if e.allclose(a.grad, e.Tensor(b.grad.numpy())):
            print("\t\033[1;32;40mpassed\033[0m test for sum vs torch.sum backward over axis ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to sum vs torch.sum backward over axis:  \n a.grad=", a.grad, "\nb.grad.numpy()=", e.Tensor(b.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to sum vs torch.sum backward over axis ")

    # test mean
    try:
        a = e.random.randn(3, 3)
        t = e.mean(a)
        if e.allclose(t.data, a.data.mean()):
            print("\t\033[1;32;40mpassed\033[0m test for mean ")
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to mean:  \nt=", t, "\na=", a)

    # backpropogation for mean
    try:
        t.grad = t
        t.backward()
        if e.allclose(t.grad/a.numel(), a.grad):
            print("\t\033[1;32;40mpassed\033[0m backpropogation for mean ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate mean: \nt.grad=", t.grad, "\na.grad=", a.grad)
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to backpropogate mean ")

    # test mean vs torch.mean
    try:
        b = torch.tensor(a.data, requires_grad=True)
        c = torch.mean(b)
        #c.grad = torch.tensor(t.grad.data)
        c.backward(torch.tensor(t.grad.data))
        if e.allclose(a.grad, e.Tensor(b.grad.numpy())):
            print("\t\033[1;32;40mpassed\033[0m test for mean vs torch.mean ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to mean vs torch.mean:  \n a.grad=", a.grad, "\nb.grad.numpy()=", e.Tensor(b.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to mean vs torch.mean ")

    # test mean vs torch.mean along axis
    try:
        a = e.random.randn(3, 3)
        t = e.mean(a, axis=1)
        b = torch.tensor(a.data, requires_grad=True)
        c = torch.mean(b, dim=1)
        t.grad = e.ones_like(t)
        t.backward()
        c.backward(torch.tensor(t.grad.data))
        if e.allclose(t, e.Tensor(c.detach().numpy())):
            print("\t\033[1;32;40mpassed\033[0m test for mean vs torch.mean along axis ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to mean vs torch.mean along axis:  \n a.grad=", a.grad, "\nb.grad.numpy()=", e.Tensor(b.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to mean vs torch.mean along axis ")

    # test mean vs torch.mean backward along axis
    try:
        a = e.random.randn(3, 3)
        t = e.mean(a, axis=1)
        b = torch.tensor(a.data, requires_grad=True)
        c = torch.mean(b, dim=1)
        c.backward(torch.ones_like(c))
        t.grad = e.ones_like(t)
        t.backward()
        if e.allclose(a.grad, e.Tensor(b.grad.numpy())):
            print("\t\033[1;32;40mpassed\033[0m test for mean vs torch.mean backward along axis ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to mean vs torch.mean backward along axis:  \n a.grad=", a.grad, "\nb.grad.numpy()=", e.Tensor(b.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to mean vs torch.mean backward along axis ")

    # test exp and backward
    try:
        x1 = e.random.randn(3, 3)
        y1 = torch.tensor(x1.data, requires_grad=True)
        x2 = e.exp(x1)
        y2 = torch.exp(y1)
        t1 = e.allclose(x2, y2.detach().numpy())
       # x2.backward(e.ones_like(x2))
        y2.backward(torch.ones_like(y2))
        t2 = e.allclose(x1.grad, y1.grad.numpy())
        if t1:
            print("\t\033[1;32;40mpassed\033[0m test for exp ")
        if t2:
            print("\t\033[1;32;40mpassed\033[0m backpropogation for exp ")
    except:
        if not t1:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to exp ")
        if not t2:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate exp ")

    # test log and backward
    try:
        x1 = e.random.randint(5, 10, size=(3, 3))
        y1 = torch.tensor(x1.data, requires_grad=True, dtype=torch.float32)
        x2 = e.log(x1)
        y2 = torch.log(y1)
        y2.retain_grad()
        t1 = e.allclose(x2, e.Tensor(y2.detach().numpy()))
        x2.grad = e.ones_like(x2)
        x2.backward()
        y2.backward(torch.ones_like(y2))
        t2 = e.allclose(x1.grad, e.Tensor(y1.grad.numpy()))
        if t1:
            print("\t\033[1;32;40mpassed\033[0m test for log ")
        if t2:
            print("\t\033[1;32;40mpassed\033[0m backpropogation for log ")
        if not t1:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to log \n x2=", x2, "\n y2=", y2)
        if not t2:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate log \n x1.grad=", x1.grad, "\n y1.grad.numpy()=", e.Tensor(y1.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to log ")

    # test max and backward
    try:
        x1 = e.random.randn(3, 3)
        y1 = torch.tensor(x1.data, requires_grad=True)
        x2 = e.max(x1)
        y2 = torch.max(y1)
        t1 = e.allclose(x2, y2.detach().numpy())
        x2.grad = e.ones_like(x2)
        x2.backward()
        y2.backward(torch.ones_like(y2))
        t2 = e.allclose(x1.grad, y1.grad.numpy())
        if t1:
            print("\t\033[1;32;40mpassed\033[0m test for max ")
        if t2:
            print("\t\033[1;32;40mpassed\033[0m backpropogation for max ")
        if not t1:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to max \n x2=", x2, "\n y2=", y2)
        if not t2:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate max \n x1.grad=", x1.grad, "\n y1.grad.numpy()=", e.Tensor(y1.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to max ")

    # test max and backward along axis
    try:
        x1 = e.random.randn(3, 3)
        y1 = torch.tensor(x1.data, requires_grad=True)
        x2 = e.max(x1, axis=0)
        y2 = torch.max(y1, dim=0)[0]
        t1 = e.allclose(x2, y2.detach().numpy())
        x2.grad = e.ones_like(x2)
        x2.backward()
        y2.backward(torch.ones_like(y2))
        t2 = e.allclose(x1.grad, y1.grad.numpy())
        if t1:
            print("\t\033[1;32;40mpassed\033[0m test for max along axis ")
        if t2:
            print("\t\033[1;32;40mpassed\033[0m backpropogation for max along axis ")
        if not t1:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to max along axis \n x2=", x2, "\n y2=", y2)
        if not t2:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate max along axis \n x1.grad=", x1.grad, "\n y1.grad.numpy()=", e.Tensor(y1.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to max along axis ")

    # test sigmoid
    try:
        x1 = e.random.randn(3, 3)
        y1 = torch.tensor(x1.data, requires_grad=True)
        x2 = e.sigmoid(x1)
        y2 = torch.sigmoid(y1)
        t1 = e.allclose(x2, y2.detach().numpy())
        x2.grad = e.ones_like(x2)
        x2.backward()
        y2.backward(torch.ones_like(y2))
        t2 = e.allclose(x1.grad, y1.grad.numpy())
        if t1:
            print("\t\033[1;32;40mpassed\033[0m test for sigmoid ")
        if t2:
            print("\t\033[1;32;40mpassed\033[0m backpropogation for sigmoid ")
        if not t1:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to sigmoid \n x2=", x2, "\n y2=", y2)
        if not t2:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate sigmoid \n x1.grad=", x1.grad, "\n y1.grad.numpy()=", e.Tensor(y1.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to sigmoid ")
    
    # test softmax along axis
    try:
        x1 = e.random.randn(3, 3)
        y1 = torch.tensor(x1.data, requires_grad=True)
        x2 = e.softmax(x1, axis=0)
        y2 = torch.softmax(y1, dim=0)
        t1 = e.allclose(x2, y2.detach().numpy())
        x2.grad = e.ones_like(x2)
        x2.backward()
        y2.backward(torch.ones_like(y2))
        t2 = e.allclose(x1.grad, y1.grad.numpy())
        if t1:
            print("\t\033[1;32;40mpassed\033[0m test for softmax ")
        if t2:
            print("\t\033[1;32;40mpassed\033[0m backpropogation for softmax ")
        if not t1:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to softmax \n x2=", x2, "\n y2=", y2)
        if not t2:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate softmax \n x1.grad=", x1.grad, "\n y1.grad.numpy()=", e.Tensor(y1.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to softmax ")


    # test tanh
    try:
        x1 = e.random.randn(3, 3)
        y1 = torch.tensor(x1.data, requires_grad=True)
        x2 = e.tanh(x1)
        y2 = torch.tanh(y1)
        t1 = e.allclose(x2, y2.detach().numpy())
        x2.grad = e.ones_like(x2)
        x2.backward()
        y2.backward(torch.ones_like(y2))
        t2 = e.allclose(x1.grad, y1.grad.numpy())
        if t1:
            print("\t\033[1;32;40mpassed\033[0m test for tanh ")
        if t2:
            print("\t\033[1;32;40mpassed\033[0m backpropogation for tanh ")
        if not t1:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to tanh \n x2=", x2, "\n y2=", y2)
        if not t2:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate tanh \n x1.grad=", x1.grad, "\n y1.grad.numpy()=", e.Tensor(y1.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to tanh ")

    # test relu
    try:
        x1 = e.random.randn(3, 3)
        y1 = torch.tensor(x1.data, requires_grad=True)
        x2 = e.relu(x1)
        y2 = torch.relu(y1)
        t1 = e.allclose(x2, y2.detach().numpy())
        x2.grad = e.ones_like(x2)
        x2.backward()
        y2.backward(torch.ones_like(y2))
        t2 = e.allclose(x1.grad, y1.grad.numpy())
        if t1:
            print("\t\033[1;32;40mpassed\033[0m test for relu ")
        if t2:
            print("\t\033[1;32;40mpassed\033[0m backpropogation for relu ")
        if not t1:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to relu \n x2=", x2, "\n y2=", y2)
        if not t2:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backpropogate relu \n x1.grad=", x1.grad, "\n y1.grad.numpy()=", e.Tensor(y1.grad.numpy()))
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to relu ")

    # test backward in engine vs torch
    try:
        x1 = e.random.randn(3, 4)
        y1 = torch.tensor(x1.data, requires_grad=True)
        x2 = e.random.randn(4, 5)
        y2 = torch.tensor(x2.data, requires_grad=True)
        x3 = x1 @ x2
        y3 = y1 @ y2
        y3.retain_grad()
        x4 = e.sum(x3)
        x4.grad = e.ones_like(x4)
        x4.backward()
        y4 = torch.sum(y3)
        y4.backward(torch.ones_like(y4))
        t1 = e.allclose(x1.grad, y1.grad.numpy())
        t2 = e.allclose(x2.grad, y2.grad.numpy())
        t3 = e.allclose(x3.grad, y3.grad.numpy())
        if t1 and t2 and t3:
            print("\t\033[1;32;40mpassed\033[0m test for backward in engine vs torch ")
        else:
            errors += 1
            print("\t\033[1;31;40mFailed\033[0m to backward: test result:  \nt1 = ", t1, "\nt2 = ", t2, "\nt3 = ", t3)
    except:
        errors += 1
        print("\t\033[1;31;40mFailed\033[0m to backward in engine vs torch ")

    if errors == 0:
        st = "\t\033[1;32;40mTest completed successfully with no errors\033[0m "
    else:
        st = f"\t\033[1;31;40mTest failed with {errors} errors\033[0m "

    print("Report:")
    print(st)
if __name__ == "__main__":
    main()