# Gradient Descent Warmup

The gradients will continue until morale improves

## Theory questions

#### 1) What is a loss function?

#### 2) What is the loss function for SSE OLS regression?


```python

'''
1) A loss function is a function representing the error between the actual values of a target variable
and the predictions made by a model

A loss function can be used by a machine learning model to minimize error between predicted and target values

Selection of a loss function often depends on both the specific model being used and the situation in which
one is doing analysis

2) Loss function for OLS using SSE is mean square error, or loss = sum((y-f(x))**2)/n

In practice, often calculated as sum((y-f(x))**2)/(2n), so that the exponent is cancelled out when taking the derivative
and the calculations are a little easier.  
'''
```

## Calculating a loss function for a simple regression model

For this example, we'll import a series of 300 points from the data folder as an `X` feature, and another series of 300 points as a `y` target


```python

#data manip
import numpy as np

#make data
from sklearn.datasets import make_regression

#testing
from test_scripts.test_class import Test

#data import
test = Test()
X = test.load_ind('X')
y = test.load_ind('y')

#used for testing
test = Test()
```

### Calculate the change in the loss function for a simple regression model, $y = \beta  x + b$, as we step down the gradient once from an initial guess of 3 for $\beta$ and 1 for $b$


#### First, set up
- define n as 300
- define `beta` and `n` as initial guesses of 3 and 1, respectively
- assign the loss function to `loss` using `y`, `X`, `n`, `beta` and `b`
  - note: go ahead and multiply the denominator by 2 to cancel the partial derivative exponent step coming up


```python

n = 300
b = 1
beta = 3

def loss(beta, b, n):
    return sum((y-beta*X-b)**2)/(2*n)
```

### Now, calculate

- determine the partial derivatives for `beta` and `b`


```python

beta_deriv = sum((y-beta*X-b)*-X)/300
b_deriv =sum((y-beta*X-b)*-1)/300
```

- re-define `beta` and `b` by "stepping down the gradient" - ie, subtract their respective partial derivatives from themselves


```python

beta = beta - beta_deriv
b = b - b_deriv

# #used for testing
# test.save(beta, 'beta')
# test.save(b, 'b')
```

- does `y = original_beta * X + original_b` have a smaller loss function value than `y = updated_beta * X + new_b`?


```python

loss_original = loss(3,1,300)

loss_new = loss(beta, b, 300)

print(f'loss from first guess: {loss_original}, loss from updated guess: {loss_new}')
```

    loss from first guess: 942.4909304580721, loss from updated guess: 15.147998163619848


#### Now, do this 10 more times!

- Put the machinery to calculate a new `beta` and `b` you've created inside of a function
  - The function should take as inputs `beta`, `b`, `X`, and `y`
  - (`X` and `y` should have default values)
  - calculate the partial derivative of the loss function for `beta`
  - calculate the partial derivative of the loss function for `b`
  - subtract their respective partial derivatives from `beta` and `b`
  - return updated `beta` and `b`


- Put the function inside of a loop which runs 10 times

- Calculate `beta` and `b`, starting with `beta`=3 and `b`=1, after 10 "steps down the gradient"

- What does it look like the final `y = mx + b` equation should be?


```python

def calculate_descent(beta, b, X=X, y=y):
    beta_deriv = sum((y-beta*X-b)*-X)/300
    b_deriv = sum((y-beta*X-b)*-1)/300
    
    new_beta = beta - beta_deriv
    new_b = b - b_deriv
    
    return {'beta':new_beta, 'b': new_b}

beta = 3
b = 1

for epoch in range(0,10):
    results = calculate_descent(beta, b)
    beta = results['beta']
    b = results['b']

print(f'beta after 10 epochs: {beta}')
print(f'b after 10 epochs: {b}')

print(f'y = {beta} * X + {b}, baby')

#used for testing
# test.save(beta, 'beta_10')
# test.save(b, 'b_10')
```

    beta after 10 epochs: 48.34690547882362
    b after 10 epochs: 0.017270934505653077
    y = 48.34690547882362 * X + 0.017270934505653077, baby


# Bonus Round

Look at the loss function as we move from epoch to epoch

Why do we not need a learning rate in this instance?

Why might we need a learning rate in other instances?


```python

beta = 3
b = 1

losses = []
for epoch in range(0,10):
    losses.append(loss(beta, b, n))
    results = calculate_descent(beta, b)
    beta = results['beta']
    b = results['b']
    
print(losses)

'''
This particular loss function is quadratic

No matter what side of the curve we start on, if we subtract the gradient,
we subtract decreasing values

We will approach but never reach the global minimum of the function

But, we might need a learning rate to "speed up" or "slow down" the 
descent down the curve, depending on how sharp the curve is (which
is defined by the data)

This particular dataset generates a cost curve that steps down nicely, 
so we don't need an additional learning rate.

There are other loss functions that aren't quadratic curves that don't operate this way

One popular one, a "hinge" loss function, decreases linearly up to a specific value and then is 0

Obviously, subtracting the slope of that cost curve at any given point will not mean we approach but never
attain the global minimum there

And so we would need a learning rate to use gradient descent for such a loss function
'''
```
