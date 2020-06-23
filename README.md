# Gradient Descent Warmup

The gradients will continue until morale improves

## Theory questions

#### 1) What is a loss function?

#### 2) What is the loss function for SSE OLS regression?


```python
#Write your answers here
```

## Calculating a loss function for a simple regression model

For this example, we'll import a series of 300 points from the data folder as an `X` feature, and another series of 300 points as a `y` target


```python
#run this cell as-is

#data manip
import numpy as np

#data import
X = test.load_ind('X')
y = test.load_ind('y')

#used for testing
from test_scripts.test_class import Test
test = Test()
```


```python

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

loss = #your code here
```

### Now, calculate

- determine the partial derivatives for `beta` and `b`


```python
beta_deriv = #your code here
b_deriv = #your code here
```

- re-define `beta` and `b` by "stepping down the gradient" - ie, subtract their respective partial derivatives from themselves


```python
beta = #your code here
b = #your code here
```


```python
#run this cell to test your work!

print('test for beta:')
test.run_test(beta, "beta")
print()
print('test for b:')
test.run_test(b, "b")
```

- does `y = original_beta * X + original_b` have a smaller loss function value than `y = updated_beta * X + new_b`?


```python
loss_original = #your code here

loss_current = #your code here
```

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
#Your work here
```


```python
#run this cell to test your work!

print('test for beta:')
test.run_test(beta, "beta_10")
print()
print('test for b:')
test.run_test(b, "b_10")
```

# Bonus Round

Look at the loss function as we move from epoch to epoch

Why do we not need a learning rate in this instance?

Why might we need a learning rate in other instances?


```python
#Your work here
```
