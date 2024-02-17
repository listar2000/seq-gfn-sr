## Design of the state tensor


### Biggest change 
- constraint
- the framework requires same-size tensors for a batch

- tensor representation versus the "tree representation"

- old approach: complete binary tree
- specify the maximum depth 2, 3, 4, ...
- d = 3 -> the maximum binary tree has 7 nodes (2^3 - 1)
- the tensor representation is of length-7 and pre/post/level-order

    -2
 -1    -1
-1 -1  -1  -1

    *
 c   sin
-1 -1  x  -1

-> c * sin(x)

-2 -> a spot need to be filled
-1 -> empty placeholder

sample a "binary token" *
sample a "constant token " c
sample a "unary token" sin
sample a "variable token" x

- new approach: use the constraint of # of tokens (operator/constant/varaiable)
- pre-order 
- maximum token is M, the tensor size is M + 2 (overhead of 2)
[# of token left, # of empty space left, 1, ..., M]

token idx -> arity

[3, 4, 5, 1, 2]
       0
    1
  2   5
3  4

-----

torchgfn framework

[batch_size, state_dim (M + 2), arity]

processing step:
1. convert the tensor back into a networkx graph
2. feed the graph into pytorch_geometric (pyg)
3. use some graph-transformer to directly encode the entire graph

directly the treat the entire thing like a "sentence"/sequence -> linear transformer to encode

(don't convert into graph early)
(try to stack all the "info" into the state tensor)

sub-TB 

state tensor --(preprocessor)--> output tensor --> policy estimator --> categorical dist

categorical dist --(masking)--> actual action