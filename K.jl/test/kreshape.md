# kreshape

    julia> using K

`i#y` reshapes `y` to have `i` len:

    julia> k""" 3#1 2 3 """
    3-element Vector{Int64}:
     1
     2
     3
    
    julia> k""" 2#1 2 3 """
    2-element Vector{Int64}:
     1
     2
    
    julia> k""" 5#1 2 3 """
    5-element Vector{Int64}:
     1
     2
     3
     1
     2
    
    julia> k""" 0#1 2 3 """
    Int64[]

`I#y` reshapes `y` to have `I` shape:

    julia> k""" 1 2#1 2 3 """
    1-element Vector{Vector{Int64}}:
     [1, 2]
    
    julia> k""" 2 2#1 2 3 4 """
    2-element Vector{Vector{Int64}}:
     [1, 2]
     [3, 4]
    
    julia> k""" 1 5#1 2 3 """
    1-element Vector{Vector{Int64}}:
     [1, 2, 3, 1, 2]
    
    julia> k""" 1 0#1 2 3 """
    1-element Vector{Vector{Int64}}:
     []
    
    julia> k""" 0 1#1 2 3 """
    Any[]

    julia> k""" 2 2#"abcd" """
    2-element Vector{Vector{Char}}:
     ['a', 'b']
     ['c', 'd']

`f#y` replicates `y` using `f`:

    julia> k""" {1}#1 2 """
    2-element Vector{Int64}:
     1
     2
    
    julia> k""" {1}#1 2 """
    2-element Vector{Int64}:
     1
     2
    
    julia> k""" {2}#1 2 """
    4-element Vector{Int64}:
     1
     1
     2
     2
    
    julia> k""" {0}#1 2 """
    Int64[]
    
    julia> k""" {2 1}#1 2 """
    3-element Vector{Int64}:
     1
     1
     2
    
    julia> k""" {2 0}#1 2 """
    2-element Vector{Int64}:
     1
     1

`x#d` builds a new dict out of `d` with keys specified by `x`:

    julia> k""" `a`b#`a`b`c!1 2 3 """
    (a = 1, b = 2)
    
    julia> k""" (,`x)#`a`b`c!1 2 3 """
    (x = -9223372036854775808,)

    julia> k""" (,`x)#`a`b`c!1.0 2 3 """
    (x = NaN,)
    
    julia> k""" ()#`a`b`c!1 2 3 """
    NamedTuple()
