# ksub

    julia> using K

`N-N` does subtraction:

    julia> k"2-1"
    1

    julia> k"2-1 2"
    2-element Vector{Int64}:
     1
     0

    julia> k"1-(1.0 2; 3 4)"
    2-element Vector{Vector{Float64}}:
     [0.0, -1.0]
     [-2.0, -3.0]
    
    julia> k"1 2-1"
    2-element Vector{Int64}:
     0
     1
    
    julia> k"1 2-1 2"
    2-element Vector{Int64}:
     0
     0
    
    julia> k"1 1-1 2 3"
    ERROR: LoadError: AssertionError: length(x) == length(y)
    ⋮

Chars are converted to ints (and string being lists of chars to lists of ints):

    julia> k"\"0\"-1"
    47
    
    julia> k"1-\"0\""
    -47

    julia> k"\"0\"-\"0\""
    0
    
    julia> k"\"01\"-1"
    2-element Vector{Int64}:
     47
     48
    
    julia> k"1-\"01\""
    2-element Vector{Int64}:
     -47
     -48
