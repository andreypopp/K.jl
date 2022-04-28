# kand

    julia> using K

`N&N` computes minimum or boolean AND:

    julia> k"2&4"
    2
    
    julia> k"1&0"
    0
    
    julia> k"2&0 1 2 3"
    4-element Vector{Int64}:
     0
     1
     2
     2
    
    julia> k"0 1 2 3&2"
    4-element Vector{Int64}:
     0
     1
     2
     2
    
    julia> k"1 2&3 0"
    2-element Vector{Int64}:
     1
     0

    julia> k"-1&0 1 2 3"
    4-element Vector{Int64}:
     -1
     -1
     -1
     -1
    
    julia> k"1 2&(1 2; 3 4)"
    2-element Vector{Vector{Int64}}:
     [1, 1]
     [2, 2]
    
    julia> k"1 2&1 2 3"
    ERROR: LoadError: AssertionError: length(x) == length(y)
    ⋮

Chars are converted to ints (and string being lists of chars to lists of ints):

    julia> k"\"0\"&2"
    2
    
    julia> k"2&\"0\""
    2
    
    julia> k"\"1\"&\"0\""
    48
    
    julia> k"\"01\"&2"
    2-element Vector{Int64}:
     2
     2
    
    julia> k"2&\"01\""
    2-element Vector{Int64}:
     2
     2
