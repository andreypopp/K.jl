# kadd

    julia> using K

`N+N` does addition:
    
    julia> k"1+1"
    2

    julia> k"1+1 2"
    2-element Vector{Int64}:
     2
     3

    julia> k"1+(1.0 2; 3 4)"
    2-element Vector{Vector{Float64}}:
     [2.0, 3.0]
     [4.0, 5.0]
    
    julia> k"1 2+1"
    2-element Vector{Int64}:
     2
     3
    
    julia> k"1 2+1 2"
    2-element Vector{Int64}:
     2
     4
    
    julia> k"1 2+1 2 3"
    ERROR: LoadError: AssertionError: length(x) == length(y)
    ⋮

Chars are converted to ints (and string being lists of chars to lists of ints):

    julia> k"\"0\"+1"
    49
    
    julia> k"1+\"0\""
    49

    julia> k"\"0\"+\"0\""
    96
    
    julia> k"\"01\"+1"
    2-element Vector{Int64}:
     49
     50
    
    julia> k"1+\"01\""
    2-element Vector{Int64}:
     49
     50
