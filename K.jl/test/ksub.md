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
