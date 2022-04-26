# add +

    julia> using K
    
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
