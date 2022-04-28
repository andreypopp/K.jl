# krev

    julia> using K

`|x` reverses `x`:

    julia> k"|1"
    1
    
    julia> k"|1.0"
    1.0
    
    julia> k"|1 2 3"
    3-element Vector{Int64}:
     3
     2
     1
    
    julia> k"|1.0 2.0 3.0"
    3-element Vector{Float64}:
     3.0
     2.0
     1.0
    
    julia> k"|(1 2; 3 4)"
    2-element Vector{Vector{Int64}}:
     [3, 4]
     [1, 2]
