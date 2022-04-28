# kflip

    julia> using K

    julia> k"+1"
    1-element Vector{Vector{Int64}}:
     [1]
    
    julia> k"+1 2 3"
    1-element Vector{Any}:
     [1, 2, 3]
    
    julia> k"+(1 2; 3 4)"
    2-element Vector{Any}:
     [1, 3]
     [2, 4]
    
    julia> k"+(1; 3 4)"
    2-element Vector{Any}:
     [1, 3]
     [1, 4]
    
    julia> k"+(1 2; 3; 4 5)"
    2-element Vector{Any}:
     [1, 3, 4]
     [2, 3, 5]

    julia> k"+()"
    1-element Vector{Vector{Any}}:
     []
