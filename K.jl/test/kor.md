# kor

    julia> using K

`N|N` computes maximum or boolean OR:

    julia> k"2|4"
    4
    
    julia> k"1|0"
    1
    
    julia> k"2|0 1 2 3"
    4-element Vector{Int64}:
     2
     2
     2
     3
    
    julia> k"0 1 2 3|2"
    4-element Vector{Int64}:
     2
     2
     2
     3
    
    julia> k"1 2|3 0"
    2-element Vector{Int64}:
     3
     2
    
    julia> k"-1|0 1 2 3"
    4-element Vector{Int64}:
     0
     1
     2
     3
    
    julia> k"1 2|(1 2; 3 4)"
    2-element Vector{Vector{Int64}}:
     [1, 2]
     [3, 4]
    
    julia> k"1 2|1 2 3"
    ERROR: LoadError: AssertionError: length(x) == length(y)
    ⋮
