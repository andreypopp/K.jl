# kwhere

    julia> using K

`&I` is `where`, it produces a list of integers by replicating a sequence from
`0` till length of `I` along the `I` itself:

    julia> k"&0"
    Int64[]
    
    julia> k"&1"
    1-element Vector{Int64}:
     0
    
    julia> k"&3"
    3-element Vector{Int64}:
     0
     0
     0
    
    julia> k"&0 1"
    1-element Vector{Int64}:
     1
    
    julia> k"&1 0 2"
    3-element Vector{Int64}:
     0
     2
     2
    
    julia> k"&1 2 3"
    6-element Vector{Int64}:
     0
     1
     1
     2
     2
     2
