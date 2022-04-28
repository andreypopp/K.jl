# kneg

    julia> using K

`-N` negates `N`:

    julia> k"-1"
    -1
    
    julia> k"-1 2 3"
    3-element Vector{Int64}:
     -1
      2
      3
    
    julia> k"-1.0 2 3"
    3-element Vector{Float64}:
     -1.0
      2.0
      3.0
    
    julia> k"-(1 2; 3 4)"
    2-element Vector{Vector{Int64}}:
     [-1, -2]
     [-3, -4]
