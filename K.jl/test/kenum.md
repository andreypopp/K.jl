# kenum

`!i` produces a list of integers of length `i` starting from `0`:

    julia> using K

    julia> k"!10"
    10-element Vector{Int64}:
     0
     1
     2
     3
     4
     5
     6
     7
     8
     9

    julia> k("!-10")
    10-element Vector{Int64}:
     -10
      -9
      -8
      -7
      -6
      -5
      -4
      -3
      -2
      -1

    julia> k"!0"
    Int64[]

    julia> k"!-0"
    Int64[]

`!I` is odometer:

    julia> k"!1 2"
    2-element Vector{Vector{Int64}}:
     [0, 0]
     [0, 1]
    
    julia> k"!1 2 3"
    3-element Vector{Vector{Int64}}:
     [0, 0, 0, 0, 0, 0]
     [0, 0, 0, 1, 1, 1]
     [0, 1, 2, 0, 1, 2]
    
    julia> k"!0 1"
    2-element Vector{Vector{Int64}}:
     []
     []
    
    julia> k"!1 0"
    2-element Vector{Vector{Int64}}:
     []
     []
