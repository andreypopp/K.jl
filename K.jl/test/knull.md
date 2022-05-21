# knull

    julia> using K

`^x` checks if `x` is a "null" value:

    julia> k"^1"
    0

    julia> k"^0"
    0

    julia> k"^0N"
    1

    julia> k"^1.0"
    0

    julia> k"^0.0"
    0

    julia> k"^0n"
    1

    julia> k"^0w"
    0

    julia> k""" ^"\0" """
    0

    julia> k""" ^" " """
    1

    julia> k""" ^"a" """
    0

    julia> k""" ^`a """
    0
    
    julia> k""" ^` """
    1

`^X` distributes `^` along `X` values:
    
    julia> k""" ^"hel \0" """
    5-element Vector{Int64}:
     0
     0
     0
     1
     0
    
    julia> k""" ^1 2 3 0N 4 """
    5-element Vector{Int64}:
     0
     0
     0
     1
     0
    
    julia> k""" ^`a``c """
    3-element Vector{Int64}:
     0
     1
     0

    julia> k""" ^(1 2; 0N 4) """
    2-element Vector{Vector{Int64}}:
     [0, 0]
     [1, 0]
    
    julia> k""" ^"" """
    Int64[]
    
    julia> k""" ^() """
    Int64[]

`^d` distributes `^` along `d` values:

    julia> k"^`a`b!1 0N"
    (a = 0, b = 1)
