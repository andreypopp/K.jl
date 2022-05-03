# knull

    julia> using K

`^x` checks if `x` is a "null" value:

    julia> k"^1"
    false

    julia> k"^0"
    false

    julia> k"^0N"
    true

    julia> k"^1.0"
    false

    julia> k"^0.0"
    false

    julia> k"^0n"
    true

    julia> k"^0w"
    false

    julia> k""" ^"\0" """
    false

    julia> k""" ^" " """
    true

    julia> k""" ^"a" """
    false

    julia> k""" ^`a """
    false
    
    julia> k""" ^` """
    true

`^X` distributes `^` along `X` values:
    
    julia> k""" ^"hel \0" """
    5-element BitVector:
     0
     0
     0
     1
     0
    
    julia> k""" ^1 2 3 0N 4 """
    5-element BitVector:
     0
     0
     0
     1
     0
    
    julia> k""" ^`a``c """
    3-element BitVector:
     0
     1
     0

    julia> k""" ^(1 2; 0N 4) """
    2-element Vector{BitVector}:
     [0, 0]
     [1, 0]
    
    julia> k""" ^"" """
    0-element BitVector
    
    julia> k""" ^() """
    Any[]

`^d` distributes `^` along `d` values:

    julia> k"^`a`b!1 0N"
    K.Runtime.OrderedDict{Symbol, Bool} with 2 entries:
      :a => 0
      :b => 1
