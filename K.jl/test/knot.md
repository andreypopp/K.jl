# knot

    julia> using K

`~x` computes logical NOT of `x`:

    julia> k"~0"
    1
    
    julia> k"~2"
    0

`~X` distributes `~` along values:
    
    julia> k""" ~(0 2;``a;"a \0";::;{}) """
    5-element Vector{Any}:
      [1, 0]
      [1, 0]
      [0, 0, 1]
     1
     0

`~d` distributes `~` along dict values:
    
    julia> k""" ~`a`b!1 0 """
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 0
      :b => 1
