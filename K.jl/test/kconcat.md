# kconcat

    julia> using K

`x,y` concats `x` and `y` together:

    julia> k"1,2"
    2-element Vector{Int64}:
     1
     2
    
    julia> k"1,2 3"
    3-element Vector{Int64}:
     1
     2
     3
    
    julia> k"1 2,3"
    3-element Vector{Int64}:
     1
     2
     3
    
    julia> k"1 2,3 4"
    4-element Vector{Int64}:
     1
     2
     3
     4
    
`d1,d2` merge dicts `d1` and `d2`:

    julia> k"(`a`b!1 2),`b`c!3 4"
    K.Runtime.OrderedDict{Symbol, Int64} with 3 entries:
      :a => 1
      :b => 3
      :c => 4
