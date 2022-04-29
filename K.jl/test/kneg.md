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

Chars are converted to ints (and string being lists of chars to lists of ints):

    julia> k"-\"0\""
    -48
    
    julia> k"-\"01\""
    2-element Vector{Int64}:
     -48
     -49

Works with dictionaries as well, distributing the operation along its values:

    julia> using K.Runtime: OrderedDict as D

    julia> d = D(:a=>1,:b=>2);

    julia> k"-d"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => -1
      :b => -2
