# kmul

    julia> using K

`N*N` does multiplication:

    julia> k"2*2"
    4

    julia> k"2*1 2"
    2-element Vector{Int64}:
     2
     4

    julia> k"2*(1.0 2; 3 4)"
    2-element Vector{Vector{Float64}}:
     [2.0, 4.0]
     [6.0, 8.0]

    julia> k"1 2*2"
    2-element Vector{Int64}:
     2
     4

    julia> k"1 2*2 3"
    2-element Vector{Int64}:
     2
     6

    julia> k"1 2*1 2 3"
    ERROR: LoadError: AssertionError: length(x) == length(y)
    ⋮

Chars are converted to ints (and string being lists of chars to lists of ints):

    julia> k"\"0\"*2"
    96
    
    julia> k"2*\"0\""
    96

    julia> k"\"0\"*\"0\""
    2304
    
    julia> k"\"01\"*2"
    2-element Vector{Int64}:
     96
     98
    
    julia> k"2*\"01\""
    2-element Vector{Int64}:
     96
     98

Works with dictionaries as well, distributing the operation along its values:

    julia> using K.Runtime: OrderedDict as D

    julia> d1, d2 = D(:a=>1,:b=>2), D(:a=>3,:c=>4);

    julia> k"d1*2"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 2
      :b => 4

    julia> k"2*d1"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 2
      :b => 4

    julia> k"d1*2 4"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 2
      :b => 8

    julia> k"2 4*d1"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 2
      :b => 8

    julia> k"d1*d2"
    K.Runtime.OrderedDict{Symbol, Int64} with 3 entries:
      :a => 3
      :b => 2
      :c => 4
