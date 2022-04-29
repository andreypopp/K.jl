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

Chars are converted to ints (and string being lists of chars to lists of ints):

    julia> k"\"0\"|2"
    48
    
    julia> k"2|\"0\""
    48
    
    julia> k"\"1\"|\"0\""
    49
    
    julia> k"\"01\"|2"
    2-element Vector{Int64}:
     48
     49
    
    julia> k"2|\"01\""
    2-element Vector{Int64}:
     48
     49

Works with dictionaries as well, distributing the operation along its values:

    julia> using K.Runtime: OrderedDict as D

    julia> d1, d2 = D(:a=>1,:b=>2), D(:a=>3,:c=>4);

    julia> k"d1|1.5"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :a => 1.5
      :b => 2.0

    julia> k"1.5|d1"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :a => 1.5
      :b => 2.0

    julia> k"d1|1.5 1.6"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :a => 1.5
      :b => 2.0

    julia> k"1.5 1.6|d1"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :a => 1.5
      :b => 2.0

    julia> k"d1|d2"
    K.Runtime.OrderedDict{Symbol, Int64} with 3 entries:
      :a => 3
      :b => 2
      :c => 4
