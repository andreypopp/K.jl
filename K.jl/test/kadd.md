# kadd

    julia> using K

`N+N` does addition:
    
    julia> k"1+1"
    2

    julia> k"1+1 2"
    2-element Vector{Int64}:
     2
     3

    julia> k"1+(1.0 2; 3 4)"
    2-element Vector{Vector}:
     [2.0, 3.0]
     [4, 5]
    
    julia> k"1 2+1"
    2-element Vector{Int64}:
     2
     3
    
    julia> k"1 2+1 2"
    2-element Vector{Int64}:
     2
     4
    
    julia> k"1 2+1 2 3"
    ERROR: AssertionError: length(x) == length(y)

Chars are converted to ints (and string being lists of chars to lists of ints):

    julia> k"\"0\"+1"
    49
    
    julia> k"1+\"0\""
    49

    julia> k"\"0\"+\"0\""
    96
    
    julia> k"\"01\"+1"
    2-element Vector{Int64}:
     49
     50
    
    julia> k"1+\"01\""
    2-element Vector{Int64}:
     49
     50

Works with dictionaries as well, distributing the operation along its values:

    julia> using K.Runtime: OrderedDict as D

    julia> d1, d2 = D(:a=>1,:b=>2), D(:a=>3,:c=>4);

    julia> k"d1+1"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 2
      :b => 3

    julia> k"1+d1"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 2
      :b => 3

    julia> k"d1+1 2"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 2
      :b => 4

    julia> k"1 2+d1"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 2
      :b => 4

    julia> k"d1+d2"
    K.Runtime.OrderedDict{Symbol, Int64} with 3 entries:
      :a => 4
      :b => 2
      :c => 4
