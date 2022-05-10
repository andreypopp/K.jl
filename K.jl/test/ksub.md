# ksub

    julia> using K

`N-N` does subtraction:

    julia> k"2-1"
    1

    julia> k"2-1 2"
    2-element Vector{Int64}:
     1
     0

    julia> k"1-(1.0 2; 3 4)"
    2-element Vector{Vector}:
     [0.0, -1.0]
     [-2, -3]
    
    julia> k"1 2-1"
    2-element Vector{Int64}:
     0
     1
    
    julia> k"1 2-1 2"
    2-element Vector{Int64}:
     0
     0
    
    julia> k"1 1-1 2 3"
    ERROR: AssertionError: length(x) == length(y)

Chars are converted to ints (and string being lists of chars to lists of ints):

    julia> k"\"0\"-1"
    47
    
    julia> k"1-\"0\""
    -47

    julia> k"\"0\"-\"0\""
    0
    
    julia> k"\"01\"-1"
    2-element Vector{Int64}:
     47
     48
    
    julia> k"1-\"01\""
    2-element Vector{Int64}:
     -47
     -48

Works with dictionaries as well, distributing the operation along its values:

    julia> using K.Runtime: OrderedDict as D

    julia> d1, d2 = D(:a=>1,:b=>2), D(:a=>3,:c=>4);

    julia> k"d1-1"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 0
      :b => 1

    julia> k"1-d1"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 0
      :b => -1

    julia> k"d1-1 2"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 0
      :b => 0

    julia> k"1 2-d1"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 0
      :b => 0

    julia> k"d1-d2"
    K.Runtime.OrderedDict{Symbol, Int64} with 3 entries:
      :a => -2
      :b => 2
      :c => -4
