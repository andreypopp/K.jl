# kdiv

    julia> using K

`N%N` does division:

    julia> k"4%2"
    2.0

    julia> k"8 4%2"
    2-element Vector{Float64}:
     4.0
     2.0

    julia> k"8%4 2"
    2-element Vector{Float64}:
     2.0
     4.0

    julia> k"(16 8; 10 5)%2"
    2-element Vector{Vector{Float64}}:
     [8.0, 4.0]
     [5.0, 2.5]

    julia> k"2%(16 8; 10 5)"
    2-element Vector{Vector{Float64}}:
     [0.125, 0.25]
     [0.2, 0.4]

    julia> k"8 4%4 2"
    2-element Vector{Float64}:
     2.0
     2.0

    julia> k"8 4%4 2 3"
    ERROR: LoadError: AssertionError: length(x) == length(y)
    ⋮

Division by `0` results in `Inf`:

    julia> k"2%0"
    Inf

    julia> k"2 3%0"
    2-element Vector{Float64}:
     Inf
     Inf

    julia> k"4%2 0"
    2-element Vector{Float64}:
      2.0
     Inf

Chars are converted to ints (and string being lists of chars to lists of ints):

    julia> k"\"0\"%2"
    24.0
    
    julia> k"2%\"0\""
    0.041666666666666664

    julia> k"\"0\"%\"0\""
    1.0
    
    julia> k"\"01\"%2"
    2-element Vector{Float64}:
     24.0
     24.5
    
    julia> k"2%\"01\""
    2-element Vector{Float64}:
     0.041666666666666664
     0.04081632653061224

Works with dictionaries as well, distributing the operation along its values:

    julia> using K.Runtime: OrderedDict as D

    julia> d1, d2 = D(:a=>1,:b=>2), D(:a=>3,:c=>4);

    julia> k"d1%2"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :a => 0.5
      :b => 1.0

    julia> k"2%d1"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :a => 2.0
      :b => 1.0

    julia> k"d1%2 4"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :a => 0.5
      :b => 0.5

    julia> k"2 4%d1"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :a => 2.0
      :b => 2.0

    julia> k"d1%d2"
    K.Runtime.OrderedDict{Symbol, Float64} with 3 entries:
      :a => 0.333333
      :b => 2.0
      :c => 0.25
