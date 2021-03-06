# ksqrt

    julia> using K

`%N` computes square root of `N`:

    julia> k"%4"
    2.0

    julia> k"%4"
    2.0

    julia> k"%3"
    1.7320508075688772

    julia> k"%4.0"
    2.0

    julia> k"%1 2 3 4"
    4-element Vector{Float64}:
     1.0
     1.4142135623730951
     1.7320508075688772
     2.0

    julia> k"%(1 2; 3 4)"
    2-element Vector{Vector{Float64}}:
     [1.0, 1.4142135623730951]
     [1.7320508075688772, 2.0]

`%N` returns `-0.0` if called with negative `N`:

    julia> k"%-2"
    -0.0

    julia> k"%-2 4 -2"
    3-element Vector{Float64}:
     -0.0
      2.0
     -0.0

Chars are converted to ints (and string being lists of chars to lists of ints):

    julia> k"%\"0\""
    6.928203230275509
    
    julia> k"%\"01\""
    2-element Vector{Float64}:
     6.928203230275509
     7.0

Works with dictionaries as well, distributing the operation along its values:

    julia> using K.Runtime: OrderedDict as D

    julia> d = D(:a=>1,:b=>2);

    julia> k"%d"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :a => 1.0
      :b => 1.41421
