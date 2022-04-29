# kmod

    julia> using K

`i!N` is integer division if `i` is negative:

    julia> k"-5!10"
    2

    julia> k"-5!10.0"
    2

    julia> k"-5!16.5"
    3

    julia> k"-5!10 12 -10 -12"
    4-element Vector{Int64}:
      2
      2
     -2
     -3

    julia> k"-5!10.0 12.0 -10.0 -12.0"
    4-element Vector{Int64}:
      2
      2
     -2
     -3

    julia> k"-5!(10 12; -10 -12)"
    2-element Vector{Vector{Int64}}:
     [2, 2]
     [-2, -3]

`i!N` is remainder if `i` is positive:

    julia> k"5!10"
    0

    julia> k"5!10.0"
    0.0

    julia> k"5!12"
    2

    julia> k"5!12.0"
    2.0

    julia> k"5!16.5"
    1.5

    julia> k"5!10 12 -10 -12"
    4-element Vector{Int64}:
     0
     2
     0
     3

    julia> k"5!10.0 12.0 -10.0 -12.0"
    4-element Vector{Float64}:
     0.0
     2.0
     0.0
     3.0

    julia> k"5!(10 12; -10 -12)"
    2-element Vector{Vector{Int64}}:
     [0, 2]
     [0, 3]

`0!N` returns just `N`:

    julia> k"0!10"
    10

    julia> k"0!-10"
    -10

    julia> k"0!10 -10 10.0 -10.0"
    4-element Vector{Float64}:
      10.0
     -10.0
      10.0
     -10.0

Works with dictionaries, distributing along its values:

    julia> using K.Runtime: OrderedDict as D

    julia> d = D(:a=>10,:b=>12,:c=>-10,:d=>-12);

    julia> k"-5!d"
    K.Runtime.OrderedDict{Symbol, Int64} with 4 entries:
      :a => 2
      :b => 2
      :c => -2
      :d => -3

    julia> k"5!d"
    K.Runtime.OrderedDict{Symbol, Int64} with 4 entries:
      :a => 0
      :b => 2
      :c => 0
      :d => 3

`x!y` creates a new dict:

    julia> k"1 2 3!4 5 6"
    K.Runtime.OrderedDict{Int64, Int64} with 3 entries:
      1 => 4
      2 => 5
      3 => 6

    julia> k"`a`b`c!4 5 6"
    K.Runtime.OrderedDict{Symbol, Int64} with 3 entries:
      :a => 4
      :b => 5
      :c => 6
