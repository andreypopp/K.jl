# kdrop

    julia> using K

`i_Y` drops `i` first (last, for negative `i`) elements from `Y` list:

    julia> k"2_1 2 3 4"
    2-element Vector{Int64}:
     3
     4
    
    julia> k"10_1 2 3 4"
    Int64[]
    
    julia> k"10_!0"
    Int64[]
    
    julia> k"-2_1 2 3 4"
    2-element Vector{Int64}:
     1
     2
    
    julia> k"-10_!0"
    Int64[]

`i_d` drops first (last, for negative `i`) elements from `d` dict:

    julia> k"1_`a`b`c!1 2 3"
    (b = 2, c = 3)
    
    julia> k"-1_`a`b`c!1 2 3"
    (a = 1, b = 2)
    
    julia> k"10_`a`b`c!1 2 3"
    NamedTuple()
    
    julia> k"-10_`a`b`c!1 2 3"
    NamedTuple()

`x_d` drops `x` key from `d` dict:

    julia> k"`a_`a`b!1 2"
    (b = 2,)
    
    julia> k"`x_`a`b!1 2"
    (a = 1, b = 2)
    
    julia> k"3_3 4!1 2" # NOTE: i_d works here
    K.Runtime.OrderedDict{Int64, Int64}()

`I_Y` cuts `Y` intro sublists at positions specified by `I`:

    julia> k"2 4 4_\"abcde\""
    3-element Vector{Vector{Char}}:
     ['c', 'd']
     []
     ['e']
    
    julia> k"2 4 4_\"aaaa\""
    3-element Vector{Vector{Char}}:
     ['a', 'a']
     []
     []
    
    julia> k"2 4 4_\"\""
    ERROR: AssertionError: domain error
    
    julia> k"2 4 4_\"aa\""
    ERROR: AssertionError: domain error

`f_Y` filters out elements from `Y` which match mask computed by `f`:

    julia> k"{0=2!x}_1 2 3 4"
    2-element Vector{Int64}:
     1
     3

    julia> k"{1 0 1 1 1}_1 2 3 4"
    ERROR: BoundsError:…

    julia> k"{1 0 1}_1 2 3 4"
    ERROR: BoundsError:…

`X_i` deletes elements from `X` at index `i`:

    julia> k"1 2 3 4_1"
    3-element Vector{Int64}:
     1
     3
     4
    
    julia> k"1 2 3 4_-2"
    4-element Vector{Int64}:
     1
     2
     3
     4
    
    julia> k"1 2 3 4_10"
    4-element Vector{Int64}:
     1
     2
     3
     4

    julia> k"(`a`b`c!1 2 3)_`a"
    (b = 2, c = 3)
    
    julia> k"(`a`b`c!1 2 3)_`x"
    (a = 1, b = 2, c = 3)
