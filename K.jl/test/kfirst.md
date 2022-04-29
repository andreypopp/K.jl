# * first

    julia> using K

`*x` returns first element of a list:

    julia> k"*1 2"
    1

    julia> k"*1.0 2.0"
    1.0

    julia> k"*(1 2; 3 4)"
    2-element Vector{Int64}:
     1
     2

`*d` returns first value of a dict:

    julia> using OrderedCollections: OrderedDict as D

    julia> d = D(:a=>1,:b=>2);

    julia> k"*d"
    1

It also works on atoms, in this case the atom itself is returned:

    julia> k"*1"
    1

    julia> k"*1.0"
    1.0

In case a list is empty the "null" element is returned:

    julia> k"*()"
    Any[]

    julia> k"*!0"
    -9223372036854775808

