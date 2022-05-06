# List Indexing

    julia> using K

K list indexing uses same mechanism as function application. Getting the first
element from a list is simply applying that list to `0` value:

    julia> k"(1 2 3) 0"
    1

    julia> k"1 2 3[0]"
    1

Passing a list of indices will result in a list of values corresponding to those
indices:

    julia> k"1 2 3 4 5[1 2 3]"
    3-element Vector{Int64}:
     2
     3
     4

The values will be arranged to the same shape as indices passed:

    julia> k"1 2 3 4 5[(1 2; 3 4)]"
    2-element Vector{Vector{Int64}}:
     [2, 3]
     [4, 5]

It's also possible to index with a dict:

    julia> k"1 2 3 4 5[`one`two!0 1]"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :one => 1
      :two => 2

If we try to index out of bounds we get null values:

    julia> k"1.0 2 3 4 5[10]"
    NaN

    julia> k"1.0 2 3 4 5[10 11]"
    2-element Vector{Float64}:
     NaN
     NaN

    julia> k"1.0 2 3 4 5[(10 1; 1 13)]"
    2-element Vector{Vector{Float64}}:
     [NaN, 2.0]
     [2.0, NaN]

Nested lists can be indexed too:

    julia> k"(1 2; 3 4)[1]"
    2-element Vector{Int64}:
     3
     4

    julia> k"(1 2; 3 4; 5 6)[0 1]"
    2-element Vector{Vector{Int64}}:
     [1, 2]
     [3, 4]

It is possible to drill down nested lists by passing multiple indices:

    julia> k"(1 2; 3 4)[1;1]"
    4

Lists of indices can be used as well when passing multiple indices:

    julia> k"(1 2; 3 4; 5 6)[0 1;1]"
    2-element Vector{Int64}:
     2
     4

    julia> k"(1 2 3; 4 5 6)[0;1 2]"
    2-element Vector{Int64}:
     2
     3

    julia> k"(1 2; 3 4; 5 6)[0 1;0 1]"
    2-element Vector{Vector{Int64}}:
     [1, 2]
     [3, 4]

    julia> k"(1 2; 3 4; 5 6)[0 1;]"
    2-element Vector{Vector{Int64}}:
     [1, 2]
     [3, 4]

    julia> k"(1 2; 3 4; 5 6)[;]"
    3-element Vector{Vector{Int64}}:
     [1, 2]
     [3, 4]
     [5, 6]

Note how `f[X;y]` and `f[X][y]` are different.

`f[X;y]` computes a list of `f[x;y]` for each `x` in `X`:

    julia> k"(1 2; 3 4; 5 6)[0 1;1]"
    2-element Vector{Int64}:
     2
     4

while `f[X][y]` computes `f[X]` and then indexing into it with `y`:

    julia> k"(1 2; 3 4; 5 6)[0 1][1]"
    2-element Vector{Int64}:
     3
     4

Lists containing dicts:

    julia> k"dicts: (`a`b!1 2; `a`b!3 4)";

    julia> k"dicts[;`a]"
    2-element Vector{Int64}:
     1
     3

    julia> k"dicts[;`a`a]"
    2-element Vector{Vector{Int64}}:
     [1, 1]
     [3, 3]

    julia> k"dicts[0;]"
    K.Runtime.OrderedDict{Symbol, Int64} with 2 entries:
      :a => 1
      :b => 2

    julia> k"dicts[0 1;]"
    2-element Vector{K.Runtime.OrderedDict{Symbol, Int64}}:
     K.Runtime.OrderedDict(:a => 1, :b => 2)
     K.Runtime.OrderedDict(:a => 3, :b => 4)

    julia> k"dicts[;]"
    2-element Vector{K.Runtime.OrderedDict{Symbol, Int64}}:
     K.Runtime.OrderedDict(:a => 1, :b => 2)
     K.Runtime.OrderedDict(:a => 3, :b => 4)

Another thing to note that a single `f[x;y]` indexing can call into functions
selected by previous indices:

    julia> k"({x+1}; {x+2})[1;1]"
    3

    julia> k"({x+1}; {x+2})[;1]"
    2-element Vector{Int64}:
     2
     3

    julia> k"({x+1}; {x+2})[0 0 1 1;1]"
    4-element Vector{Int64}:
     2
     2
     3
     3
