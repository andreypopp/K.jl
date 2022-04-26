# Basics

Import `K` module:

    julia> using K

which exposes `k` function to evaluate K language expressions:

    julia> k("1")
    1

and `@k_str` macro:

    julia> k"1"
    1

Those work the same so we are going to use `@k_str` going forward as it is less
verbose.

## Literals

Let's try compute some values. There are numbers, integers and floats:

    julia> k"42"
    42

    julia> k"4.2"
    4.2

Lists of numbers can be entered with [strand notation][]:

    julia> k"1 2 3 4"
    4-element Vector{Int64}:
     1
     2
     3
     4

    julia> k"1.0 2.0 3.0 4.0"
    4-element Vector{Float64}:
     1.0
     2.0
     3.0
     4.0

There's also a way to enter lists using `(..)` notation:

    julia> k"(1; 2; 3)"
    3-element Vector{Int64}:
     1
     2
     3

Note that strand notation within `(..)` doesn't produce a nested list:

    julia> k"(1 2 3)"
    3-element Vector{Int64}:
     1
     2
     3

Now `(..)` notation can be used to create nested list:

    julia> k"(1 2; 3 4; 5 6)"
    3-element Vector{Vector{Int64}}:
     [1, 2]
     [3, 4]
     [5, 6]

    julia> k"((1 2; 3 4); 5 6)"
    2-element Vector{Vector}:
     [[1, 2], [3, 4]]
     [5, 6]

## Verbs / Functions

K calls functions verbs. There are built-in verbs and user-defined verbs.

K built-in verbs are mostly monadic (accept a single argument) and dyadic
(accept two arguments).

Let's explore using monadic verbs first:

    julia> k"+(1 2; 3 4)"
    2-element Vector{Any}:
     [1, 3]
     [2, 4]

This is flip, now the same `+` can be used as dyadic addition if supplied with
two arguments:

    julia> k"10+(1 2; 3 4)"
    2-element Vector{Vector{Int64}}:
     [11, 12]
     [13, 14]

Alternatively there's `f[a1;a2;..;an]` syntax to call a function. This works
both for monadic and dyadic verbs (and other arities):

    julia> k"+[(1 2; 3 4)]"
    2-element Vector{Any}:
     [1, 3]
     [2, 4]

    julia> k"+[10;(1 2; 3 4)]"
    2-element Vector{Vector{Int64}}:
     [11, 12]
     [13, 14]

[strand notation]: https://aplwiki.com/wiki/Strand_notation
