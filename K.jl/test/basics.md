# Basics

Import `K` module:

    julia> using K

which `@k_str` macro which parses, compiles and evals K code in the scope of the
current module:

    julia> k"1"
    1

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

Observe that having just a single `Float64` value inside a strand makes it
produce `Vector{Float64}`:

    julia> k"1 2.0 3 4"
    4-element Vector{Float64}:
     1.0
     2.0
     3.0
     4.0

There's a way to enter lists using `(..)` notation:

    julia> k"(1; 2; 3)"
    3-element Vector{Int64}:
     1
     2
     3

Note that strand notation within `(..)` doesn't produce a nested list as `(..)`
is also used to group expressions together:

    julia> k"(1 2 3)"
    3-element Vector{Int64}:
     1
     2
     3

Create an empty list:

    julia> k"()"
    Any[]

Create a nested list:

    julia> k"(1 2; 3 4; 5 6)"
    3-element Vector{Vector{Int64}}:
     [1, 2]
     [3, 4]
     [5, 6]

Lists can be irregular (non-rectangular) in its shape:

    julia> k"((1 2; 3 4); 5 6)"
    2-element Vector{Any}:
     [[1, 2], [3, 4]]
     [5, 6]

## Order of evaluation / Precedence

K has unusual precedence rules or rather an order of evaluation for dyadic verbs
(infix operators). Observe this:

    julia> k"2*2+2"
    8

which would be `6` in any other regular programming language or even when using
math notation.

By default K evaluates things right to left and this is the only precedence
rule. The only mechanism to force other order is to use `(..)` syntax to group
expressions:

    julia> k"(2*2)+2"
    6

## Verbs & Functions

K has verbs & functions. Verbs are primitive function values built into the
language, there are monadic verbs (accept a single argument) and dyadic verbs
(accept two arguments and **can be called infix**).

Let's explore using monadic verbs first:

    julia> k"+(1 2; 3 4)"
    2-element Vector{Any}:
     [1, 3]
     [2, 4]

This was `flip` which transposes a list.

Now the same `+` symbol can be used as a dyadic verb, in this case it means
`add` and performs addition:

    julia> k"10+(1 2; 3 4)"
    2-element Vector{Vector{Int64}}:
     [11, 12]
     [13, 14]

There's another syntax to call a function `f[a1;a2;..;an]`. This works for
monadic, dyadic verbs and other functions of any arity. Let's call `flip` and
`add` using this syntax:

    julia> k"+[(1 2; 3 4)]"
    2-element Vector{Any}:
     [1, 3]
     [2, 4]

    julia> k"+[10;(1 2; 3 4)]"
    2-element Vector{Vector{Int64}}:
     [11, 12]
     [13, 14]

Users can define their own functions using lambda syntax `{..}`; `x`, `y` and
`z` names are implicit function arguments (they also determine if a function has
arity 1, 2 or 3 correspondingly):

    julia> k"{x+1} 1"
    2

    julia> k"{x+1}[1]"
    2

    julia> k"{x+y}[1;2]"
    3

    julia> k"{x+y+z}[1;2;3]"
    6

Functions which mention no implicit argument (`x`, `y` or `z`) have arity 1:

    julia> k"{1+2}"
    … (generic function with 1 method)

    julia> k"{1+2}[42]"
    3

    julia> k"{1+2}[]"
    3

Note that `f[]` means `f[::]` (calling `f` with an `::` (self) as argument):

    julia> k"{x 42}[]"
    42

Note that user-defined functions canot be called infix, even if they have
2-arity:

    julia> k"1{x+y}2"
    ERROR: …
    ⋮

In fact K doesn't allow users to define their own dyadic functions

It is an error to supply extra arguments to a function:

    julia> k"{x+1}[1;2]"
    ERROR: AssertionError: arity error

    julia> k"{x+y+1}[1;2;3]"
    ERROR: AssertionError: arity error

But it is possible to supply less arguments than the function's arity, in this
case a new partially applied function is returned:

    julia> k"{x+y+z}[1]"
    *2-pfunction*

    julia> k"{x+y+z}[1][2]"
    *1-pfunction*

    julia> k"{x+y+z}[1][2][3]"
    6

    julia> k"{x+y+z}[1;2][3]"
    6

    julia> k"{x+y+z}[1][2;3]"
    6

Partial application is also possible with dyadic verbs if the supplied argument
is on the left side:

    julia> k"(1+)2"
    3

There's a special monadic verb `:x` which interrupt execution flow and returns
`x` as value of an expression being evaluated. This works at top level:

    julia> k":1"
    1

    julia> k":1;2"
    1

and inside user defined functions:

    julia> k"{:x+1;x+2}[1]"
    2

## Adverbs

K has adverbs (higher-order functions), for example `f/` makes a left fold with
`f`. Let's try it with `+`:

    julia> k"+/1 2 3"
    6

Note that `+/` reads "fold with addition" and not "fold with flip" (both `flip`
and `add` correspond to `+` symbol). The use of adverb picks the dyadic version
of a verb.

Observe that `f` in `f/` can be any expression which evaluates to a function:

    julia> k"(+)/1 2 3"
    6

    julia> k"{x+y}/1 2 3"
    6

Adverbs can have a dyadic form as well, in case of `+/` it is a "left fold with
seed":

    julia> k"1+/1 2 3"
    7

    julia> k"1{x+y}/1 2 3"
    7

[strand notation]: https://aplwiki.com/wiki/Strand_notation
