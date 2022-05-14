# Function projections

    julia> using K

In K if you pass less arguments to a function than it accepts then a function
projection is constructed:

    julia> k"{x+y}[1]"
    *1:1-pfunction*

One can think of a projection as a function which remembers the original
function and a number of arguments fixed at the moment of its construction.

Pass more arguments to the projection and once the number arguments passed so
far matches the arity of the original function, the original function is called:

    julia> k"{x+y}[1][2]"
    3

As you can see the result is the same as if we were supplying 2 required
arguments at once:

    julia> k"{x+y}[1;2]"
    3

It is an error to call a projection with more arguments than required:

    julia> k"{x+y}[1][2;3]"
    ERROR: AssertionError: arity error

For functions with arity >2 we can project by passing several arguments at once:

    julia> k"{x,y,z}[1;2]"
    *1:1-pfunction*

    julia> k"{x,y,z}[1;2][3]"
    3-element Vector{Int64}:
     1
     2
     3

    julia> k"{x,y,z}[1][2;3]"
    3-element Vector{Int64}:
     1
     2
     3

Projecting a projection constructs another projection:

    julia> k"{x,y,z}[1]"
    *2:2-pfunction*

    julia> k"{x,y,z}[1][2]"
    *1:1-pfunction*

    julia> k"{x,y,z}[1][2][3]"
    3-element Vector{Int64}:
     1
     2
     3

Note that we've used `f[..]` syntax to project but that's not necessary. Below
are examples of projecting a function using juxtaposition:

    julia> k"{x,y,z}1"
    *2:2-pfunction*

    julia> k"({x,y,z}1)2"
    *1:1-pfunction*

    julia> k"(({x,y,z}1)2)3"
    3-element Vector{Int64}:
     1
     2
     3

So far we've been constructing function projections by passing first arguments
only. Using `f[..]` syntax with elided arguments we can project a function by
fixing arguments at any position.

Below are examples of such projections which lead to the same final value
computed:

    julia> k"{x,y,z}[;2;][;3][1]"
    3-element Vector{Int64}:
     1
     2
     3

    julia> k"{x,y,z}[;2;][1][3]"
    3-element Vector{Int64}:
     1
     2
     3

    julia> k"{x,y,z}[;;3][;2][1]"
    3-element Vector{Int64}:
     1
     2
     3

    julia> k"{x,y,z}[;;3][1][2]"
    3-element Vector{Int64}:
     1
     2
     3

    julia> k"{x,y,z}[1;;][2][3]"
    3-element Vector{Int64}:
     1
     2
     3

    julia> k"{x,y,z}[1;;][;3][2]"
    3-element Vector{Int64}:
     1
     2
     3

Note that when we use `f[..]` syntax with elided arguments a projection is
created even if supply more arguments than neccessary, calling the projection
will result in an error though:

    julia> k"{x,y,z}[;1;2;3;4]"
    *1:1-function*

    julia> k"{x,y,z}[;1;2;3;4][5]"
    ERROR: AssertionError: arity error
