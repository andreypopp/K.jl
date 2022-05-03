# kmatch

    julia> using K

`x~y` checks if `x` matches `y` exactly.

For atoms `~` checks both values and types:

    julia> k"1~1"
    1
    
    julia> k"1~2"
    0
    
    julia> k"1~1.0"
    0
    
    julia> k"-1~-1"
    1

    julia> k""" "a"~"a" """
    1
    
    julia> k""" "a"~"b" """
    0
    
    julia> k"1~\"a\""
    0

Lists are matched element-wise:

    julia> k"1 2 3~1 2 3"
    1
    
    julia> k"1 2 3~1.0 2 3"
    0
    
    julia> k"(1 2; 3 4)~(1 2; 3 4)"
    1
    
    julia> k""" ("ab"; 3 4)~("ab"; 3 4) """
    1

Dict are matched element-wise as well, the order should match too:

    julia> k""" (`a`b!1 2)~(`a`b!1 2) """
    1
    
    julia> k""" (`a`b!1 3)~(`a`b!1 2) """
    0
    
    julia> k""" (`b`a!1 2)~(`a`b!1 2) """
    0
