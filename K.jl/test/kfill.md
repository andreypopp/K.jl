# kfill

    julia> using K

`a^y` fills nulls in `y` with `a`:

    julia> k"1^2"
    2
    
    julia> k"1^0N"
    1
    
    julia> k"1.0^2.0"
    2.0
    
    julia> k"1.0^0n"
    1.0
    
    julia> k""" "a"^"b" """
    'b': ASCII/Unicode U+0062 (category Ll: Letter, lowercase)
    
    julia> k""" "a"^" " """
    'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)

    julia> k""" 1^1 2 0N 3 """
    4-element Vector{Int64}:
     1
     2
     1
     3
    
    julia> k""" "_"^"h w" """
    3-element Vector{Char}:
     'h': ASCII/Unicode U+0068 (category Ll: Letter, lowercase)
     '_': ASCII/Unicode U+005F (category Pc: Punctuation, connector)
     'w': ASCII/Unicode U+0077 (category Ll: Letter, lowercase)

    julia> k"2^`a`b`c!1 0N 3"
    (a = 1, b = 2, c = 3)

`X^y` removes from `X` everything occuring in `y`:

    julia> k""" 1 2 3^2 """
    2-element Vector{Int64}:
     1
     3
    
    julia> k""" 1 2 3^2 3 """
    1-element Vector{Int64}:
     1
    
    julia> k""" "abc"^"c" """
    2-element Vector{Char}:
     'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)
     'b': ASCII/Unicode U+0062 (category Ll: Letter, lowercase)
    
    julia> k""" "abc"^"bc" """
    1-element Vector{Char}:
     'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)

    julia> k""" 1 2 3^!0 """
    3-element Vector{Int64}:
     1
     2
     3

    julia> k""" "abc"^"" """
    3-element Vector{Char}:
     'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)
     'b': ASCII/Unicode U+0062 (category Ll: Letter, lowercase)
     'c': ASCII/Unicode U+0063 (category Ll: Letter, lowercase)

    julia> k""" (1 2; 3 4)^1 2 """
    2-element Vector{Vector{Int64}}:
     [1, 2]
     [3, 4]

    julia> k""" (1 2; 3 4)^,1 2 """
    1-element Vector{Vector{Int64}}:
     [3, 4]

k"(`a`b!1 2)^,`a"
