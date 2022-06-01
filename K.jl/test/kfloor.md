# kfloor

    julia> using K

`_n` computes floor of `n`:

    julia> k"_1"
    1

    julia> k"_1.2"
    1

    julia> k"_1 2 3"
    3-element Vector{Int64}:
     1
     2
     3

    julia> k"_1.2 2.9 3.2"
    3-element Vector{Int64}:
     1
     2
     3

    julia> k"_`a`b!1.2 3.9"
    (a = 1, b = 3)

`_C` lowercases string `C`:

    julia> k"_\"A\""
    'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)
    
    julia> k"_\"a\""
    'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)
    
    julia> k"_\"\""
    Char[]
    
    julia> k"_\"AbCD\""
    4-element Vector{Char}:
     'a': ASCII/Unicode U+0061 (category Ll: Letter, lowercase)
     'b': ASCII/Unicode U+0062 (category Ll: Letter, lowercase)
     'c': ASCII/Unicode U+0063 (category Ll: Letter, lowercase)
     'd': ASCII/Unicode U+0064 (category Ll: Letter, lowercase)
    
    julia> k"_`a`b!(\"Hello\"; \"World\")"
    (a = ['h', 'e', 'l', 'l', 'o'], b = ['w', 'o', 'r', 'l', 'd'])
