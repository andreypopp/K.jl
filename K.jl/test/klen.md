# klen

    julia> using K

`#x` computes length of `x`:

    julia> k""" #1 """
    1
    
    julia> k""" #"a" """
    1
    
    julia> k""" #1 2 3 """
    3
    
    julia> k""" #"abc" """
    3
    
    julia> k""" #(1 2; 3 4) """
    2
    
    julia> k""" #`a`b!1 2 """
    2
    
    julia> k""" #!0 """
    0
    
    julia> k""" #() """
    0
