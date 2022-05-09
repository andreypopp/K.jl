# kgroup

    julia> using K

`=i` produces a nested list structure ressembling unitary matrix of shape `i i`:

    julia> k"=0"
    Vector{Int64}[]
    
    julia> k"=1"
    1-element Vector{Vector{Int64}}:
     [1]
    
    julia> k"=2"
    2-element Vector{Vector{Int64}}:
     [1, 0]
     [0, 1]
    
    julia> k"=4"
    4-element Vector{Vector{Int64}}:
     [1, 0, 0, 0]
     [0, 1, 0, 0]
     [0, 0, 1, 0]
     [0, 0, 0, 1]

    julia> k"=-10"
    ERROR: ArgumentError: invalid Array dimensions

`=X` groups elements in `X` producing a dict from with unique values in `X` as
keys and lists of indicies corresponding to those as values:

    julia> k"=1 2 3 4"
    K.Runtime.OrderedDict{Int64, Vector{Int64}} with 4 entries:
      1 => [0]
      2 => [1]
      3 => [2]
      4 => [3]
    
    julia> k"=1 1 2 3 3 4"
    K.Runtime.OrderedDict{Int64, Vector{Int64}} with 4 entries:
      1 => [0, 1]
      2 => [2]
      3 => [3, 4]
      4 => [5]
    
    julia> k""" ="abracadabra" """
    K.Runtime.OrderedDict{Char, Vector{Int64}} with 5 entries:
      'a' => [0, 3, 5, 7, 10]
      'b' => [1, 8]
      'r' => [2, 9]
      'c' => [4]
      'd' => [6]
    
    julia> k"=(1 2; 3 4; 1 2; 3 4)"
    K.Runtime.OrderedDict{Vector{Int64}, Vector{Int64}} with 2 entries:
      [1, 2] => [0, 2]
      [3, 4] => [1, 3]

    julia> k"=()"
    K.Runtime.OrderedDict{Any, Vector{Int64}}()
