# kmul

    julia> using K

`N*N` does multiplication:

    julia> k"2*2"
    4

    julia> k"2*1 2"
    2-element Vector{Int64}:
     2
     4

    julia> k"2*(1.0 2; 3 4)"
    2-element Vector{Vector{Float64}}:
     [2.0, 4.0]
     [6.0, 8.0]

    julia> k"1 2*2"
    2-element Vector{Int64}:
     2
     4

    julia> k"1 2*2 3"
    2-element Vector{Int64}:
     2
     6

    julia> k"1 2*1 2 3"
    ERROR: LoadError: AssertionError: length(x) == length(y)
    ⋮

