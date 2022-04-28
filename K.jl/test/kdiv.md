# kdiv

    julia> using K

`N%N` does division:

    julia> k"4%2"
    2.0

    julia> k"8 4%2"
    2-element Vector{Float64}:
     4.0
     2.0

    julia> k"8%4 2"
    2-element Vector{Float64}:
     2.0
     4.0

    julia> k"(16 8; 10 5)%2"
    2-element Vector{Vector{Float64}}:
     [8.0, 4.0]
     [5.0, 2.5]

    julia> k"2%(16 8; 10 5)"
    2-element Vector{Vector{Float64}}:
     [0.125, 0.25]
     [0.2, 0.4]

    julia> k"8 4%4 2"
    2-element Vector{Float64}:
     2.0
     2.0

    julia> k"8 4%4 2 3"
    ERROR: LoadError: AssertionError: length(x) == length(y)
    ⋮

Division by `0` results in `Inf`:

    julia> k"2%0"
    Inf

    julia> k"2 3%0"
    2-element Vector{Float64}:
     Inf
     Inf

    julia> k"4%2 0"
    2-element Vector{Float64}:
      2.0
     Inf
