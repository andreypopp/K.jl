# kscan

    julia> using K

# scan-converge / scan-fixedpoint

The form `f\` is called scan-converge (or scan-fixedpoint).

`f\x` computes new `x` by doing `f x` repeatedly collecting all `x` values into
a vector until `x` stops changing:

    julia> k"{-2!x}\8"
    5-element Vector{Int64}:
     8
     4
     2
     1
     0

or initial `x` value is repeated:

    julia> k"{$[x=2;4;x-1]}\4"
    3-element Vector{Int64}:
     4
     3
     2
