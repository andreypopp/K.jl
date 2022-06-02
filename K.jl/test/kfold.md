# kfold

    julia> using K

# converge / fixedpoint

The form `f/` is called converge (or fixedpoint).

`f/x` computes new `x` by doing `f x` repeatedly until `x` stops changing:

    julia> k"{-2!x}/8"
    0

or initial `x` value is repeated:

    julia> k"{$[x=2;4;x-1]}/4"
    2
