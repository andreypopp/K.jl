# K.jl

**WARNING: EXPERIMENTAL, DO NOT USE**

[K programming language][K] dialect embedded in [Julia][].

There are multiple incompatible version of K, K.jl is inspired by [ngn/k][]
which in turn cites K6 as inspiration.

## Installation & Usage

Start Julia interpreter:

    $ julia

and install `K.jl` from github:

    julia> using Pkg
    julia> Pkg.install("https://github.com/andreypopp/K.jl

Now you can start using K.jl:

    julia> using K

    julia> k"+/1 2 3 4"
    10

There's REPL mode available which is enabled by default in REPL, press `\` key
to activate it:

      +/1 2 3 4
    10

## Status & Roadmap

Current K.jl is incomplete, some chunks of the K are not implemented.

[Julia]: https://julialang.org
[K]: https://en.wikipedia.org/wiki/K_(programming_language)
[ngn/k]: https://codeberg.org/ngn/k
