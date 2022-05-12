# K.jl

**WARNING: EXPERIMENTAL, DO NOT USE**

[K programming language][K] dialect embedded in [Julia][], inspired by
[ngn/k][].

## Installation

The package is not registered and therefore requires cloning the repo with
development environment:

    git clone https://github.com/andreypopp/K.jl
    cd K.jl
    julia

Then:

    julia> using Pkg
    julia> Pkg.instantiate()

## Usage

Start `julia` and:

    julia> using K

    julia> k"+/1 2 3 4"
    10

The K REPL can be entered via `\` key:

      +/1 2 3 4
    10

## Status & Roadmap

Current K.jl is incomplete, any help would be very much appreciated - open
issues, pull requests.

[Julia]: https://julialang.org
[K]: https://en.wikipedia.org/wiki/K_(programming_language)
[ngn/k]: https://codeberg.org/ngn/k
