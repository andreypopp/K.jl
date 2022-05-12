# K.jl

**WARNING: EXPERIMENTAL, DO NOT USE**

[K programming language][K] dialect embedded in [Julia][], inspired by
[ngn/k][].

## Installation

The package is not registered and therefore requires cloning the repo with
development environment:

    $ git clone --recursive https://github.com/andreypopp/K.jl
    $ cd K.jl
    $ julia
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

K.jl is wildly incomplete at this point, any help would be very much
appreciated, list of incomplete items include:

- special forms
  - [ ] `(x;..):y unpack`
  - [ ] `o[..]    recur`
  - [ ] `nv:x     reassignment`
  - [ ] `n[..]:x  indexed assignment`
- verbs
  - [ ] `i?x roll`
  - [ ] `i?x deal`
  - [ ] `s$y cast`
  - [ ] `s$y int`
  - [ ] ` $x string`
  - [ ] `i$C pad`
  - [ ] `s$y cast`
  - [ ] `?i  uniform`
  - [ ] `.S  get`
  - [ ] `.C  eval`
  - [ ] `@x  type` missing impl for function values
  - [ ] `@[x;y;f]   amend`
  - [ ] `@[x;y;F;z] amend`
  - [ ] `.[x;y;f]   drill`
  - [ ] `.[x;y;F;z] drill`
  - [ ] `.[f;y;f]   try`
  - [ ] `?[x;y;z]   splice`
- adverbs
  - [ ] `   X' binsearch`
  - [ ] `   I/ decode` missing impl for higher rank arrays
  - [ ] `   I\ encode` missing impl for higher rank arrays
  - [ ] `  i': window`
  - [ ] `i f': stencil`
  - [ ] `  F': eachprior`
  - [ ] `x F': seeded ':`
  - [ ] `x F/: eachright`
  - [ ] `x F\: eachleft`
- [ ] tables
- [ ] pretty printer / value string representation

[Julia]: https://julialang.org
[K]: https://en.wikipedia.org/wiki/K_(programming_language)
[ngn/k]: https://codeberg.org/ngn/k
