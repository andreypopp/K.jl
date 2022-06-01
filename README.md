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

## Motivation

K.jl is not an independent implementation of K programming language but a
notation to manipulate native Julia data structures.

K.jl works by parsing a string with K notation and turning it into Julia AST,
you can of think of K.jl as of a non-trivial macro. Fortunately semantics of K
and Julia are not so different and thus K.jl is relatively simple.

Why K though? K is terse and expressive. Programs in K are very small and it is
easier to keep them entirely in one's head. K programs can be used for
communication (it's easier to send a K one liner, the information density is
high).

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
- [ ] tables
- [ ] pretty printer / value string representation

[Julia]: https://julialang.org
[K]: https://en.wikipedia.org/wiki/K_(programming_language)
[ngn/k]: https://codeberg.org/ngn/k
