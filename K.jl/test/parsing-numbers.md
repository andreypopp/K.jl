# Parsing numbers

    julia> using K

Integers:

    julia> K.parse("1")
    Seq
    └─ Lit(1)

    julia> K.parse("42")
    Seq
    └─ Lit(42)

    julia> K.parse("-1")
    Seq
    └─ Lit(-1)

    julia> K.parse("-42")
    Seq
    └─ Lit(-42)

Int null:

    julia> K.parse("0N")
    Seq
    └─ Lit(-9223372036854775808)

    julia> K.parse("1b")
    Seq
    └─ Lit(1)

Bit masks:

    julia> K.parse("0b")
    Seq
    └─ Lit(0)

    julia> K.parse("101b")
    Seq
    └─ Seq
       ├─ Lit(1)
       ├─ Lit(0)
       └─ Lit(1)

Negative bitmasks are not supported and instead this parses as application of
`-101` to `b` (this is consistent with ngn/k):

    julia> K.parse("-101b")
    Seq
    └─ App
       ├─ Lit(-101)
       └─ Id(:b)

Floats:

    julia> K.parse("42.0")
    Seq
    └─ Lit(42.0)

    julia> K.parse("-42.0")
    Seq
    └─ Lit(-42.0)

    julia> K.parse("2e2")
    Seq
    └─ Lit(200.0)

    julia> K.parse("2e-2")
    Seq
    └─ Lit(0.02)

    julia> K.parse("2.5e2")
    Seq
    └─ Lit(250.0)

    julia> K.parse("2.5e-2")
    Seq
    └─ Lit(0.025)

    julia> K.parse("0w")
    Seq
    └─ Lit(Inf)

    julia> K.parse("-0w")
    Seq
    └─ Lit(-Inf)

    julia> K.parse("0n")
    Seq
    └─ Lit(NaN)

    julia> K.parse("-0n")
    Seq
    └─ Lit(NaN)

Stranding:

    julia> K.parse("1 2 3")
    Seq
    └─ Seq
       ├─ Lit(1)
       ├─ Lit(2)
       └─ Lit(3)

    julia> K.parse("1.0 2 3")
    Seq
    └─ Seq
       ├─ Lit(1.0)
       ├─ Lit(2)
       └─ Lit(3)

    julia> K.parse("1.0 -2 3")
    Seq
    └─ Seq
       ├─ Lit(1.0)
       ├─ Lit(-2)
       └─ Lit(3)

    julia> K.parse("-1 -2 -3")
    Seq
    └─ Seq
       ├─ Lit(-1)
       ├─ Lit(-2)
       └─ Lit(-3)

    julia> K.parse("1 -2")
    Seq
    └─ Seq
       ├─ Lit(1)
       └─ Lit(-2)

Infix `-`:

    julia> K.parse("1- 2")
    Seq
    └─ App
       ├─ Verb(:-)
       ├─ Lit(1)
       └─ Lit(2)

    julia> K.parse("1 - 2")
    Seq
    └─ App
       ├─ Verb(:-)
       ├─ Lit(1)
       └─ Lit(2)

    julia> K.parse("1-2")
    Seq
    └─ App
       ├─ Verb(:-)
       ├─ Lit(1)
       └─ Lit(2)
