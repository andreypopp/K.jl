# Parsing numbers

    julia> using K

Integers:

    julia> K.parse("1")
    Node(seq)
    └─ Lit(1)

    julia> K.parse("42")
    Node(seq)
    └─ Lit(42)

    julia> K.parse("-1")
    Node(seq)
    └─ Lit(-1)

    julia> K.parse("-42")
    Node(seq)
    └─ Lit(-42)

    julia> K.parse("0N")
    Node(seq)
    └─ Lit(-9223372036854775808)

Floats:

    julia> K.parse("42.0")
    Node(seq)
    └─ Lit(42.0)

    julia> K.parse("-42.0")
    Node(seq)
    └─ Lit(-42.0)

    julia> K.parse("2e2")
    Node(seq)
    └─ Lit(200.0)

    julia> K.parse("2e-2")
    Node(seq)
    └─ Lit(0.02)

    julia> K.parse("2.5e2")
    Node(seq)
    └─ Lit(250.0)

    julia> K.parse("2.5e-2")
    Node(seq)
    └─ Lit(0.025)

    julia> K.parse("0w")
    Node(seq)
    └─ Lit(Inf)

    julia> K.parse("-0w")
    Node(seq)
    └─ Lit(-Inf)

    julia> K.parse("0n")
    Node(seq)
    └─ Lit(NaN)

    julia> K.parse("-0n")
    Node(seq)
    └─ Node(app)
       ├─ Lit(0)
       ├─ Lit(1)
       └─ Name(:n)

Stranding:

    julia> K.parse("1 2 3")
    Node(seq)
    └─ Node(seq)
       ├─ Lit(1)
       ├─ Lit(2)
       └─ Lit(3)

    julia> K.parse("1.0 2 3")
    Node(seq)
    └─ Node(seq)
       ├─ Lit(1.0)
       ├─ Lit(2)
       └─ Lit(3)

    julia> K.parse("1.0 -2 3")
    Node(seq)
    └─ Node(seq)
       ├─ Lit(1.0)
       ├─ Lit(-2)
       └─ Lit(3)

    julia> K.parse("-1 -2 -3")
    Node(seq)
    └─ Node(seq)
       ├─ Lit(-1)
       ├─ Lit(-2)
       └─ Lit(-3)

    julia> K.parse("1 -2")
    Node(seq)
    └─ Node(seq)
       ├─ Lit(1)
       └─ Lit(-2)

Infix `-`:

    julia> K.parse("1- 2")
    Node(seq)
    └─ Node(app)
       ├─ Node(verb)
       │  └─ Prim(:-)
       ├─ Lit(2)
       ├─ Lit(1)
       └─ Lit(2)

    julia> K.parse("1 - 2")
    Node(seq)
    └─ Node(app)
       ├─ Node(verb)
       │  └─ Prim(:-)
       ├─ Lit(2)
       ├─ Lit(1)
       └─ Lit(2)

    julia> K.parse("1-2")
    Node(seq)
    └─ Node(app)
       ├─ Node(verb)
       │  └─ Prim(:-)
       ├─ Lit(2)
       ├─ Lit(1)
       └─ Lit(2)
