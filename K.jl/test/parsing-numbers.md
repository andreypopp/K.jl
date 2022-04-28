# Parsing numbers

    julia> using K

Number literals:

    julia> K.parse("1")
    Node(seq)
    └─ Lit(1)


    julia> K.parse("42")
    Node(seq)
    └─ Lit(42)


    julia> K.parse("42.0")
    Node(seq)
    └─ Lit(42.0)


    julia> K.parse("-1")
    Node(seq)
    └─ Lit(-1)


    julia> K.parse("-42")
    Node(seq)
    └─ Lit(-42)


    julia> K.parse("-42.0")
    Node(seq)
    └─ Lit(-42.0)

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
