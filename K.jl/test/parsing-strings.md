# Parsing strings

    julia> using K

Chars:

    julia> K.parse("\"c\"")
    Node(seq)
    └─ Lit("c")

    julia> K.parse("\"some\"")
    Node(seq)
    └─ Lit("some")

    julia> K.parse("\"so\\\"me\"")
    Node(seq)
    └─ Lit("so\\\"me")

    julia> K.parse("\"so\nme\"")
    Node(seq)
    └─ Lit("so\nme")

