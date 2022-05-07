# Parsing strings

    julia> using K

Chars:

    julia> K.parse("\"c\"")
    Seq
    └─ Lit("c")

    julia> K.parse("\"some\"")
    Seq
    └─ Lit("some")

    julia> K.parse("\"so\\\"me\"")
    Seq
    └─ Lit("so\"me")

    julia> K.parse("\"so\nme\"")
    Seq
    └─ Lit("so\nme")

