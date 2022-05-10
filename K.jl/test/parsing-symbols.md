# Parsing symbols

    julia> using K

Symbols:

    julia> K.parse("`some")
    Seq
    └─ Lit(:some)
    
    julia> K.parse("`")
    Seq
    └─ Lit(Symbol(""))
    
Lists of symbol (stranding):

    julia> K.parse("`some`another")
    Seq
    └─ LSeq
       ├─ Lit(:some)
       └─ Lit(:another)
    
    julia> K.parse("`some`")
    Seq
    └─ LSeq
       ├─ Lit(:some)
       └─ Lit(Symbol(""))
    
    julia> K.parse("```some`")
    Seq
    └─ LSeq
       ├─ Lit(Symbol(""))
       ├─ Lit(Symbol(""))
       ├─ Lit(:some)
       └─ Lit(Symbol(""))
