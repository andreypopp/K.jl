# Parsing symbols

    julia> using K

Symbols:

    julia> K.parse("`some")
    Node(seq)
    └─ Lit(:some)
    
    julia> K.parse("`")
    Node(seq)
    └─ Lit(Symbol(""))
    
Lists of symbol (stranding):

    julia> K.parse("`some`another")
    Node(seq)
    └─ Node(seq)
       ├─ Lit(:some)
       └─ Lit(:another)
    
    julia> K.parse("`some`")
    Node(seq)
    └─ Node(seq)
       ├─ Lit(:some)
       └─ Lit(Symbol(""))
    
    julia> K.parse("```some`")
    Node(seq)
    └─ Node(seq)
       ├─ Lit(Symbol(""))
       ├─ Lit(Symbol(""))
       ├─ Lit(:some)
       └─ Lit(Symbol(""))
