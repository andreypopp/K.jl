# Parsing

    julia> using K

# Left bind

In K `n v` syntax binds `n` noun as the first argument (hence "left bind") to
dyadic verb `v`:

    julia> K.parse("1+")
    Seq
    └─ LBind
       ├─ Verb(:+)
       └─ Lit(1)

    julia> K.parse("1+/")
    Seq
    └─ LBind
       ├─ Adverb(:/)
       │  └─ Verb(:+)
       └─ Lit(1)

    julia> K.parse("1{x+y}/")
    Seq
    └─ LBind
       ├─ Adverb(:/)
       │  └─ Fun
       │     └─ App
       │        ├─ Verb(:+)
       │        ├─ Id(:x)
       │        └─ Id(:y)
       └─ Lit(1)

Note that `v` means verb and given that `(e)` (for any expression `e`) makes `n`
noun then the following parses as juxtaposition:

    julia> K.parse("1(+)")
    Seq
    └─ App
       ├─ Lit(1)
       └─ Verb(:+)

# Trains

In K it is possible to build function values out other function values by
prepending 1-arity function to either verbs or left binds:

    julia> K.parse("*+")
    Seq
    └─ Train
       ├─ Verb(Symbol("*:"))
       └─ Verb(:+)

    julia> K.parse("*+:")
    Seq
    └─ Train
       ├─ Verb(Symbol("*:"))
       └─ Verb(Symbol("+:"))

    julia> K.parse("*+%")
    Seq
    └─ Train
       ├─ Verb(Symbol("*:"))
       └─ Train
          ├─ Verb(Symbol("+:"))
          └─ Verb(:%)

    julia> K.parse("*1+")
    Seq
    └─ Train
       ├─ Verb(Symbol("*:"))
       └─ LBind
          ├─ Verb(:+)
          └─ Lit(1)

    julia> K.parse("1*1+")
    Seq
    └─ Train
       ├─ LBind
       │  ├─ Verb(:*)
       │  └─ Lit(1)
       └─ LBind
          ├─ Verb(:+)
          └─ Lit(1)

    julia> K.parse("(1*)1+")
    Seq
    └─ Train
       ├─ LBind
       │  ├─ Verb(:*)
       │  └─ Lit(1)
       └─ LBind
          ├─ Verb(:+)
          └─ Lit(1)

Note that `{..}` lambdas are nouns syntactically and therefore cannot be used to
start a train:

    julia> K.parse("-{x+1}")
    Seq
    └─ App
       ├─ Verb(Symbol("-:"))
       └─ Fun
          └─ App
             ├─ Verb(:+)
             ├─ Id(:x)
             └─ Lit(1)

But trains where lambdas is called first are possible by starting with `@`
(application verb):

    julia> K.parse("-{x+1}@")
    Seq
    └─ Train
       ├─ Verb(Symbol("-:"))
       └─ LBind
          ├─ Verb(Symbol("@"))
          └─ Fun
             └─ App
                ├─ Verb(:+)
                ├─ Id(:x)
                └─ Lit(1)

Again `(v)` is a noun so doesn't form a train if composed:

    julia> K.parse("(+)#")
    Seq
    └─ LBind
       ├─ Verb(Symbol("#"))
       └─ Verb(:+)

    julia> K.parse("+(#)")
    Seq
    └─ App
       ├─ Verb(Symbol("+:"))
       └─ Verb(Symbol("#"))

    julia> K.parse("1*(1+)")
    Seq
    └─ App
       ├─ Verb(:*)
       ├─ Lit(1)
       └─ LBind
          ├─ Verb(:+)
          └─ Lit(1)

Note that `n:e` and `:e` are special forms and are parsed as applications (and
not trains) regardless if `e` is a verb or not:

    julia> K.parse("n:+")
    Seq
    └─ App
       ├─ Verb(:(:))
       ├─ Id(:n)
       └─ Verb(:+)

    julia> K.parse(":+")
    Seq
    └─ App
       ├─ Verb(:(:))
       └─ Verb(:+)
