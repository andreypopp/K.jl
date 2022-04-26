module K

module Tokenize

import Automa
import Automa.RegExp: @re_str

re = Automa.RegExp

colon    = re":"
adverb   = re"'" | re"/" | re"\\" | re"':" | re"/:" | re"\\:"
verb1    = colon | re"[\+\-*%!&\|<>=~,^#_\\$?@\.]"
verb     = verb1 | re.cat(verb1, colon)
name     = re"[a-zA-Z]+[a-zA-Z0-9]*"
backq    = re"`"
symbol   = backq | re.cat(backq, name)
int      = re"[-]?[0-9]+"
float0   = re"[-]?[0-9]+\.[0-9]*"
exp      = re"[eE][-+]?[0-9]+"
float    = float0 | re.cat(float0 | int, exp)
lparen   = re"\("
rparen   = re"\)"
lbracket = re"\["
rbracket = re"\]"
lbrace   = re"{"
rbrace   = re"}"
space    = re" +"
semi     = re";"

tokenizer = Automa.compile(
  float    => :(emit(:float)),
  int      => :(emit(:int)),
  name     => :(emit(:name)),
  symbol   => :(emit(:symbol)),
  verb     => :(emit(:verb)),
  adverb   => :(emit(:adverb)),
  lparen   => :(emit(:lparen)),
  rparen   => :(emit(:rparen)),
  lbracket => :(emit(:lbracket)),
  rbracket => :(emit(:rbracket)),
  lbrace   => :(emit(:lbrace)),
  rbrace   => :(emit(:rbrace)),
  semi     => :(emit(:semi)),
  space    => :(),
)

context = Automa.CodeGenContext()

Token = Tuple{Symbol,String}

@eval function tokenize(data::String)::Vector{Token}
  $(Automa.generate_init_code(context, tokenizer))
  p_end = p_eof = sizeof(data)
  toks = Token[]
  emit(kind) = push!(toks, (kind, data[ts:te]))
  while p ≤ p_eof && cs > 0
    $(Automa.generate_exec_code(context, tokenizer))
  end
  if cs < 0; error("failed to tokenize") end
  push!(toks, (:eof, "␀"))
  return toks
end

end

module Parse
import ..Tokenize: Token

abstract type Syntax end

struct Node <: Syntax
  type::Symbol
  body::Vector{Syntax}
end

struct Lit <: Syntax
  v::Union{Int64,Float64,Symbol}
end

struct Name <: Syntax
  v::Symbol
  Name(v::Symbol) = new(v)
  Name(v::String) = new(Symbol(v))
end

struct Prim <: Syntax
  v::Symbol
  Prim(v::Symbol) = new(v)
  Prim(v::String) = new(Symbol(v))
end

isnoun(syn::Syntax) = !isverb(syn)
isverb(syn::Syntax) = syn isa Node && syn.type === :verb

import AbstractTrees

AbstractTrees.nodetype(node::Parse.Node) = node.type
AbstractTrees.children(node::Parse.Node) = node.body

function Base.show(io::IO, node::Parse.Node)
  compact = get(io, :compact, false)
  if compact; print(io, "Node($(node.type))")
  else; AbstractTrees.print_tree(io, node)
  end
end

mutable struct ParseContext
  tokens::Vector{Token}
  pos::Int64
end

peek(ctx::ParseContext) =
  ctx.pos <= length(ctx.tokens) ? begin
    tok = ctx.tokens[ctx.pos]
    tok[1]
  end :
    nothing
skip!(ctx::ParseContext) =
  ctx.pos += 1
consume!(ctx::ParseContext) =
  begin
    @assert ctx.pos <= length(ctx.tokens)
    tok = ctx.tokens[ctx.pos]
    ctx.pos += 1
    tok
  end

import ..Tokenize

function parse(data::String)
  tokens = Tokenize.tokenize(data)
  ctx = ParseContext(tokens, 1)
  node = Node(:seq, exprs(ctx))
  @assert peek(ctx) === :eof
  node
end

function exprs(ctx::ParseContext)
  es = Syntax[]
  while true
    e = expr(ctx)
    if e !== nothing
      push!(es, e)
    end
    if peek(ctx) === :semi
      consume!(ctx)
    else
      return es
    end
  end
  es
end

function expr(ctx::ParseContext)
  t = term(ctx)
  if t === nothing
    nothing
  elseif isverb(t)
    a = expr(ctx)
    if a !== nothing
      Node(:app, [t, Lit(1), a])
    else
      t
    end
  elseif isnoun(t)
    ve = expr(ctx)
    if ve !== nothing
      if ve isa Node &&
         ve.type === :app &&
         length(ve.body) == 3 &&
         ve.body[1] isa Node && ve.body[1].type === :verb
        v, _, x = ve.body
        Node(:app, [v, Lit(2), t, x])
      elseif ve isa Node && ve.type === :verb
        Node(:app, [ve, Lit(2), t])
      else
        Node(:app, [t, Lit(1), ve])
      end
    else
      t
    end
  elseif peek(ctx) === :eof
    nothing
  else
    @assert false
  end
end

function number0(ctx::ParseContext)
  tok, value = consume!(ctx)
  value =
    if tok === :int
      Base.parse(Int64, value)
    elseif tok === :float
      Base.parse(Float64, value)
    else
      @assert false
    end
  Lit(value)
end

function number(ctx::ParseContext)
  syn = number0(ctx)
  next = peek(ctx)
  if next === :int || next === :float
    syn = Node(:seq, [syn])
    while true
      push!(syn.body, number0(ctx))
      next = peek(ctx)
      next === :int || next === :float || break
    end
  end
  syn
end

function term(ctx::ParseContext)
  next = peek(ctx)
  t = 
    if next === :verb
      _, value = consume!(ctx)
      adverb(Node(:verb, [Prim(value)]), ctx)
    elseif next === :name
      _, value = consume!(ctx)
      maybe_adverb(Name(value), ctx)
    elseif next === :int || next === :float
      lit = number(ctx)
      maybe_adverb(lit, ctx)
    elseif next === :symbol
      _, value = consume!(ctx)
      maybe_adverb(Lit(Symbol(value)), ctx)
    elseif next === :lbrace
      consume!(ctx)
      es = exprs(ctx)
      @assert peek(ctx) === :rbrace
      consume!(ctx)
      maybe_adverb(Node(:fun, es), ctx)
    elseif next === :lparen
      consume!(ctx)
      es = exprs(ctx)
      @assert peek(ctx) === :rparen
      consume!(ctx)
      syn = length(es) == 1 ? es[1] : Node(:seq, es)
      maybe_adverb(syn, ctx)
    else
      nothing
    end
  if t !== nothing
    app(t, ctx)
  else
    t
  end
end

function app(t, ctx::ParseContext)
  while peek(ctx) === :lbracket
    consume!(ctx)
    es = exprs(ctx)
    @assert peek(ctx) === :rbracket
    consume!(ctx)
    t = Node(:app, [t, Lit(length(es)), es...])
  end
  t
end

function adverb(verb, ctx::ParseContext)
  while peek(ctx) === :adverb
    _, value = consume!(ctx)
    push!(verb.body, Prim(value))
  end
  verb
end

function maybe_adverb(noun, ctx::ParseContext)
  if peek(ctx) === :adverb
    adverb(Node(:verb, [noun]), ctx)
  else
    noun
  end
end

end

module Runtime

struct PFunction
  f::Function
  args::Tuple
  arity::Int64
end
(s::PFunction)(args...) = s.f(s.args..., args...)
Base.show(io::IO, s::PFunction) = print(io, "*$(s.arity)-kfun*")

arity(f::Function) = methods(f)[1].nargs - 1
arity(f::PFunction) = f.arity

papp(f::Function, args, narr) =
  PFunction(f, args, narr)
papp(f::PFunction, args, narr) =
  PFunction(f.f, (f.args..., args...), narr)

app(f::Union{Function, PFunction}, args...) =
  begin
    flen, alen = arity(f), length(args)
    if flen === alen; f(args...)
    elseif flen > alen; papp(f, args, flen - alen)
    else; @assert false "arity error"
    end
  end

# verbs

# :: x
kself(x) = x

# : right
kright(x, y) = y

# + x
kflip(x) = [[x]]
kflip(x::Vector) =
  begin
    if isempty(x); return [x] end
    y = []
    leading = findfirst(xe -> xe isa Vector, x)
    leading = leading === nothing ? x[1] : x[leading]
    len = length(leading)
    for i in 1:len
      push!(y, [xe isa Vector ? xe[i] : xe for xe in x])
    end
    y
  end

# x + y
kadd(x, y) = x + y
kadd(x, y::Vector) = kadd.(x, y)
kadd(x::Vector, y) = kadd.(x, y)
kadd(x::Vector, y::Vector) =
  (@assert length(x) == length(y); kadd.(x, y))

# - x
kneg(x::X) where {X<:Number} = -x
kneg(x::Vector) = kneg.(x)

# x - y
ksub(x, y) = x - y
ksub(x::Vector, y) = ksub.(x, y)
ksub(x, y::Vector) = ksub.(x, y)
ksub(x::Vector, y::Vector) =
  (@assert length(x) == length(y); ksub.(x, y))

# * x
kfirst(x) = x
kfirst(x::Vector) = isempty(x) ? null(eltype(x)) : x[1]

# x * y
kmul(x, y) = x * y
kmul(x::Vector, y) = kmul.(x, y)
kmul(x, y::Vector) = kmul.(x, y)
kmul(x::Vector, y::Vector) =
  (@assert length(x) == length(y); kmul.(x, y))

# % x
# TODO: ...
ksqrt(x::Float64) = sqrt(x)
ksqrt(x::Vector{Float64}) = sqrt.(x)

# x % y
kdiv(x, y) = x / y
kdiv(x::Vector, y) = kdiv.(x, y)
kdiv(x, y::Vector) = kdiv.(x, y)
kdiv(x::Vector, y::Vector) =
  (@assert length(x) == length(y); kdiv.(x, y))

# ! i
# TODO: maybe remove collect?
kenum(x::Int64) = collect(x < 0 ? (-x:0) : (0:(x - 1)))

# ! I
kenum(x) = @assert false "not implemented"

kkeys(x) = @assert false "not implemented"
knskeys(x) = @assert false "not implemented"
kdict(x, y) = @assert false "not implemented"
kmod(x, y) = @assert false "not implemented"
kwhere(x) = @assert false "not implemented"
kdeepwhere(x) = @assert false "not implemented"
kmin(x, y) = @assert false "not implemented"
krev(x) = @assert false "not implemented"
kmax(x, y) = @assert false "not implemented"
kasc(x) = @assert false "not implemented"
kdesc(x) = @assert false "not implemented"
kless(x, y) = @assert false "not implemented"
kmore(x, y) = @assert false "not implemented"
kgroup(x) = @assert false "not implemented"
keq(x, y) = @assert false "not implemented"
knot(x) = @assert false "not implemented"
kmatch(x, y) = @assert false "not implemented"
kenlist(x) = @assert false "not implemented"
kconcat(x, y) = @assert false "not implemented"
kmerge(x, y) = @assert false "not implemented"
knull(x) = @assert false "not implemented"
kfill(x, y) = @assert false "not implemented"
kwithout(x, y) = @assert false "not implemented"
klen(x) = @assert false "not implemented"
kreshape(x, y) = @assert false "not implemented"
ktake(x, y) = @assert false "not implemented"
kfloor(x) = @assert false "not implemented"
kdrop(x, y) = @assert false "not implemented"
kstr(x) = @assert false "not implemented"
kpad(x, y) = @assert false "not implemented"
kcast(x, y) = @assert false "not implemented"
kuniq(x) = @assert false "not implemented"
kfind(x, y) = @assert false "not implemented"
ktype(x) = @assert false "not implemented"
kget(x) = @assert false "not implemented"
kappn(x, y) = @assert false "not implemented"

# adverbs

# keach(f::M) = (x) -> map(f, x)
# keach(f::D) = (x::Float64, y::Float64) -> f(x, y)
# keach(f::D) = (x::Vector{Float64}, y::Float64) -> f.(x, y)
# keach(f::D) = (x::Float64, y::Vector{Float64}) -> f.(x, y)
# keach(f::D) = (x::Vector{Float64}, y::Vector{Float64}) -> begin
#   @assert length(x) == length(y)
#   f.(x, y)
# end
# keach(x::Vector{Float64}) = @assert false "not implemented"

kfoldM(@nospecialize(f)) = (x) ->
  isempty(x) ? identity(f) : foldl(f, x)
kfoldD(@nospecialize(f)) = (x, y) ->
  foldl(f, y, init=x)

identity(f) =
  if f === kadd; 0
  else; ""
  end

null(::Type{Float64}) = 0.0
null(::Type{Int64}) = 0
# See https://chat.stackexchange.com/transcript/message/58631508#58631508 for a
# reasoning to return "" (an empty string).
null(::Type{Any}) = []

end

module Compile
import ..Parse: Syntax, Node, Lit, Name, Prim
import ..Runtime

verbs = Dict(
             (:(::), 1) => Runtime.kself,
             (:(:), 2) => Runtime.kright,
             (:+, 1) => Runtime.kflip,
             (:+, 2) => Runtime.kadd,
             (:*, 1) => Runtime.kfirst,
             (:*, 2) => Runtime.kmul,
             (:-, 1) => Runtime.kneg,
             (:-, 2) => Runtime.ksub,
             (:(!), 1) => Runtime.kenum,
            )

adverbs = Dict(
               (:(/), 1) => Runtime.kfoldM,
               (:(/), 2) => Runtime.kfoldD,
              )

function compile(syn::Node)
  es = map(compile1, syn.body)
  quote $(es...) end
end

compile1(syn::Node) =
  if syn.type === :seq
    es = map(compile1, syn.body)
    :([$(es...)])
  elseif syn.type === :app
    f, arity, args... = syn.body
    args = map(compile1, args)
    if f.type===:verb
      @assert arity.v == 1 || arity.v === 2
      f = compilefun(f, arity.v)
      compileapp(f, args)
    elseif f.type===:fun
      @assert arity.v === length(args)
      f = eval(compile1(f)) # as this is a function, we can eval it now
      compileapp(f, args)
    else # generic case
      @assert arity.v === length(args)
      f = compile1(f)
      compileapp(f, args)
    end
  elseif syn.type === :fun
    x,y,z=implicitargs(syn.body)
    body = map(compile1, syn.body)
    if z
      :((x, y, z) -> $(body...))
    elseif y
      :((x, y) -> $(body...))
    elseif x
      :((x) -> $(body...))
    else
      :(() -> $(body...))
    end
  elseif syn.type===:verb
    hasavs = length(syn.body) > 1
    prefer_arity = hasavs ? 1 : 2
    :($(compilefun(syn, prefer_arity)))
  end

compile1(syn::Name) =
  :($(syn.v))

compile1(syn::Lit) =
  :($(syn.v))

compile1(syn::Prim) =
  :($(compilefun(syn, 2)))

compileapp(f, args) =
  :($(Runtime.app)($(f), $(args...)))

compileapp(f, args, arity) =
  let alen = length(args)
    if arity == alen
      :($f($(args...)))
    elseif arity > alen
      :($(Runtime.papp)($f, $(tuple(args...)), $(arity - alen)))
    else
      @assert false "invalid arity"
    end
  end

compileapp(f::Union{Function,Runtime.PFunction}, args) =
  compileapp(f, args, Runtime.arity(f))

compilefun(syn::Prim, prefer_arity::Int64) =
  begin
    f = get(verbs, (syn.v, prefer_arity), nothing)
    @assert f !== nothing "primitive is not implemented: $(syn.v) ($prefer_arity arity)"
    f
  end
compilefun(syn::Node, prefer_arity::Int64) =
  if syn.type===:verb
    f, avs... = syn.body
    f = compilefun(f, isempty(avs) ? prefer_arity : 2)
    for av in avs
      @assert av isa Prim
      makef = get(adverbs, (av.v, prefer_arity), nothing)
      @assert makef !== nothing "primitive is not implemented: $(av.v) ($prefer_arity arity)"
      if f isa Expr
        f = :($makef($f))
      else
        f = makef(f)
      end
    end
    f
  else
    compile1(syn)
  end

implicitargs(syn::Name) =
  syn.v===:x,syn.v===:y,syn.v===:z
implicitargs(syn::Lit) =
  false,false,false
implicitargs(syn::Prim) =
  false,false,false
implicitargs(syn::Node) =
  if syn.type === :fun; false,false,false
  else; implicitargs(syn.body)
  end
implicitargs(syns::Vector{Syntax}) =
  begin
    x,y,z=false,false,false
    for syn in syns
      x0,y0,z0=implicitargs(syn)
      x,y,z=x||x0,y||y0,z||z0
      if x&&y&&z; break end
    end
    x,y,z
  end

end

parse = Parse.parse
compile = Compile.compile

k(k::String) =
  begin
    syn = parse(k)
    # @info "syn" syn
    jlcode = compile(parse(k))
    # @info "jlcode" jlcode
    Base.eval(jlcode)
  end

module Repl
using ReplMaker

import ..k

function init()
  # show_function(io::IO, mime::MIME"text/plain", x) = print(io, x)
  initrepl(k,
           prompt_text="k) ",
           prompt_color=:blue, 
           # show_function=show_function,
           startup_text=true,
           start_key=')', 
           mode_name="k")
  nothing
end
end

macro k_str(code); k(code) end

export k
export @k_str

end # module
