export Token, tokenize

import Automa
import Automa.RegExp: @re_str

re = Automa.RegExp

colon    = re":"
adverb   = re"'" | re"/" | re"\\" | re"':" | re"/:" | re"\\:"
verb1    = colon | re"[\+\-*%!&\|<>=~,^#_$?@\.]"
verb     = verb1 | (verb1 * colon)
id       = re"[a-zA-Z]+[a-zA-Z0-9]*"
name     = id * re.rep(re.cat('.', id))
backq    = re"`"
int      = re"0N" | re"\-?[0-9]+"
bitmask  = re"[01]+b"
float0   = re"\-?[0-9]+\.[0-9]*"
exp      = re"[eE][-+]?[0-9]+"
float    = re"-0n" | re"0n" | re"0w" | re"-0w" | float0 | ((float0 | int) * exp)
str      = re.cat('"', re.rep(re"[^\"]" | re.cat("\\\"")), '"')
symbol   = backq | (backq * id) | (backq * str)
lparen   = re"\("
rparen   = re"\)"
lbracket = re"\["
rbracket = re"\]"
lbrace   = re"{"
rbrace   = re"}"
space    = re" +"
comment  = re.rep1(space) * re"/[^\r\n]*"
newline  = re"\n+"
semi     = re";"

tokenizer = Automa.compile(
  comment  => :(),
  float    => :(emitnumber(:float)),
  int      => :(emitnumber(:int)),
  bitmask  => :(emitnumber(:bitmask)),
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
  str      => :(emitstr()),
  space    => :(markspace()),
  newline  => :(emit(:newline)),
)

context = Automa.CodeGenContext()

Token = Tuple{Symbol,String}

keepneg(tok) =
  tok===:adverb||
  tok===:verb||
  tok===:lparen||
  tok===:lbracket||
  tok===:lbrace||
  tok===:semi||
  tok===:newline

@eval function tokenize(data::AbstractString)::Vector{Token}
  $(Automa.generate_init_code(context, tokenizer))
  p_end = p_eof = sizeof(data)
  toks = Token[]
  space = nothing
  markspace() = (space = te)
  emit(kind) = push!(toks, (kind, data[ts:te]))
  emitstr() = begin
    str = data[ts+1:te-1]
    str = replace(str,
                  "\\0" => "\0",
                  "\\n" => "\n",
                  "\\t" => "\t",
                  "\\r" => "\r",
                  "\\\\" => "\\",
                  "\\\"" => "\"")
    push!(toks, (:str, str))
  end
  emitnumber(kind) =
    begin
      num = data[ts:te]
      if num[1]=='-' && space!=(ts-1) && !isempty(toks) && !keepneg(toks[end][1])
        push!(toks, (:verb, "-"))
        push!(toks, (kind, num[2:end]))
      else
        push!(toks, (kind, num))
      end
    end
  while p ≤ p_eof && cs > 0
    $(Automa.generate_exec_code(context, tokenizer))
  end
  if cs < 0 || te < ts
    error("failed to tokenize")
  end
  push!(toks, (:eof, "␀"))
  return toks
end

