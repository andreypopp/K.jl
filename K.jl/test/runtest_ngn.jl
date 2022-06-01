using K
using Test

f = open("./k/t/t.k", "r")
lines = readlines(f)

cases = Tuple{String,String}[]
for line in lines
  isempty(line) && continue
  t, e = map(strip, split(line, " / "))
  push!(cases, (t, e))
end

@testset "ngn/k" begin
  # @testset "parsing" begin
  #   for (t, e) in cases
  #     @test begin
  #       @info "parsing" t
  #       K.parse(t)
  #       true
  #     end
  #     @test begin
  #       @info "parsing" e
  #       K.parse(e)
  #       true
  #     end
  #   end
  # end
  eval_skip = Set{String}([
                           # parsing standalone adverbs
                           raw"(')"
                           raw"(\)"
                           raw"(/)"
                           # parsing 0x
                           raw"(0x;0x00094142;0x01094142)"
                           raw"0 1+0xff"
                           raw"0xff/,\"ab\""
                           # ?
                           raw"?/'(!0;0#0n)"
                           # \:
                           raw"<':\"abcd\"!3 5 4 8"
                           # tables
                           raw"+\+`a`b!+(1 2;3 4;5 6)"
                           raw"t:+d:`a`b!2 3#!6;t,d"
                           raw"t:+d:`a`b!2 3#!6;d,t"
                           raw"t:+d:`a`b!2 3#!6;t,t"
                           raw"(,`b)_+`a`b!2 3#!6"
                           raw"(,`b)#+`a`b!2 3#!6"
                           raw"+`a`b!0 1"
                           raw"$(+``a!!1 1)!,2"
                           raw"`a!`b!`c!1"
                           raw",`a!`b!1"
                           raw",`a`b!(0 1;2 3)"
                           raw"`a!`b!1"
                           raw",`a`b!(0 1;2)"
                           raw"(+`a`b!(0 1;2 3))`b"
                           raw",`a`b!0 1"
                           raw"(+`a`b!(0 1;2 3))1"
                           raw"(+`a`b!(0 1;2 3))2"
                           raw"(`a`b!0 1;`a`b!2 3)"
                           # ??? ngn/k returns (0; -0.0)
                           raw"-/'(!0;0#0n)"
                           # support 0N in # reshape
                           raw"3 0N#!0"
                           raw"2 0N#\"abcdef\""
                           raw"0N 3#\"abcdefgh\""
                           raw"3 0N#\"abcdefgh\""
                           raw"0N 2#\"abcdef\""
                           raw"4 0N#10#1"
                           raw"4 0N#9#1"
                           raw"0N 3#`a`b`c`d`e`f`g"
                           raw"3 0N#`a`b`c`d`e`f`g"
                           # $[x;y;z]
                           raw"{$[x=1;1;2!x;1+3*x;-2!x]}/17"
                           raw"{$[x=1;1;2!x;1+3*x;-2!x]}\17"
                           # .C
                           raw"{*/.'$+/1+!x}\2"
                           raw"{*/.'$+/1+!x}/2"
                           # ??? ngn/k strips last empty element
                           strip(raw""" "\n"\\"ab\ncd\n" """)
                           # non dyadic folds/scans are not supported
                           raw"{(x;y;z)}/[1;2 3 4;5 6 7]"
                           raw"{(x,y,z)}/[1;2 3 4]5 6 7"
                           raw"{(x;y;z)}\[1;2 3 4;5 6 7]"
                           # multi decode
                           raw"2/(1 1 0;0 1 0;1 0 1)"
                           # d' is not supported
                           raw"(\"abc\"!1 3 5)'0 2"
                           # assignment forms
                           raw"{(a;b):2 3;a-b}0"
                           raw"a.b:!2;{a.b,:x}2;a.b"
                           raw"a:()!0;a[\"bc\"]:1;a\"bc\""
                           raw"(a;b):3 4;{(c;d):!2;d+b}0"
                           raw"{-x,:x}@,!2"
                           raw"a+:a:!2;a"
                           # not supported
                           raw"(`?`@{1+2*x})3"
                           # progn
                           raw"+/[-3;0;1]"
                           # unicode
                           raw"(∘):+;1∘2"
                           # system
                           raw"s:`prng[];a:9?0;`prng s;a~9?0"
                           # structural function equality
                           raw"({z}1)2"
                           raw"{f:+';f[1 2]}0"
                           raw"+[2;]"
                           raw"1 2#'"
                           raw"#[1 2;]"
                           raw"#'[1 2;]"
                           # func args
                           raw"{[a;b;c;d;e;f;g] 3}.!7"
                           ## DIVERGE
                           # here we keep LTR order within (..)
                           raw"a:0;(a;a:1;a)"
                          ])
  @testset "eval" begin
    for (n, (t, e)) in enumerate(cases)
      skipped = t in eval_skip ||
                startswith(t, "`?") ||
                startswith(t, "`j") ||
                startswith(t, "`hex") ||
                startswith(t, "`k") ||
                startswith(t, "?[") || # splice
                startswith(t, ".[") || # try/catch
                startswith(t, "`0")
      if !skipped
        @info "$n: $t -> $e"
      end
      @test begin
        K.Runtime.isequal(k(t, Main), k(e, Main))
      end skip=skipped
    end
  end
end
