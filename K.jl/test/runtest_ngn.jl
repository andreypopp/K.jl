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
                           # >
                           raw"(20>)(2*)\1"
                           raw"(20>)(2*)/1"
                           # ?
                           raw"?/'(!0;0#0n)"
                           # tables
                           raw"+\+`a`b!+(1 2;3 4;5 6)"
                           # ??? ngn/k returns (0; -0.0)
                           raw"-/'(!0;0#0n)"
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
                           # reassignment
                           raw"a.b:!2;{a.b,:x}2;a.b"
                          ])
  @testset "eval" begin
    for (t, e) in cases
      if t in eval_skip
        @warn "$t -> $e"
      else
        @info "$t -> $e"
      end
      @test begin
        K.Runtime.isequal(k(t, Main), k(e, Main))
      end skip=(t in eval_skip)
    end
  end
end
