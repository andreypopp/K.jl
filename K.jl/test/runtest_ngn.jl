using K
using Test

f = open("../../k/t/t.k", "r")
lines = readlines(f)

cases = Tuple{String,String}[]
for line in lines
  isempty(line) && continue
  t, e = map(strip, split(line, " / "))
  push!(cases, (t, e))
end

@testset "ngn/k" begin
  @testset "parsing" begin
    for (t, e) in cases
      @test begin
        @info "parsing" t
        K.parse(t)
        true
      end
      @test begin
        @info "parsing" e
        K.parse(e)
        true
      end
    end
  end
  @testset "eval" begin
    for (t, e) in cases[1:100]
      @test begin
        @info "$t -> $e"
        K.Runtime.isequal(k(t, Main), k(e, Main))
      end
    end
  end
end
