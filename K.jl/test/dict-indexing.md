# Dict Indexing

    julia> using K

Dict indexing refers to looking up values in a dict by keys. Dict indexing in K
uses same mechanisms as function application:

    julia> k"d:`a`b`c!1.0 2 3";

    julia> k"d[`a]"
    1.0

    julia> k"d`a"
    1.0

Dicts can be indexed by lists, the result is the list of the same shape as
index:

    julia> k"d`a`b"
    2-element Vector{Float64}:
     1.0
     2.0

    julia> k"d(`a`b;`b`a)"
    2-element Vector{Vector{Float64}}:
     [1.0, 2.0]
     [2.0, 1.0]

    julia> k"d(`a`b;`a)"
    2-element Vector{Any}:
      [1.0, 2.0]
     1.0

Dicts can be indexed by other dicts, the result is the dict with the same keys
as index:

    julia> k"d`aa`bb!`a`b"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :aa => 1.0
      :bb => 2.0

    julia> k"d`aa`bb!(`a;`b`b)"
    K.Runtime.OrderedDict{Symbol, Any} with 2 entries:
      :aa => 1.0
      :bb => [2.0, 2.0]

In case a key is missing, null value will be returned for this key:

    julia> k"d`missing"
    NaN

    julia> k"d`missing`b"
    2-element Vector{Float64}:
     NaN
       2.0

    julia> k"d`aa`bb!`a`missing"
    K.Runtime.OrderedDict{Symbol, Float64} with 2 entries:
      :aa => 1.0
      :bb => NaN

## Dicts with nested lists/dicts

    julia> k"lists:`a`b!(1.0 2;3 4)";

    julia> k"dicts:`a`b!(`c`d!1.0 2;`d`e!3 4)";

    julia> k"lists[`a;0]"
    1.0

    julia> k"dicts[`a;`c]"
    1.0
    
    julia> k"lists[;0]"
    K.Runtime.OrderedDict{Symbol, Real} with 2 entries:
      :a => 1.0
      :b => 3

    julia> k"dicts[;`d]"
    K.Runtime.OrderedDict{Symbol, Real} with 2 entries:
      :a => 2.0
      :b => 3

## Dicts with complex keys

    julia> k"points:(1 2;3 4)!41.0 42";

    julia> k"points[1 2]"
    41.0

    julia> k"points[(1 2; 3 4)]"
    2-element Vector{Float64}:
     41.0
     42.0

    julia> k"points[1.0 2.0]"
    NaN

    julia> k"points[1]"
    ERROR: AssertionError: rank error
