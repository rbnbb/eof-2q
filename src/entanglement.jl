include("partial_trace.jl")

"""
Compute bipartite SvN for pure state
"""
function von_neumann(psi::Statevector, dimA::Integer=2, dimB::Integer=2)
    @assert dimA * dimB == length(psi)
    rdo = partial_trace(psi * psi', dimA, dimB)
    return von_neumann(rdo)
end

function von_neumann(rdo::Matrix)
    return realify(sum(-x * log2(x) for x in eigvals(rdo) if !isapprox(x, 0.0; atol=1e-8)))
end
