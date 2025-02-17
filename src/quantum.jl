using LinearAlgebra

Statevector = Vector{T} where {T<:Number}
DensityMatrix = Matrix{T} where {T<:Number}

_qstate_atol() = 1e-11  # is seems 14 is too much for rho trace check

function isstatevec(psi::Statevector)
    return isapprox(psi' * psi, 1.0; atol=_qstate_atol())
end

function isdensitymatrix(rho::DensityMatrix)
    atol = _qstate_atol()
    ispositiveor0 = x -> isapprox(x, 0; atol) ? true : x > 0.0
    if size(rho, 1) != size(rho, 2)
        @warn "Density Matrix is not square $(size(rho))"
        return false
    elseif !isapprox(tr(rho), 1.0; atol)
        @warn "Trace of Density Matrix is not 1, BUT $(tr(rho))."
        return false
    elseif !all(isapprox.(rho, rho'; atol))
        @warn "Density matrix is not Hermitian."
        return false
    elseif !all(ispositiveor0, realify.(eigvals(rho)))
        @warn "Density matrix is not positive semi-definite."
        return false
    end
    return true
end

function rand_density_matrix(n::Integer)
    A = rand(ComplexF64, n, n)
    A = A * A'
    A = A / tr(A)
    return A
end

numqubits(psi::Statevector) = Int(log2(length(psi)))
