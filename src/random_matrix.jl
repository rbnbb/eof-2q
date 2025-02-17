using LinearAlgebra

"""
Return inner product of specified columns of A and B.
"""
function _inner_prod(A::Matrix, B::Matrix, nA::Integer, nB::Integer)
    x = zero(eltype(A))
    for j in axes(A, 1)
        x += conj(A[j, nA]) * B[j, nB]
    end
    return x
end

function _divide_col_by!(A::Matrix, ncol::Integer, divisor::Number)
    for j in axes(A, 1)
        A[j, ncol] = A[j, ncol] / divisor
    end
end

function _norm_of_col(A::Matrix, ncol::Integer)
    x = zero(eltype(A))
    for j in axes(A, 1)
        x += A[j, ncol] * conj(A[j, ncol])
    end
    return sqrt(real(x))
end

function _remove_components_along!(A::Matrix, ncol::Integer)
    y = _inner_prod(A, A, ncol, ncol)
    for j in (ncol + 1):size(A, 2)
        x = _inner_prod(A, A, ncol, j) / y
        for k in axes(A, 1)
            A[k, j] -= x * A[k, ncol]
        end
    end
end

function _normalize_cols!(A::Matrix)
    for ncol in axes(A, 2)
        x = _norm_of_col(A, ncol)  # assign norm of v_n to x
        _divide_col_by!(A, ncol, x)  # renormalize column
    end
end

function gram_schmidt(M::Matrix)
    A = copy(M)  # to hold orthonormal basis
    gram_schmidt!(A)
    return A
end

function gram_schmidt!(M::Matrix)
    for n in 1:(size(M, 2) - 1)  # apply Gram-Schmidt algorithm
        # make all column vectors after n be orthogonal to
        # the n-th column vector
        _remove_components_along!(M, n)
    end
    return _normalize_cols!(M)
end

"""
Sample a matrix from the Circular Real Ensemble
i.e. Haar measure on O(n)
"""
function rand_orthogonal(n::Integer)
    M = randn(n, n)
    gram_schmidt!(M)
    return M
end

"""
Sample a matrix from the Circular Unitary Ensemble
i.e. Haar measure on U(n)
"""
function rand_unitary(n::Integer)
    M = randn(ComplexF64, n, n)
    gram_schmidt!(M)
    return M
end

"""
Return a statevector of n qubits generated from a 4 layer Haar random unitary circuit.
"""
function haar_rand_statevec(n=2)
    N = 2^n
    psi = zeros(ComplexF64, N)  # |0>
    psi[1] = 1.0
    for _ in 1:5
        U = rand_unitary(N)
        psi = U * psi
    end
    return psi
end
