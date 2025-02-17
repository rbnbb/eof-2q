# http://arxiv.org/abs/quant-ph/9709029
using LinearAlgebra

include("pauli_ops.jl")
include("quantum.jl")
include("entanglement.jl")
include("utils.jl")

default_atol() = 1e-14
do_asserts() = true

"""
Ensemble of pure states that form a DensityMatrix.

That is ∑|x_j><x_j| = ρ for |x_j> ∈ Ensemble.
"""
Ensemble = Vector{Vector{T}} where {T<:Number}

function spin_flip(psi::Statevector)
    # do_asserts() && @assert isstatevec(psi) "Provided argument is not good statevector"
    return reduce(kron, Iterators.repeated(σ2, numqubits(psi))) * conj(psi)
end

function spin_flip(rho::DensityMatrix)
    # do_asserts() && @assert isdensitymatrix(rho) "Provided argument is not good rho"
    U = reduce(kron, Iterators.repeated(σ2, Int(log2(size(rho, 1)))))
    return U * conj(rho) * U
end

"""
Return concurrece of state def as C(ψ)≡|<ψ|~ψ>|.
"""
function concurrence(psi::Statevector)
    return norm(psi' * spin_flip(psi))
end

"""
Return the concurrence of rho-ensemble.
"""
function concurrence(vs::Ensemble)
    return max(0.0, realify(sum(v' * spin_flip(v) for v in vs)))
end

"""
Return preconcurrece of state def as C(ψ)≡<ψ|~ψ>/<ψ|ψ>.
"""
function preconcurrence(psi::Statevector)
    return (psi' * spin_flip(psi)) / (psi' * psi)  # possibly negative
end

function concurrence(rho::DensityMatrix)
    λs = sqrt.(strip_zeros(eigvals(rho * spin_flip(rho)), default_atol()))
    λs = real.(λs)
    λs = sort(λs; rev=true)
    return max(0, λs[1] - λs[2] - λs[3] - λs[4])
end

"""
A convex function that maps concurrence to von Neumann entropy.
"""
function epsilon(C::Number)
    x = 1 - C^2
    sq = isapprox(x, 0.0; atol=default_atol()) ? 0.0 : sqrt(x)
    if isapprox(sq, 1.0; atol=default_atol())
        return 0.0  # this is to avoid NaNs due to log
    end
    a = (1 + sq) / 2
    b = (1 - sq) / 2
    return -a * log2(a) - b * log2(b)
end

function _areorthogonal(vs::Ensemble)
    M = [v' * w for v in vs, w in vs]
    return isdiag(strip_zeros(M, default_atol()))
end

function _aretildeorthogonal(vs::Ensemble)
    M = [v' * spin_flip(w) for v in vs, w in vs]
    @debug "M:" strip_zeros(M, default_atol())
    return isdiag(strip_zeros(M, default_atol()))
end

"""
Return ρ-ensemble {|v_j>} fulfilling <v_j|v_k> = λ_j δ_jk.
"""
function eigen_ensemble(rho::DensityMatrix)::Ensemble
    eigs, eigvecs = eigen(rho)
    eigs = realify.(eigs)  # because rho hermitian
    # get a vector of vectors from columns of matrix
    eigvecs = [eigvecs[:, j] for j in 1:length(eigs)]
    # trim_small_eigvals!(eigs, eigvecs, 1e-8)
    eigenvalue_cutoff = 1e-10
    # construct the eigen-ensemble <v_j|v_i> = eigenval_j δ_ij
    vs = reverse([
        sqrt(eigs[j]) * eigvecs[j] for
        j in 1:length(eigs) if !isapprox(eigs[j], 0.0; atol=eigenvalue_cutoff)
    ])
    do_asserts() && @assert _areorthogonal(vs) "Invalid eigen-ensemble"
    # Check Eigen-ensemble is properly subnormalized
    do_asserts() &&
        @assert all([vs[j]' * vs[j] for j in 1:length(vs)] .≈ reverse(eigs)[1:length(vs)])
    return vs
end

"""
Reconstruct ρ from the ensemble of pure states vs.
"""
function ensemble2rho(vs)
    return sum([v * v' for v in vs])
end

function _issymmetric(M::Matrix{<:Number})
    return all(
        isapprox(M[i, j], M[j, i]; atol=default_atol()) for i in 2:size(M, 1) for
        j in 1:(i - 1)
    )
end

"""
Return U such that U*A*U^T is diagonal.
"""
function _decompose_complex_symmetric(A::Matrix{<:Complex})
    # Here follow https://en.wikipedia.org/wiki/Symmetric_matrix#Complex_symmetric_matrices
    do_asserts() && @assert _issymmetric(A)
    _, V = eigen(A' * A)
    C = transpose(V) * A * V
    X = real.(C)
    Y = imag.(C)
    do_asserts() && @assert all(isapprox.(X * Y, Y * X; atol=default_atol()))  # X and Y commute
    _, W = eigen(X)
    do_asserts() && @assert isdiag(strip_zeros(transpose(W) * Y * W, default_atol()))
    do_asserts() && @assert isdiag(strip_zeros(transpose(W) * X * W, default_atol()))
    U = transpose(W) * transpose(V)
    do_asserts() && @assert U * U' ≈ I(size(A, 1)) "U should be unitary $(display(U*U'))"
    do_asserts() &&
        @assert isdiag(strip_zeros(U * A * transpose(U), default_atol())) "U*A*U^T should be diag $(display(U*A*transpose(U)))"
    # make it so the λs are real and positive
    phaseU = diagm([exp(-0.5im * angle(z)) for z in diag((U * A * transpose(U)))])
    @debug "Before phasing" U
    U = phaseU * U
    D = U * A * transpose(U)
    do_asserts() &&
        @assert isdiag(strip_zeros(D, default_atol())) "$(display((U*A*transpose(U)))) should be diag"
    do_asserts() &&
        @assert all(isapprox.(imag.(diag(D)), 0.0; atol=default_atol())) "The eigenvalues should be real"
    return U
end

"""
Reorder vectors in `vs` sucht that the ~overlaps are in decreasing order
"""
function _sort_ensemble_decreasing(vs::Ensemble)::Ensemble
    lams = realify.([v' * spin_flip(v) for v in vs])
    perm = reverse(sortperm(lams))
    return vs[perm]
end

"""
Return ρ-ensemble {|x_j>} fulfilling <x_j|~x_k> = λ_j δ_jk.
"""
function tilde_ortho_ensemble(vs::Ensemble)::Ensemble
    n = length(vs)  # this is the rank of rho
    # the matrix A is denoted τ in arXiv:quant-ph/9709029
    A = [vs[i]' * spin_flip(vs[j]) for i in 1:n, j in 1:n]
    U = _decompose_complex_symmetric(A)
    @debug "" U * A * transpose(U)
    # this is a ρ-ensamble such that <x_i|~x_j> = δ_ij* λ_j
    xs = [sum(conj.(U[j, :]) .* vs) for j in 1:n]
    xs = _sort_ensemble_decreasing(xs)
    do_asserts() && @assert _aretildeorthogonal(xs) "xs don't form a ~-ortho ρ-ensemble"
    return xs
end

"""
Return a ρ-ensemble with optimal entanglement.

Construct it from the tilde orthogonal ensemble by repeatedly:
Take max and min preconcurrence states and mix them via a
rotation till they have equal preconcurrence equal to <c>.
i.e. solve for θ and thus |z1> and |z2>, knowing <z1|~z2> = <c>:
( cosθ sinθ ) ( |y1> )  =  ( |z1> )
( -sinθ cosθ) ( |y2> )  =  ( |z2> )
"""
function make_states_have_avg_preconcurrence(ys::Ensemble)::Ensemble
    function solve_algebraic_eq(
        ket_small_preconcurrence, ket_big_preconcurrence, avg_c
    )::Real
        # here we solve eq preconcurrence(some_ket) = avg_c
        # for some_ket parametrized as a rotation of the
        # two states having largest and smallest
        # preconcurrence in ρ-ensemble
        s = ket_small_preconcurrence
        b = ket_big_preconcurrence
        # define preconcurrences of two states
        C_s = realify(s' * spin_flip(s))
        C_b = realify(b' * spin_flip(b))
        n_s = realify(s' * s)
        n_b = realify(b' * b)
        # some terms in the eq
        B = realify(s' * b + b' * s)::Real
        A = C_b - C_s + avg_c * (n_s - n_b)::Real
        D = avg_c * n_b - C_b
        sols = solve_quadratic_eq(A^2 + avg_c^2 * B^2, 2 * A * D - avg_c^2 * B^2, D^2)
        sols = realify.(sols)
        function g(t)
            return (cos(t)^2 * C_s + sin(t)^2 * C_b) /
                   (cos(t)^2 * n_s + sin(t)^2 * n_b - B * sin(t) * cos(t))
        end  # preconcurrence as a function of θ
        # the good solution must be between [0,1] and give correct
        # @info "Check atol for matching avg_c:" sols avg_c g.(acos.(sqrt.(sols)))
        cos_squared = sols[findfirst(
            x -> x > 0 && x < 1 && isapprox(g(acos(sqrt(x))), avg_c; atol=1e-4), sols
        )]
        do_asserts() && @assert 0 <= cos_squared <= 1 "There is a problem"
        return acos(sqrt(cos_squared))
    end
    # average preconcurrence <c> with weights p_j = <y_j|y_j>
    avg_c = realify(sum([ys[j]' * spin_flip(ys[j]) for j in 1:length(ys)]))
    # preconcurrences of each state in a vector
    cs = realify.([preconcurrence(ys[j]) for j in 1:length(ys)])
    zs = deepcopy(ys)  # this will hold our final result
    for _ in 1:(length(ys) - 1)  # it takes 3 steps to do it
        _, m = findmin(cs)
        _, M = findmax(cs)
        # here we find the θ by solving algebraic equation directly
        θ = solve_algebraic_eq(zs[m], zs[M], avg_c)
        # now update the ρ-ensemble with appropriate rotation
        zs[m], zs[M] = cos(θ) * zs[m] - sin(θ) * zs[M],
        sin(θ) * zs[m] + cos(θ) * zs[M]
        # each such transformation leaves the average preconcurrence unchanged
        do_asserts() && @assert avg_c ≈ sum([z' * spin_flip(z) for z in zs])
        cs = realify.([preconcurrence(z) for z in zs])
    end
    return zs
end

"""
Return a minimally entangled decomposition of rho into pure states.

The method proceeds in 3 decompositions as elaborated in the reference
arXiv:quant-ph/9709029 beginning page 6
"""
function optimal_decomp(rho::DensityMatrix)::Ensemble
    vs = eigen_ensemble(rho)  # eigen-ensemble is the starting point
    ### 1st step {|x_j> | <x_j|~x_k> = λ_j δ_{jk}} with λs in decreasing order
    xs = tilde_ortho_ensemble(vs)
    @debug "xs=" snap.(x' * x for x in xs)
    ### 2nd step just phases the |x_j> decomposition
    ys = [j == 1 ? xs[j] : im * xs[j] for j in 1:length(xs)]
    # the states |y_j> are chosen such that:
    do_asserts() &&
        @assert isapprox(concurrence(rho), concurrence(ys); atol=default_atol() * 1e8) "$(concurrence(rho)) ≠ $(concurrence(ys))"
    ### 3rd step finds ρ-ensemble such that all states have equal preconcurrence
    zs = make_states_have_avg_preconcurrence(ys)
    do_asserts() && @assert rho ≈ ensemble2rho(zs)  # sanity check that {|z_j>} is a ρ-ensemble
    p(ket) = realify(ket' * ket)
    # verify that the decomposition does indeed minimize entanglement entropy
    do_asserts() &&
        @assert isapprox(E_f(rho), epsilon(concurrence(zs)); atol=default_atol() * 1e8)
    return zs
end

E_f(psi) = epsilon(concurrence(psi))

"""
Verify that ϵ(C(ψ)) = S(ψ).
"""
function check_pure_state_formula()
    function rand_psi()
        psi = rand(4)
        return psi ./ norm(psi)
    end
    states = [rand_psi() for _ in 1:5]
    myCheck(
        all(isapprox.(E_f.(states), von_neumann.(states, 2, 2), atol=default_atol())),
        "S(ψ) = E_f(C(ψ))",
    )
    rhos = [states[j] * states[j]' for j in 1:length(states)]
    return myCheck(
        all(isapprox.(E_f.(rhos), E_f.(states), atol=default_atol())),
        "ϵ(C(|ψ><ψ|)) = ϵ(C(ψ))",
    )
end

# keep this parametrization here, but know it's not ok
# "Return a random state of two qubits"
# function rand_statevector()::Statevector
#     t = randn(Float64, 6)
#     return [
#         cos(t[1] * pi / 2.0);
#         exp(im * t[2] * 2 * pi) * cos(t[3] * pi / 2.0) * sin(t[1] * pi / 2.0);
#         exp(im * t[5] * 2 * pi) * cos(t[4] * pi / 2.0) * sin(t[3] * pi / 2.0) * sin(t[1] * pi / 2.0);
#         exp(im * t[6] * 2 * pi) * sin(t[4] * pi / 2.0) * sin(t[3] * pi / 2.0) * sin(t[1] * pi / 2.0)]
# end

"""
Return a (2 qubit) state with the same entanglement as psi
"""
function same_ee_state(psi::Statevector)::Statevector
    return kron(unitary_mat(rand(3)...), unitary_mat(rand(3)...)) * psi
end
