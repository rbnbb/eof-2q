"""Study the Ensemble Averaged Entanglement Entropy of sets of kets (ensembles
made of |ψj>) that decompose some density matrix ρ, i.e., ∑|ψj><ψj| = ρ, where
ρ is obtained by applying quantum channels described by sets of KrausOps Es to
simple input states."""

using Statistics
using Optim

include("entanglement_formation_2q.jl")
include("noise_model.jl")
include("noise_types.jl")
include("random_matrix.jl")
include("trajectory_non_unitarity.jl")

"""
Returned the ensemble averaged von_neumann entanglement entropy of the
provided ensemble of pure states
"""
function eaee_of(vs)
    p(ket) = realify(ket' * ket)
    return sum([
        p(v) * von_neumann(v / sqrt(p(v)), 2, 2) for
        v in vs if !isapprox(0.0, p(v); atol=default_atol())
    ])
end

function eaee_of(vs)
    p(ket) = realify(ket' * ket)
    return sum([
        p(v) * von_neumann(v / sqrt(p(v)), 2, 2) for
        v in vs if !isapprox(0.0, p(v); atol=default_atol())
    ])
end

"""
Check if the kraus operators implement a unital quantum channel
"""
function is_unital(Es::KrausOps)
    rho0 = LinearAlgebra.I(2) / 2 # single qubit Kraus Ops
    rho = sum(E * rho0 * E' for E in Es)
    return all(isapprox.(rho, rho0; atol=1e-10))
end

"""
From 2x2 matrices build 4x4 matrices M⊗I if j is odd, the opposite if even.
"""
twoqubitify(j::Integer, Es) =
    isodd(j) ? [kron(E, I(2)) for E in Es] : [kron(I(2), E) for E in Es]

function apply_quantum_channel(rho::DensityMatrix, E::KrausOps)::DensityMatrix
    do_asserts() && @assert isdensitymatrix(rho)
    if size(rho, 1) == size(E[1], 1)
        return sum([e * rho * e' for e in E])
    elseif size(rho, 1) == 4 == 2 * size(E[1], 1)
        return sum([kron(e, I(2)) * rho * kron(e, I(2))' for e in E])
    end
    throw(ArgumentError("I don't know what to do! Please do better!"))
end

"""
Compute ρ-ensemble obtained by applying E_1,E_2 to each state in `kets`
"""
function apply_quantum_channel(kets, E::KrausOps)
    newkets = Vector{ComplexF64}[]  # new ρ-ensemble after update
    if length(kets[1]) != size(E[1], 1)  # if 2x2 E, apply on 1st qubit
        E = [kron(e, I(2)) for e in E]
    end
    do_asserts() && @assert length(kets[1]) == size(E[1], 1)
    for j in eachindex(kets), k in eachindex(E)
        push!(newkets, E[k] * kets[j])  # non-normalisation is expected, no issue
    end
    newkets::Ensemble
    return newkets
end

"""
    transformation_matrix(xs, ys)

Return a unitary matrix such that |y_j> = ∑U_jk |x_k>.

The provided xs and ys must both be ρ-ensembles of same mixed state.
"""
function transformation_matrix(xs::Ensemble, ys::Ensemble)
    rho = ensemble2rho(xs)
    do_asserts() &&
        @assert rho ≈ ensemble2rho(ys) "xs an ys are not ensembles of the same ρ"
    es = [e / (e' * e) for e in eigen_ensemble(rho)]  # <e_j|e_k> = λ_j*δ_jk
    Uy = [es[j]' * ys[l] for l in eachindex(ys), j in eachindex(es)]
    Ux = [es[j]' * xs[l] for l in eachindex(xs), j in eachindex(es)]
    # @show size(Ux) size(Uy)
    U = Uy * pinv(Ux)
    do_asserts() && @assert ys ≈ U * xs
    return U
end

"""
Return 2x2 unitary matrix parametrized as:

{\\displaystyle \\ U=e^{i\\varphi /2}{\\begin{bmatrix}e^{i\\alpha }\\cos \\theta &e^{i\\beta }\\sin \\theta \\\\-e^{-i\\beta }\\sin \\theta &e^{-i\\alpha }\\cos \\theta \\\\\\end{bmatrix}}\\ ,}
"""
function unitary_from_all_params(α, β, θ, ϕ=0.0)
    return exp(im * ϕ / 2) * [
        exp(im * α)*cos(θ) exp(im * β)*sin(θ)
        -exp(-im * β)*sin(θ) exp(-im * α)*cos(θ)
    ]
end

"""
Provided a 2x2 unitrary matrix return its parametrisation in terms of 4 angles.

Use 1st method described in
https://en.wikipedia.org/wiki/Unitary_matrix#Elementary_constructions
"""
function unitary2params(U::Matrix{<:Number})
    do_asserts() && @assert size(U) == (2, 2) && U * U' ≈ I(2)  # U must be 2x2 unitary
    ϕ = angle(det(U))
    U1 = exp(-im * ϕ / 2) * U  # this is now element of SU(2)
    do_asserts() &&
        @assert isapprox(det(U1), 1.0; atol=default_atol() * 100) "determinant of U1 is $(det(U1)) not 1"
    α = angle(U1[1, 1])
    β = angle(U1[1, 2])
    sinθ = norm(U1[1, 2])
    θ = asin(sinθ)
    do_asserts() && @assert U ≈ unitary_from_all_params(α, β, θ, ϕ)
    return (θ, (α - β) / 2)  # we only really need 2 pars for optimal entropy
    # return (α, β, θ, ϕ)
end

function optimal_params(ket::Statevector, Es::KrausOps)
    psi = ket / sqrt(ket' * ket)
    rho = apply_quantum_channel(psi * psi', Es)
    basic_ensemble = [E * psi for E in Es]
    optim_ensemble = optimal_decomp(rho)
    return unitary2params(transformation_matrix(basic_ensemble, optim_ensemble))
end

"""
Given a ρ-ensemble `kets` return the optimal branching for given Es.

We leverage the complete freedom for each branching in trajectory.
"""
function apply_quantum_channel_optimally(
    kets, Es::KrausOps, twoqubitify; save_pars=false, good_angles=nothing
)
    do_asserts() && @assert size(Es[1]) == (2, 2)
    # optimized calculation trying to approach E_f
    vs = Statevector[]  # ensemble to fill
    angles = []
    for x in kets  # for each state |x> we branch using the optimal params
        pars = optimal_params(x, twoqubitify(Es))
        Fs = angles2su2mat(pars...) * Es  # good form of Kraus operators
        if save_pars
            pars = snap(pars ./ pi)
            push!(angles, (θ=pars[1], δ=pars[2]))
        end
        do_asserts() && @assert sum(F' * F for F in Fs) ≈ I(2)
        # the ρ-ensemble consisting only of those states we can get from |x>
        branch_kets = apply_quantum_channel([x], twoqubitify(Fs))
        push!(vs, branch_kets...)
    end
    if save_pars
        do_asserts() && @assert !isnothing(good_angles)
        push!(good_angles, angles)
    end
    return vs
end

"""
Given a ρ-ensemble `kets` return branching maximizing non-unitarity for given Es.

This is in-house developed NUMU method.
"""
function apply_quantum_channel_numu(
    kets, Es::KrausOps, twoqubitify; save_pars=false, good_angles=nothing
)
    do_asserts() && @assert size(Es[1]) == (2, 2)
    # optimized calculation trying to approach E_f
    vs = Statevector[]  # ensemble to fill
    angles = []
    for x in kets  # for each state |x> we branch using the optimal params
        pars_guess = [pi / 4, 0.0]
        function cost_function(pars)
            return -averaged_non_unitarity(x, angles2su2mat(pars[1], pars[2]) * Es)
        end
        res = Optim.optimize(cost_function, pars_guess)
        nu_pars = res.minimizer
        Fs = angles2su2mat(nu_pars...) * Es
        if save_pars
            pars = snap(nu_pars ./ pi)
            push!(angles, (θ=pars[1], δ=pars[2]))
        end
        do_asserts() && @assert sum(F' * F for F in Fs) ≈ I(2)
        # the ρ-ensemble consisting only of those states we can get from |x>
        branch_kets = apply_quantum_channel([x], twoqubitify(Fs))
        push!(vs, branch_kets...)
    end
    if save_pars
        do_asserts() && @assert !isnothing(good_angles)
        push!(good_angles, angles)
    end
    return vs
end

function ops_n_Npc_for_numu(psi::Statevector, Es::KrausOps, qubit::Integer)
    pars_guess = [pi / 4, 0.0]
    function cost_function(pars)
        return -averaged_non_unitarity(
            psi, twoqubitify(qubit, angles2su2mat(pars[1], pars[2]) * Es)
        )
    end
    res = Optim.optimize(cost_function, pars_guess)
    nu_pars = res.minimizer
    nu = res.minimum
    Fs = angles2su2mat(nu_pars...) * Es
    return Fs, nu
end

# UNFINISHED
function apply_quantum_channel_ordered_numu(kets, Es::KrausOps)
    do_asserts() && @assert size(Es[1]) == (2, 2)
    # optimized calculation trying to approach E_f
    vs = Statevector[]  # ensemble to fill
    angles = []
    for x in kets  # for each state |x> we branch using the optimal params
        # try 1 -> 2
        Fs12_1, nu12_1 = ops_n_Npc_for_numu(x, Es, 1)
        branch_kets = apply_quantum_channel([x], twoqubitify(1, Fs12_1))
        for y in branch_kets
        end
        params12_2, nu12_2 = params_n_Npc_for_numu(x, Es, 2)
        function cost_function(pars)
            return -averaged_non_unitarity(
                x, twoqubitify(2, angles2su2mat(pars[1], pars[2]) * Es)
            )
        end
        res = Optim.optimize(cost_function, pars_guess)
        nu_pars_2 = res.minimizer
        nu12_2 = res.minimum
        Fs2 = angles2su2mat(nu_pars_2...) * Es
        nu12 = nu12_1 + nu12_2
        do_asserts() && @assert sum(F' * F for F in Fs) ≈ I(2)
        # the ρ-ensemble consisting only of those states we can get from |x>
        branch_kets = apply_quantum_channel([x], twoqubitify(Fs))
        push!(vs, branch_kets...)
    end
    if save_pars
        do_asserts() && @assert !isnothing(good_angles)
        push!(good_angles, angles)
    end
    return vs
end
