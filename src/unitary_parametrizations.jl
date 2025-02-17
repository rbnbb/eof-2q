# Define some utility functions for the paramétrisation of unitary functions

"""
Return a 2x2 special unitary matrix parametrized by rotation θ and imaginary phase ϕ.

Kraus operators are defined up to a unitary freedom, but for entropy the 2
angles θ and ϕ are enough to parametrize all relevant ρ-ensembles.
Relevant contex: any unitary matrix can be decomposed into the product of:
(see Wikipedia article "Unitary matrix")

  - a global phase exp(im*ϕ) which leaves entropy unchanged
  - a diagonal unitary diagm(exp(im*α), exp(im*α) which leaves entropy unchanged
  - a rotation matrix R(θ) and another diag
  - a diagonal unitary diagm(exp(im*ϕ), exp(im*ϕ)
"""
function angles2su2mat(θ=0.0, ϕ=0.0)
    if ϕ == 0.0
        return [
            cos(θ) sin(θ)
            -sin(θ) cos(θ)
        ]
    end
    return [
        cos(θ) sin(θ)
        -sin(θ) cos(θ)
    ] * LinearAlgebra.diagm([exp(im * ϕ), exp(-im * ϕ)])
end

"""
Return a 4x4 special unitary matrix parametrized by rotations θ1,2 and phases ϕ1,2.

Useful for expressing the 4x4 Kraus operators F as Kronecker products of 2x2 Kraus ops:
F_J = f_j⊗f_k = ∑ₗₖ (uⱼₗ e_l)⊗(uₖₘ e_m), where lowercase denotes 2x2 ops and uppercase 4x4
"""
function _su2xsu2mat_from_params(θ1::Float64, ϕ1::Float64, θ2::Float64, ϕ2::Float64)
    u1 = angles2su2mat(θ1, ϕ1)
    u2 = angles2su2mat(θ2, ϕ2)
    U = zeros(eltype(u1), 4, 4)
    # F_J = f_j⊗f_k = ∑ₗₖ (uⱼₗ e_l)⊗(uₖₘ e_m) = ∑_K u_J(j,k)K(l,m) * E_K
    for j in 1:2, k in 1:2, l in 1:2, m in 1:2
        U[2 * (j - 1) + k, 2 * (l - 1) + m] = u1[j, l] * u2[k, m]
    end
    return U
end

"""
Return a 4x4 matrix ∈ SU(4) parametrized by 6 rotation angles θ1-6 and 8 phases.

Based on the paramétrisation given in [arXiv:1303.5904](https://arxiv.org/abs/1303.5904):

U[4] = Ω_{4,3}' * Ω_{4,2}' * Ω_{4,1}' * Ω_{3,2}' * Ω_{3,1}' * Ω_{2,1}' * Φ[4]

Explained in detail in my notes. U(4) elements need 4^2=16 parameters,
BUT because (1) our use case is how rotated Kraus operators change avg. entanglement
of sets of kets and (2) a phase on a state in an ensemble of kets is irrelevant
for  its entanglement (then:) 4 phases shouldn't matter, i.e., neglect diagm(ϕ1-4).
In practice, I could eliminate 1+(global phase)=2 properly (on pen and paper)
and I am left with 16 params - 6 angles - 2 redundant phase = 8 phases.
"""
function angles2su4mat(
    θ::Vector{<:Number}=pi / 4 * ones(6), ϕ::Vector{<:Number}=zeros(Float64, 8)
)::Matrix{<:Number}
    # check dimensionality
    @assert length(θ) == 6 && length(ϕ) == 8
    # define factor matrices Ω as in arXiv:1303.5904, but conjugate transpose
    # so that Φ matrix is rightmost instead of leftmost, this allows to kick
    # a phase easily from Ω43, as to be seen
    function Ω21(θ, χ, ϕ)
        return [
            sin(θ)*exp(im * χ) -cos(θ)*exp(-im * ϕ) 0 0
            cos(θ)*exp(im * ϕ) sin(θ)*exp(-im * χ) 0 0
            0 0 1 0
            0 0 0 1
        ]'
    end
    function Ω31(θ, ϕ)
        return [
            sin(θ) 0 -cos(θ)*exp(-im * ϕ) 0
            0 1 0 0
            cos(θ)*exp(im * ϕ) 0 sin(θ) 0
            0 0 0 1
        ]'
    end
    function Ω32(θ, χ, ϕ)
        return [
            1 0 0 0
            0 sin(θ)*exp(-im * χ) -cos(θ)*exp(-im * ϕ) 0
            0 cos(θ)*exp(im * ϕ) sin(θ)*exp(im * χ) 0
            0 0 0 1
        ]'
    end
    function Ω41(θ, ϕ)
        return [
            sin(θ) 0 0 -cos(θ)*exp(-im * ϕ)
            0 1 0 0
            0 0 1 0
            cos(θ)*exp(im * ϕ) 0 0 sin(θ)
        ]'
    end
    function Ω42(θ, ϕ)
        return [
            1 0 0 0
            0 sin(θ) 0 -cos(θ)*exp(-im * ϕ)
            0 0 1 0
            0 cos(θ)*exp(im * ϕ) 0 sin(θ)
        ]'
    end
    # this term lacks one phase compared to formula because I decomposed it as
    # "diagm(χ-ϕ) * rotmat_43 * diagm(χ+ϕ≡ε)", and dropped the left phase χ-ϕ
    function Ω43(θ, ε)
        return [
            1 0 0 0
            0 1 0 0
            0 0 sin(θ) cos(θ)
            0 0 -cos(θ) sin(θ)
        ] * LinearAlgebra.diagm([1, 1, exp(im * ε), exp(-im * ε)])
    end
    Φ = [0 0 0 1; 0 0 1 0; 0 1 0 0; 1 0 0 0]  # flip matrix
    return Ω43(θ[6], ϕ[8]) *
           Ω42(θ[5], ϕ[7]) *
           Ω41(θ[4], ϕ[6]) *
           Ω32(θ[3], ϕ[5], ϕ[4]) *
           Ω31(θ[2], ϕ[3]) *
           Ω21(θ[1], ϕ[2], ϕ[1]) *
           Φ
end
