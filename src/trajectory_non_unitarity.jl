"""
Given a 2 qubit operator O, return its subnormalisation p and non-unitarity d_U
"""
function non_unitarity(psi0::Statevector, O::Matrix{<:Number})
    @assert size(O, 1) == size(O, 2)  # must be square
    if size(O, 1) == 2  # make it two qubits
        O = kron(O, I(2))
    end
    @assert size(O, 1) == length(psi0)
    phi = O * psi0
    p = realify(phi' * phi)
    Q = O ./ sqrt(p)
    D = Q' * Q - I(length(psi0))
    d_U = realify(tr(D' * D))
    return (p, d_U)
end

function averaged_non_unitarity(psi0::Statevector, Es::KrausOps)
    # this agrees with the analytic formula:
    # -4 + sum(tr(E'*E*E'*E)/p_my(psi, E) for E \
    # in twoqubitify(1, angles2su2mat(pi/4, 0)*Es))
    sum_d_U = 0.0
    sum_weights = 0.0
    for E in Es
        p, d_U = non_unitarity(psi0, E)
        sum_d_U += p * d_U
        sum_weights += p
    end
    return sum_d_U / sum_weights
end
