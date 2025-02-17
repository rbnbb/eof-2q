"""
Compute efficiently the partial trace as per https://arxiv.org/pdf/1601.07458.pdf.
"""
function partial_trace(rho::Matrix, dimA::Integer, dimB::Integer)
    @assert dimA * dimB == size(rho, 1) == size(rho, 2)
    rdo = zeros(typeof(rho[1]), dimA, dimA)  # we will return rho_A = tr_B(rho)
    for k in 1:dimA
        for l in k:dimA
            rdo[k, l] = sum([rho[(k - 1) * dimB + j, (l - 1) * dimB + j] for j in 1:dimB])
            if k != l
                rdo[l, k] = conj(rdo[k, l])
            end
        end
    end
    return rdo
end
