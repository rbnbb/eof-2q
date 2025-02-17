function myCheck(is_fulfilled, s::AbstractString)
    is_fulfilled && println(s * " is fullfilled")
    return is_fulfilled
end

function strip_zeros(M, atol=1e-16)
    zero_val = zero(M[1])
    A = similar(M)
    for (j, x) in enumerate(M)
        if isapprox(M[j], zero_val; atol)
            A[j] = zero_val
        else
            A[j] = M[j]
        end
    end
    return A
end

"""
    realify(z, atol=1e-6)

Return the real part of number z provided the imaginary part is small enough.
"""
function realify(z, atol=1e-6)
    if -atol < imag(z) < atol
        return real(z)
    end
    @info "realify() called, but imag(z) is not small" z
    return z
end

"""
Floor input number(s) to 3 digits for pretty print reasons.
"""
function snap(nums)
    try
        if typeof(nums[1]) <: Complex
            return floor.(real.(nums); digits=3) + 1im * floor.(imag.(nums); digits=3)
        else
            return floor.(nums; digits=3)
        end
    catch
        if typeof(nums) <: Complex
            return floor.(real.(nums); digits=3) + 1im * floor.(imag.(nums); digits=3)
        else
            return floor.(nums; digits=3)
        end
    end
end

"""
Return x such that \$ a*x^2 + b*x + c = 0 \$
"""
function solve_quadratic_eq(a, b, c)
    delta = b^2 - 4 * a * c
    if delta ≈ 0
        return -b / (2 * a)
    elseif delta > 0
        return (-b - sqrt(delta)) / (2 * a), (-b + sqrt(delta)) / (2 * a)
    elseif delta < 0
        return (-b - im * sqrt(-delta)) / (2 * a), (-b + 1im * sqrt(-delta)) / (2 * a)
    end
end

rotation_mat(θ) = [
    cos(θ) sin(θ);
    -sin(θ) cos(θ)
]

function unitary_mat(a, t, b)
    return diagm([exp(im * a); exp(-im * a)]) * rotation_mat(t) *
           diagm([exp(im * b); exp(-im * b)])
end
