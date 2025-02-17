using Plots

"""
Return a list of coordinates of the points making up surface z with x and y axis as given.
"""
function surface2points(xs, ys, zs)
    @assert length(xs) * length(ys) == length(zs)
    pts_z = reshape(zs, length(zs))
    pts_x = [x for x in xs for _ in eachindex(ys)]
    pts_y = [y for _ in eachindex(xs) for y in ys]
    return pts_x, pts_y, pts_z
end

function pitick(start, stop, denom; mode=:text)
    a = Int(cld(start, π / denom))
    b = Int(fld(stop, π / denom))
    tick = range(a * π / denom, b * π / denom; step=π / denom)
    ticklabel = piticklabel.((a:b) .// denom, Val(mode))
    return tick, ticklabel
end

function piticklabel(x::Rational, ::Val{:text})
    iszero(x) && return "0"
    S = x < 0 ? "-" : ""
    n, d = abs(numerator(x)), denominator(x)
    N = n == 1 ? "" : repr(n)
    d == 1 && return S * N * "π"
    return S * N * "π/" * repr(d)
end

function piticklabel(x::Rational, ::Val{:latex})
    iszero(x) && return L"0"
    # S = x < 0 ? "-" : ""
    _, d = abs(numerator(x)), denominator(x)
    # N = n == 1 ? "" : repr(n)
    d == 1 && return L"%$S%$N\pi"
    L"%$S\frac{%$N\pi}{%$d}"
end

function my_plot_defaults()
    fs = 10
    return Plots.default(; fontfamily="Computer Modern",
        linewidth=2, framestyle=:box, label=nothing, grid=:dot,
        legendfontsize=8, tickfontsize=fs, titlefontsize=11, guidefontsize=fs, markersize=3,
        markerstrokewidth=0, size=(600, 400),
        palette=:tableau_10)
end

my_plot_defaults()
