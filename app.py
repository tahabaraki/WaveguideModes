"""
Waveguide Mode Visualizer
Run locally with: streamlit run app.py
"""

import re
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from scipy.special import jv, jvp, jn_zeros, jnp_zeros


st.set_page_config(
    page_title="Waveguide Mode Visualizer",
    page_icon="〰️",
    layout="wide",
)

st.title("Waveguide Mode Visualizer")
st.caption("Visualize longitudinal and transverse electromagnetic fields in rectangular and circular waveguides.")

# ----------------------------- Mode parsing -----------------------------
def parse_mode(value: str, shape: str):
    match = re.match(r"^(TE|TM)[\s_]*(\d)[\s_]*(\d+)$", value.strip().upper())
    if not match:
        raise ValueError("Invalid mode format. Use examples such as TE21, TM11, or TE10.")

    mode_type, p, q = match.group(1), int(match.group(2)), int(match.group(3))

    if shape == "Circular":
        if q < 1:
            raise ValueError("For circular waveguides, the radial index m must be at least 1.")
    else:
        if p == 0 and q == 0:
            raise ValueError("Mode 00 does not exist.")
        if mode_type == "TM" and (p == 0 or q == 0):
            raise ValueError("For TM modes in rectangular waveguides, both indices must be at least 1.")

    return mode_type, p, q


# ----------------------------- Circular waveguide -----------------------------
def plot_circular(mode_type, n, m, N=41, a=1.0):
    kca = jn_zeros(n, m)[-1] if mode_type == "TM" else jnp_zeros(n, m)[-1]
    kc = kca / a

    x, y = np.meshgrid(np.linspace(-a, a, N), np.linspace(-a, a, N))
    radius = np.hypot(x, y)
    phi = np.arctan2(y, x)
    mask = radius <= a
    radius = np.where(radius < 1e-6, 1e-6, radius)

    Jn = jv(n, kc * radius)
    Jnp = jvp(n, kc * radius)

    longitudinal = Jn * np.cos(n * phi)
    Pr = Jnp * np.cos(n * phi)
    Pphi = -(n / radius) * Jn * np.sin(n * phi)

    if mode_type == "TM":
        Er, Ephi = Pr, Pphi
        Hr, Hphi = -Pphi, Pr
    else:
        Hr, Hphi = Pr, Pphi
        Er, Ephi = Pphi, -Pr

    Ex = Er * np.cos(phi) - Ephi * np.sin(phi)
    Ey = Er * np.sin(phi) + Ephi * np.cos(phi)
    Hx = Hr * np.cos(phi) - Hphi * np.sin(phi)
    Hy = Hr * np.sin(phi) + Hphi * np.cos(phi)

    for array in (longitudinal, Ex, Ey, Hx, Hy):
        array[~mask] = np.nan

    theta = np.linspace(0, 2 * np.pi, 400)
    outline = (a * np.cos(theta), a * np.sin(theta))
    title = f"{mode_type}{n}{m} | k_c a = {kca:.4f}"
    return (
        x, y, longitudinal, (Ex, Ey), (Hx, Hy), outline,
        title, (-a, a), (-a, a), mode_type, f"{mode_type}{n}{m}",
        f"circular_{mode_type}{n}{m}", kc, kca
    )


# ----------------------------- Rectangular waveguide -----------------------------
def plot_rectangular(mode_type, m, n, a=1.0, b=0.5, Nx=33):
    Ny = max(9, int(round(Nx * b / a)))
    x, y = np.meshgrid(np.linspace(0, a, Nx), np.linspace(0, b, Ny))
    kx, ky = m * np.pi / a, n * np.pi / b
    kc = np.hypot(kx, ky)

    if mode_type == "TE":
        longitudinal = np.cos(kx * x) * np.cos(ky * y)
        Px = -kx * np.sin(kx * x) * np.cos(ky * y)
        Py = -ky * np.cos(kx * x) * np.sin(ky * y)
        Hx, Hy = Px, Py
        Ex, Ey = Py, -Px
    else:
        longitudinal = np.sin(kx * x) * np.sin(ky * y)
        Px = kx * np.cos(kx * x) * np.sin(ky * y)
        Py = ky * np.sin(kx * x) * np.cos(ky * y)
        Ex, Ey = Px, Py
        Hx, Hy = -Py, Px

    outline = ([0, a, a, 0, 0], [0, 0, b, b, 0])
    title = f"{mode_type}{m}{n} | k_c = {kc:.4f} (a={a:g}, b={b:g})"
    return (
        x, y, longitudinal, (Ex, Ey), (Hx, Hy), outline,
        title, (0, a), (0, b), mode_type, f"{mode_type}{m}{n}",
        f"rectangular_{mode_type}{m}{n}", kc, None
    )


# ----------------------------- Figure rendering -----------------------------
def make_figure(data, vector_scale=1.0):
    (
        x, y, longitudinal, (Ex, Ey), (Hx, Hy), outline,
        title, xlim, ylim, mode_type, mode_label, filename, kc, kca
    ) = data

    longitudinal_label = "E_z" if mode_type == "TM" else "H_z"
    width = xlim[1] - xlim[0]
    height = ylim[1] - ylim[0]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, max(4, min(6, 5.2 * height / width + 1.2))),
        facecolor="white"
    )

    for index, ax in enumerate(axes):
        mesh = ax.pcolormesh(x, y, longitudinal, shading="gouraud", cmap="jet")
        fig.colorbar(mesh, ax=ax, label=f"{longitudinal_label} (normalized)")

        if index == 0:
            ax.quiver(x, y, Ex, Ey, color="black", angles="xy", scale=vector_scale)
            ax.set_title(f"{mode_label}: {longitudinal_label} and Eₜ")
        else:
            ax.quiver(x, y, Hx, Hy, color="red", angles="xy", scale=vector_scale)
            ax.set_title(f"{mode_label}: {longitudinal_label} and Hₜ")

        ax.plot(*outline, color="black", linewidth=2)
        ax.set_aspect("equal")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.grid(alpha=0.12)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig, filename, kc, kca


# ----------------------------- Sidebar controls -----------------------------
with st.sidebar:
    st.header("Waveguide settings")
    shape = st.radio("Waveguide geometry", ["Rectangular", "Circular"])

    if shape == "Rectangular":
        st.subheader("Cross-section dimensions")
        a = st.number_input("Width a", min_value=0.001, value=1.0, step=0.1, format="%.3f")
        b = st.number_input("Height b", min_value=0.001, value=0.5, step=0.1, format="%.3f")
    else:
        st.subheader("Cross-section dimensions")
        a = st.number_input("Radius a", min_value=0.001, value=1.0, step=0.1, format="%.3f")
        b = None

    resolution = st.slider("Plot resolution", min_value=21, max_value=81, value=41, step=2)
    st.caption("Dimensions are relative units; the field patterns depend on their ratios.")

st.subheader("Mode selection")
if shape == "Circular":
    examples = "TE21, TE12, TM11"
    placeholder = "TE21, TE12, TM11"
else:
    examples = "TE10, TE11, TM11"
    placeholder = "TE10, TE11, TM11"

mode_text = st.text_input(
    "Enter one or more modes (comma- or space-separated)",
    value=placeholder,
    help=f"Examples: {examples}"
)

st.markdown(
    "**Field legend:** colored background = longitudinal field "
    "(`H_z` for TE, `E_z` for TM); black arrows = transverse electric field "
    "(`Eₜ`); red arrows = transverse magnetic field (`Hₜ`)."
)

if st.button("Plot modes", type="primary", use_container_width=True):
    tokens = mode_text.replace(",", " ").split()
    if not tokens:
        st.warning("Enter at least one mode.")
    else:
        for token in tokens:
            try:
                mode_type, p, q = parse_mode(token, shape)
                if shape == "Circular":
                    data = plot_circular(mode_type, p, q, N=resolution, a=a)
                else:
                    data = plot_rectangular(mode_type, p, q, a=a, b=b, Nx=resolution)

                fig, filename, kc, kca = make_figure(data)
                st.divider()
                st.subheader(f"Mode {mode_type}{p}{q}")
                metric_col1, metric_col2 = st.columns(2)
                metric_col1.metric("Cutoff wavenumber (k_c)", f"{kc:.5f}")
                if kca is not None:
                    metric_col2.metric("Dimensionless cutoff (k_c a)", f"{kca:.5f}")

                st.pyplot(fig, use_container_width=True)

                image_buffer = BytesIO()
                fig.savefig(image_buffer, format="png", dpi=200, bbox_inches="tight")
                st.download_button(
                    label=f"Download {filename}.png",
                    data=image_buffer.getvalue(),
                    file_name=f"{filename}.png",
                    mime="image/png",
                    key=f"download_{filename}",
                )
                plt.close(fig)

            except Exception as error:
                st.error(f"{token}: {error}")
