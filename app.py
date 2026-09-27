import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Physics Lab Graph Generator",
    page_icon="🔬",
    layout="wide"
)

st.markdown(
    '<meta name="google-site-verification" content="e4ceOcApPdH1Bqi8CXRf_FzyNcEJIaYRglnhaP0SJN8" />',
    unsafe_allow_html=True
)
st.title("🔬 Physics Lab Graph Generator")
st.caption("Enter your experimental readings and generate a clean lab graph with automatic best-fit analysis.")

PRESETS = {
    "Custom Experiment": {"x_label": "X", "x_unit": "", "y_label": "Y", "y_unit": "", "title": "Physics Lab Graph"},
    "Ohm's Law (V vs I)": {"x_label": "Current (I)", "x_unit": "A", "y_label": "Voltage (V)", "y_unit": "V", "title": "Ohm's Law: Voltage vs Current"},
    "Convex Lens (u vs v)": {"x_label": "Object distance (u)", "x_unit": "cm", "y_label": "Image distance (v)", "y_unit": "cm", "title": "Convex Lens: u vs v"},
    "Convex Lens (1/u vs 1/v)": {"x_label": "1/u", "x_unit": "cm⁻¹", "y_label": "1/v", "y_unit": "cm⁻¹", "title": "Convex Lens: 1/u vs 1/v"},
    "Prism (i vs δ)": {"x_label": "Angle of incidence (i)", "x_unit": "°", "y_label": "Angle of deviation (δ)", "y_unit": "°", "title": "Prism: Angle of Deviation vs Angle of Incidence"},
    "Meter Bridge (R vs l)": {"x_label": "Balance length (l)", "x_unit": "cm", "y_label": "Resistance (R)", "y_unit": "Ω", "title": "Meter Bridge: Resistance vs Balance Length"},
    "Sonometer (L vs 1/f)": {"x_label": "1/f", "x_unit": "s", "y_label": "Length (L)", "y_unit": "cm", "title": "Sonometer: Length vs 1/f"},
    "Galvanometer (V vs I)": {"x_label": "Current (I)", "x_unit": "A", "y_label": "Potential difference (V)", "y_unit": "V", "title": "Galvanometer: V vs I"},
    "Forward V-I Characteristics (Diode)": {"x_label": "Forward Voltage (V)", "x_unit": "V", "y_label": "Forward Current (I)", "y_unit": "mA", "title": "Forward V-I Characteristics of a Diode"},
    "Reverse V-I Characteristics (Diode)": {"x_label": "Reverse Voltage (V)", "x_unit": "V", "y_label": "Reverse Current (I)", "y_unit": "μA", "title": "Reverse V-I Characteristics of a Diode"},
    "log(n) vs 10³/T": {"x_label": "10³/T", "x_unit": "K⁻¹", "y_label": "log(n)", "y_unit": "", "title": "log(n) vs 10³/T"},
    "Diffraction Grating (sin θ vs n)": {"x_label": "Order (n)", "x_unit": "", "y_label": "sin θ", "y_unit": "", "title": "Diffraction Grating: sin θ vs n"},
    "Thermionic Emission (log I₀ vs 1/T)": {"x_label": "1/T", "x_unit": "K⁻¹", "y_label": "log I₀", "y_unit": "", "title": "Thermionic Emission: log I₀ vs 1/T"},
    "Radius of Light Spot vs Distance (r vs d)": {"x_label": "Distance (d)", "x_unit": "cm", "y_label": "Radius of light spot (r)", "y_unit": "cm", "title": "r vs d"},
    "Newton's Rings (d vs x)": {"x_label": "Distance (x)", "x_unit": "cm", "y_label": "Diameter (d)", "y_unit": "cm", "title": "d vs x"},
    "Photoelectric Effect (K vs f)": {"x_label": "Frequency (f)", "x_unit": "Hz", "y_label": "Kinetic Energy (K)", "y_unit": "eV", "title": "Photoelectric Effect: K vs f"},
    "Stewart & Gee (y vs B²)": {"x_label": "B²", "x_unit": "T²", "y_label": "y", "y_unit": "m", "title": "y vs B²"},
    "Malus's Law (I vs cos²θ)": {"x_label": "cos²θ", "x_unit": "", "y_label": "Intensity (I)", "y_unit": "mA", "title": "Malus's Law: Intensity vs cos²θ"},
    "Malus's Law (I vs θ)": {"x_label": "Angle (θ)", "x_unit": "°", "y_label": "Intensity (I)", "y_unit": "mA", "title": "Malus's Law: Intensity vs θ"},
}

QUADRANT_OPTIONS = {
    "Quadrant I only (+x, +y)": [1],
    "Quadrant I & II (+y, both x)": [1, 2],
    "Quadrant I & IV (+x, both y)": [1, 4],
    "All four quadrants": [1, 2, 3, 4],
}

QUADRANT_SIGN = {1: (1, 1), 2: (-1, 1), 3: (-1, -1), 4: (1, -1)}
QUADRANT_TITLE = {
    1: "Quadrant I  (+x, +y)",
    2: "Quadrant II  (\u2212x, +y)",
    3: "Quadrant III  (\u2212x, \u2212y)",
    4: "Quadrant IV  (+x, \u2212y)",
}


def parse_numbers(text):
    if not text.strip():
        return np.array([], dtype=float)
    cleaned = text.replace(",", " ").replace(";", " ").replace("\n", " ")
    return np.array([float(tok) for tok in cleaned.split()], dtype=float)


def unit_text(label, unit):
    return f"{label} ({unit})" if unit else label


def calculate_fit(x, y):
    if len(x) < 2:
        return None
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 if ss_tot == 0 else 1 - ss_res / ss_tot
    return slope, intercept, r2


def compute_axis_limits(x, y):
    # NOTE: this no longer forces 0 into the visible range.
    # Zooming tightly around the actual data (whatever its scale —
    # decimals, fractions converted to decimals, large or small numbers)
    # is what keeps the points centered instead of crushed into a corner.
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))

    x_range = x_max - x_min
    y_range = y_max - y_min

    # if all x (or all y) values are identical, fall back to a small
    # padding based on the value itself so the point isn't glued to an edge
    x_pad = x_range * 0.15 if x_range != 0 else (abs(x_max) * 0.1 or 1.0)
    y_pad = y_range * 0.15 if y_range != 0 else (abs(y_max) * 0.1 or 1.0)

    x_lo = x_min - x_pad
    x_hi = x_max + x_pad
    y_lo = y_min - y_pad
    y_hi = y_max + y_pad
    return x_lo, x_hi, y_lo, y_hi


def add_student_info(ax, x, y, x_lo, x_hi, y_lo, y_hi, student_name, roll_no):
    """Optional label inside the plot with student name / roll number.
    Only drawn if at least one of the two fields was filled in.
    Placed on the right side, but auto-picks top or bottom — whichever
    corner has fewer data points nearby — so it doesn't sit on top of
    the readings or the best-fit line/legend."""
    parts = []
    if student_name.strip():
        parts.append(f"Name: {student_name.strip()}")
    if roll_no.strip():
        parts.append(f"Roll No.: {roll_no.strip()}")
    if not parts:
        return

    x_mid = (x_lo + x_hi) / 2
    y_mid = (y_lo + y_hi) / 2
    top_right_count = int(np.sum((x >= x_mid) & (y >= y_mid)))
    bottom_right_count = int(np.sum((x >= x_mid) & (y < y_mid)))

    if top_right_count <= bottom_right_count:
        y_pos, va = 0.96, "top"
    else:
        y_pos, va = 0.04, "bottom"

    ax.text(
        0.97, y_pos, "\n".join(parts),
        transform=ax.transAxes, ha="right", va=va,
        fontsize=9, color="dimgray",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.75, edgecolor="lightgray"),
        zorder=5,
    )


def draw_common(ax, x, y, title, xlabel, ylabel, grid_enabled, point_labels, student_name="", roll_no=""):
    x_lo, x_hi, y_lo, y_hi = compute_axis_limits(x, y)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # only draw the x=0 / y=0 reference lines if zero actually falls
    # inside the current view — otherwise they get skipped instead of
    # forcing zero into the frame
    if x_lo <= 0 <= x_hi:
        ax.axvline(0, color="black", linewidth=1)
    if y_lo <= 0 <= y_hi:
        ax.axhline(0, color="black", linewidth=1)

    if point_labels:
        for i, (xx, yy) in enumerate(zip(x, y), start=1):
            ax.annotate(str(i), (xx, yy), xytext=(6, 6), textcoords="offset points", fontsize=9)

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(grid_enabled, alpha=0.3)

    add_student_info(ax, x, y, x_lo, x_hi, y_lo, y_hi, student_name, roll_no)


def make_figure(x, y, title, xlabel, ylabel, fit_enabled, grid_enabled, point_labels, student_name="", roll_no=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(x, y, s=65, marker="o", label="Experimental readings", zorder=3)

    if fit_enabled and len(x) >= 2:
        fit = calculate_fit(x, y)
        if fit:
            slope, intercept, r2 = fit
            x_line = np.linspace(np.min(x), np.max(x), 200)
            y_line = slope * x_line + intercept
            ax.plot(x_line, y_line, linewidth=2, label=f"Best fit: y = {slope:.4g}x + {intercept:.4g}")

    draw_common(ax, x, y, title, xlabel, ylabel, grid_enabled, point_labels, student_name, roll_no)
    ax.legend()
    fig.tight_layout()
    return fig


def make_connected_figure(x, y, title, xlabel, ylabel, grid_enabled, point_labels, student_name="", roll_no=""):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(x, y, marker="o", linewidth=2, markersize=6, label="Experimental readings")
    draw_common(ax, x, y, title, xlabel, ylabel, grid_enabled, point_labels, student_name, roll_no)
    ax.legend()
    fig.tight_layout()
    return fig


with st.sidebar:
    st.header("⚙️ Graph Settings")

    preset = st.selectbox("Experiment", list(PRESETS.keys()))
    p = PRESETS[preset]

    st.subheader("Axis labels")
    if preset == "Custom Experiment":
        x_label = st.text_input("X-axis quantity", p["x_label"])
        x_unit = st.text_input("X-axis unit", p["x_unit"])
        y_label = st.text_input("Y-axis quantity", p["y_label"])
        y_unit = st.text_input("Y-axis unit", p["y_unit"])
        title = st.text_input("Graph title", p["title"])
    else:
        x_label, x_unit, y_label, y_unit = p["x_label"], p["x_unit"], p["y_label"], p["y_unit"]
        title = st.text_input("Graph title", p["title"])

    st.subheader("Quadrants")
    quadrant_choice = st.selectbox(
        "Which quadrants should the graph cover?",
        list(QUADRANT_OPTIONS.keys()),
        index=0,
        help="Select more than one quadrant for wave diagrams or experiments with negative x/y values (e.g. u-v graphs)."
    )
    allowed_quadrants = QUADRANT_OPTIONS[quadrant_choice]

    graph_type = st.selectbox("Graph style", ["Scatter + best-fit line", "Scatter only", "Connected points"])
    grid_enabled = st.checkbox("Show grid", True)
    point_labels = st.checkbox("Number data points", False)

    st.subheader("🧑‍🎓 Student Details (optional)")
    student_name = st.text_input("Your Name", "", placeholder="e.g. Rudransh Sharma")
    roll_no = st.text_input("Roll No.", "", placeholder="e.g. 21")
    st.caption("Leave blank if you don't want this shown on the graph.")

    if preset == "Malus's Law (I vs θ)":
        st.warning(
            "I vs θ is not a straight line (I ∝ cos²θ). "
            "Use 'Scatter only' or 'Connected points' instead of the best-fit line here — "
            "for a linear check of Malus's Law, use the 'I vs cos²θ' preset instead."
        )

    st.divider()
    st.info(
        "Tip: For a straight-line experiment, use 'Scatter + best-fit line'. "
        "For wave diagrams or data that goes negative, pick more than one quadrant "
        "and fill in the readings under each quadrant separately below."
    )

st.divider()
st.subheader("📥 Experimental Data")

parse_error = None
length_mismatch = False

if len(allowed_quadrants) == 1:
    st.write("Enter X and Y readings. Use commas, spaces, semicolons, or new lines.")
    c1, c2 = st.columns(2)
    with c1:
        x_text = st.text_area(f"{x_label}" + (f" [{x_unit}]" if x_unit else ""), "1, 2, 3, 4, 5", height=150)
    with c2:
        y_text = st.text_area(f"{y_label}" + (f" [{y_unit}]" if y_unit else ""), "2, 4, 6, 8, 10", height=150)

    try:
        xq = parse_numbers(x_text)
        yq = parse_numbers(y_text)
        sx, sy = QUADRANT_SIGN[allowed_quadrants[0]]
        x_all = np.abs(xq) * sx
        y_all = np.abs(yq) * sy
        quadrant_ids = [allowed_quadrants[0]] * len(x_all)
        length_mismatch = len(xq) != len(yq)
    except ValueError as e:
        x_all, y_all, quadrant_ids = np.array([]), np.array([]), []
        parse_error = str(e)

else:
    st.write(
        "This graph spans more than one quadrant, so enter readings **as positive "
        "magnitudes** separately for each quadrant below — the correct sign (+/-) "
        "is applied automatically based on the quadrant."
    )
    x_parts, y_parts, quadrant_ids = [], [], []

    for q in allowed_quadrants:
        sx, sy = QUADRANT_SIGN[q]
        with st.expander(QUADRANT_TITLE[q], expanded=(q == 1)):
            c1, c2 = st.columns(2)
            default_x = "1, 2, 3, 4, 5" if q == 1 else ""
            default_y = "2, 4, 6, 8, 10" if q == 1 else ""
            with c1:
                xt = st.text_area(
                    f"{x_label} magnitude" + (f" [{x_unit}]" if x_unit else ""),
                    default_x, height=120, key=f"x_{q}"
                )
            with c2:
                yt = st.text_area(
                    f"{y_label} magnitude" + (f" [{y_unit}]" if y_unit else ""),
                    default_y, height=120, key=f"y_{q}"
                )
            try:
                xq = parse_numbers(xt)
                yq = parse_numbers(yt)
            except ValueError as e:
                parse_error = f"Quadrant {q}: {e}"
                continue
            if len(xq) != len(yq):
                length_mismatch = True
                st.error(f"In {QUADRANT_TITLE[q]}, X has {len(xq)} readings but Y has {len(yq)} readings.")
                continue
            x_parts.append(np.abs(xq) * sx)
            y_parts.append(np.abs(yq) * sy)
            quadrant_ids.extend([q] * len(xq))

    x_all = np.concatenate(x_parts) if x_parts else np.array([])
    y_all = np.concatenate(y_parts) if y_parts else np.array([])

st.divider()
left, right = st.columns([1, 1])

with left:
    st.subheader("🧮 Analysis")
    valid = False
    if parse_error:
        st.error(f"Could not read the data: {parse_error}")
    elif length_mismatch:
        st.error("Fix the mismatched reading counts above before continuing.")
    elif len(x_all) < 2:
        st.warning("Enter at least 2 pairs of readings (across the quadrant(s) above).")
    elif not np.all(np.isfinite(x_all)) or not np.all(np.isfinite(y_all)):
        st.error("All readings must be finite numbers.")
    else:
        valid = True
        st.success(f"✓ {len(x_all)} reading pairs loaded across {len(set(quadrant_ids))} quadrant(s).")

        fit = calculate_fit(x_all, y_all)
        if fit:
            slope, intercept, r2 = fit
            c1, c2, c3 = st.columns(3)
            c1.metric("Slope", f"{slope:.6g}")
            c2.metric("Intercept", f"{intercept:.6g}")
            c3.metric("R²", f"{r2:.6f}")
            st.latex(rf"y = {slope:.6g}x {'+' if intercept >= 0 else '-'} {abs(intercept):.6g}")

with right:
    st.subheader("📋 Combined reading table")
    if len(x_all) > 0:
        xlabel_preview = unit_text(x_label, x_unit)
        ylabel_preview = unit_text(y_label, y_unit)
        table = pd.DataFrame({xlabel_preview: x_all, ylabel_preview: y_all, "Quadrant": quadrant_ids})
        st.dataframe(table, use_container_width=True, height=220)
    else:
        st.write("No readings yet.")

st.divider()
st.subheader("📊 Generated Graph")

if valid:
    xlabel = unit_text(x_label, x_unit)
    ylabel = unit_text(y_label, y_unit)

    if graph_type == "Scatter + best-fit line":
        fig = make_figure(x_all, y_all, title, xlabel, ylabel, True, grid_enabled, point_labels,
                           student_name, roll_no)
    elif graph_type == "Scatter only":
        fig = make_figure(x_all, y_all, title, xlabel, ylabel, False, grid_enabled, point_labels,
                           student_name, roll_no)
    else:
        fig = make_connected_figure(x_all, y_all, title, xlabel, ylabel, grid_enabled, point_labels,
                                     student_name, roll_no)

    st.pyplot(fig, clear_figure=False)

    png_buffer = io.BytesIO()
    fig.savefig(png_buffer, format="png", dpi=300, bbox_inches="tight")
    png_buffer.seek(0)

    pdf_buffer = io.BytesIO()
    fig.savefig(pdf_buffer, format="pdf", bbox_inches="tight")
    pdf_buffer.seek(0)

    d1, d2 = st.columns(2)
    with d1:
        st.download_button("⬇️ Download PNG", data=png_buffer, file_name="physics_lab_graph.png",
                            mime="image/png", use_container_width=True)
    with d2:
        st.download_button("⬇️ Download PDF", data=pdf_buffer, file_name="physics_lab_graph.pdf",
                            mime="application/pdf", use_container_width=True)

    plt.close(fig)
else:
    st.info("Enter valid readings above to generate the graph.")

st.divider()
st.caption("Physics Lab Graph Generator • Built with Python, Streamlit, NumPy and Matplotlib")