import re

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sympy as sp


TEC_BLUE = "#0039A6"
TEC_LIGHT_BLUE = "#4DA3FF"
ORANGE = "#F28C28"
GRID = "#D9E2F2"

x = sp.Symbol("x", real=True)

ALLOWED_NAMES = {
    "x": x,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "pi": sp.pi,
    "e": sp.E,
}


def parse_function(text: str) -> sp.Expr:
    """Parse a short mathematical expression with a restricted vocabulary."""
    cleaned = text.strip().lower().replace("^", "**")
    if not cleaned or len(cleaned) > 120:
        raise ValueError("La expresión está vacía o es demasiado larga.")
    if "__" in cleaned or not re.fullmatch(r"[0-9a-z+\-*/().,\s*]+", cleaned):
        raise ValueError("La expresión contiene caracteres no permitidos.")

    names = set(re.findall(r"[a-z]+", cleaned))
    unknown = names.difference(ALLOWED_NAMES)
    if unknown:
        raise ValueError(f"Nombre no permitido: {', '.join(sorted(unknown))}.")

    expr = sp.sympify(cleaned, locals=ALLOWED_NAMES)
    if expr.free_symbols - {x}:
        raise ValueError("La función solo puede depender de x.")
    if sp.count_ops(expr) > 60:
        raise ValueError("La función es demasiado compleja para esta demostración.")
    return expr


def finite_real(value) -> float:
    evaluated = complex(sp.N(value))
    if abs(evaluated.imag) > 1e-9 or not np.isfinite(evaluated.real):
        raise ValueError("El resultado no es un número real finito.")
    return float(evaluated.real)


def evaluate_grid(expr: sp.Expr, values: np.ndarray) -> np.ndarray:
    fn = sp.lambdify(x, expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        raw = fn(values)
    raw = np.asarray(raw)
    if raw.ndim == 0:
        raw = np.full(values.shape, raw)
    raw = np.real_if_close(raw).astype(float)
    raw[~np.isfinite(raw)] = np.nan
    return raw


def common_layout(title: str, xmin: float, xmax: float) -> dict:
    return {
        "title": {"text": title, "x": 0.02},
        "xaxis": {"title": "x", "range": [xmin, xmax], "gridcolor": GRID},
        "yaxis": {"title": "y", "gridcolor": GRID, "zerolinecolor": "#AAB7C4"},
        "template": "plotly_white",
        "height": 520,
        "margin": {"l": 35, "r": 25, "t": 65, "b": 35},
        "legend": {"orientation": "h", "y": 1.08, "x": 0.52, "xanchor": "center"},
    }


def animation_buttons() -> list:
    return [
        {
            "type": "buttons",
            "direction": "left",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.13,
            "buttons": [
                {
                    "label": "▶ Reproducir",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 180, "redraw": True},
                                     "transition": {"duration": 70},
                                     "fromcurrent": True}],
                },
                {
                    "label": "⏸ Pausar",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                       "mode": "immediate"}],
                },
            ],
        }
    ]


def static_figure(expr, derivative, xmin, xmax, x0):
    xs = np.linspace(xmin, xmax, 700)
    ys = evaluate_grid(expr, xs)
    y0 = finite_real(expr.subs(x, x0))
    slope = finite_real(derivative.subs(x, x0))
    tangent = y0 + slope * (xs - x0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, name="f(x)", line={"color": TEC_BLUE, "width": 3}))
    fig.add_trace(go.Scatter(x=xs, y=tangent, name="Tangente", line={"color": ORANGE, "width": 2, "dash": "dash"}))
    fig.add_trace(go.Scatter(x=[x0], y=[y0], name="Punto", mode="markers",
                             marker={"size": 11, "color": "#D7263D"}))
    fig.update_layout(**common_layout("Función y recta tangente", xmin, xmax))
    return fig


def tangent_animation(expr, derivative, xmin, xmax):
    xs = np.linspace(xmin, xmax, 500)
    ys = evaluate_grid(expr, xs)
    positions = np.linspace(xmin, xmax, 36)
    frames = []

    for pos in positions:
        try:
            y_pos = finite_real(expr.subs(x, float(pos)))
            slope = finite_real(derivative.subs(x, float(pos)))
            tangent = y_pos + slope * (xs - pos)
        except (ValueError, TypeError):
            y_pos, tangent = np.nan, np.full_like(xs, np.nan)
        frames.append(go.Frame(
            name=f"{pos:.3g}",
            data=[
                go.Scatter(x=xs, y=ys),
                go.Scatter(x=xs, y=tangent),
                go.Scatter(x=[pos], y=[y_pos]),
            ],
        ))

    fig = go.Figure(
        data=[
            go.Scatter(x=xs, y=ys, name="f(x)", line={"color": TEC_BLUE, "width": 3}),
            go.Scatter(x=xs, y=frames[0].data[1].y, name="Tangente móvil",
                       line={"color": ORANGE, "width": 2}),
            go.Scatter(x=[positions[0]], y=frames[0].data[2].y, name="Punto móvil",
                       mode="markers", marker={"size": 12, "color": "#D7263D"}),
        ],
        frames=frames,
    )
    layout = common_layout("La derivada como pendiente local", xmin, xmax)
    layout["updatemenus"] = animation_buttons()
    fig.update_layout(**layout)
    return fig


def secant_animation(expr, derivative, xmin, xmax, x0):
    xs = np.linspace(xmin, xmax, 500)
    ys = evaluate_grid(expr, xs)
    y0 = finite_real(expr.subs(x, x0))
    span = xmax - xmin
    direction = 1.0 if xmax - x0 >= x0 - xmin else -1.0
    max_h = max(min(span * 0.35, (xmax - x0) if direction > 0 else (x0 - xmin)), span * 0.02)
    h_values = direction * np.geomspace(max_h, max(span * 0.002, 1e-5), 34)
    frames = []

    for h in h_values:
        x1 = x0 + h
        try:
            y1 = finite_real(expr.subs(x, float(x1)))
            secant_slope = (y1 - y0) / h
            secant = y0 + secant_slope * (xs - x0)
            label = f"h={h:.4g}; pendiente={secant_slope:.4g}"
        except (ValueError, TypeError):
            y1, secant = np.nan, np.full_like(xs, np.nan)
            label = f"h={h:.4g}"
        frames.append(go.Frame(
            name=label,
            data=[
                go.Scatter(x=xs, y=ys),
                go.Scatter(x=xs, y=secant),
                go.Scatter(x=[x0, x1], y=[y0, y1]),
            ],
            layout={"title": {"text": f"Secante → tangente · {label}", "x": 0.02}},
        ))

    fig = go.Figure(
        data=[
            go.Scatter(x=xs, y=ys, name="f(x)", line={"color": TEC_BLUE, "width": 3}),
            go.Scatter(x=xs, y=frames[0].data[1].y, name="Recta secante",
                       line={"color": TEC_LIGHT_BLUE, "width": 2}),
            go.Scatter(x=[x0, x0 + h_values[0]], y=frames[0].data[2].y,
                       name="Puntos", mode="markers+lines",
                       marker={"size": 10, "color": "#D7263D"},
                       line={"color": "#D7263D", "dash": "dot"}),
        ],
        frames=frames,
    )
    layout = common_layout("Construcción de la derivada con rectas secantes", xmin, xmax)
    layout["updatemenus"] = animation_buttons()
    fig.update_layout(**layout)
    return fig


st.set_page_config(page_title="Laboratorio de cálculo diferencial", page_icon="📈", layout="wide")
st.markdown(
    f"""
    <style>
      [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2 {{color: {TEC_BLUE};}}
      .stButton button {{background-color: {TEC_BLUE}; color: white; border: 0;}}
      .formula-card {{padding: 1rem 1.2rem; border-left: 6px solid {TEC_BLUE};
                      background: #F4F7FC; border-radius: 6px; margin-bottom: 1rem;}}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📈 Laboratorio de cálculo diferencial")
st.caption("Explora la derivada como expresión, valor numérico y pendiente de la recta tangente.")

with st.sidebar:
    st.header("Controles")
    function_text = st.text_input("Función f(x)", value="sin(x) + x^2/4")
    st.caption("Admite: +, −, *, /, ^, sin, cos, tan, exp, log, sqrt y abs.")
    x0 = st.number_input("Punto de evaluación x₀", value=1.0, step=0.1, format="%.3f")
    col_a, col_b = st.columns(2)
    xmin = col_a.number_input("Inicio", value=-5.0, step=0.5)
    xmax = col_b.number_input("Final", value=5.0, step=0.5)
    st.info("Consejo docente: reproduce primero la secante y luego la tangente móvil.")

if xmin >= xmax:
    st.error("El inicio del intervalo debe ser menor que el final.")
    st.stop()
if not xmin <= x0 <= xmax:
    st.error("El punto x₀ debe estar dentro del intervalo.")
    st.stop()

try:
    expression = parse_function(function_text)
    derivative = sp.diff(expression, x)
    value_at_x0 = finite_real(expression.subs(x, x0))
    derivative_at_x0 = finite_real(derivative.subs(x, x0))
except Exception as exc:
    st.error(f"No se pudo procesar la función: {exc}")
    st.stop()

st.markdown('<div class="formula-card">', unsafe_allow_html=True)
formula_col, deriv_col = st.columns(2)
formula_col.markdown("**Función**")
formula_col.latex(r"f(x)=" + sp.latex(expression))
deriv_col.markdown("**Derivada simbólica**")
deriv_col.latex(r"f'(x)=" + sp.latex(derivative))
st.markdown("</div>", unsafe_allow_html=True)

m1, m2, m3 = st.columns(3)
m1.metric("x₀", f"{x0:.4g}")
m2.metric("f(x₀)", f"{value_at_x0:.6g}")
m3.metric("f′(x₀) = pendiente", f"{derivative_at_x0:.6g}")

tab_static, tab_secant, tab_tangent, tab_guide = st.tabs(
    ["Gráfica estática", "Secante → tangente", "Tangente móvil", "Guía didáctica"]
)

with tab_static:
    st.plotly_chart(static_figure(expression, derivative, xmin, xmax, x0), width="stretch")
    st.latex(
        rf"y-{value_at_x0:.5g}={derivative_at_x0:.5g}(x-{x0:.5g})"
    )

with tab_secant:
    st.write("Observa cómo, al disminuir **h**, la pendiente de la secante se aproxima a la derivada.")
    st.plotly_chart(secant_animation(expression, derivative, xmin, xmax, x0), width="stretch")
    st.latex(r"f'(x_0)=\lim_{h\to 0}\frac{f(x_0+h)-f(x_0)}{h}")

with tab_tangent:
    st.write("La recta tangente y su punto de contacto recorren el intervalo seleccionado.")
    st.plotly_chart(tangent_animation(expression, derivative, xmin, xmax), width="stretch")

with tab_guide:
    st.subheader("Secuencia sugerida para una demostración de 15 minutos")
    st.markdown(
        """
        1. **Predicción:** pregunta qué representa la pendiente en un punto (2 min).
        2. **Cálculo simbólico:** compara la función con su derivada (3 min).
        3. **Aproximación:** reproduce la animación de secantes y explica el límite (4 min).
        4. **Interpretación geométrica:** reproduce la tangente móvil (3 min).
        5. **Exploración:** cambia la función o el punto y pide una conclusión (3 min).
        """
    )
    st.warning("Las discontinuidades y los puntos donde la derivada no existe deben analizarse por separado.")

st.divider()
st.caption("Recurso educativo interactivo · Cálculo diferencial")
