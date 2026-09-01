# Laboratorio de cálculo diferencial

Aplicación educativa creada con Streamlit, SymPy y Plotly. Permite:

- introducir una función de una variable;
- calcular su derivada simbólica;
- evaluar la función y la derivada en un punto;
- mostrar la función y su recta tangente;
- animar la aproximación de secantes a la tangente;
- animar una recta tangente que recorre el intervalo.

## Ejecutarla en la computadora

Se necesita Python 3.10 o superior. Desde la carpeta del proyecto:

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

En macOS o Linux:

```bash
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Subirla a GitHub

1. Crea un repositorio público en GitHub.
2. Sube `app.py`, `requirements.txt` y `README.md`.
3. Confirma los cambios con **Commit changes**.

## Publicarla para los estudiantes

1. Entra en https://share.streamlit.io e inicia sesión con GitHub.
2. Selecciona **Create app**.
3. Elige el repositorio y la rama principal.
4. Especifica `app.py` como archivo de entrada.
5. Pulsa **Deploy** y comparte el enlace generado.

## Sintaxis admitida

Ejemplos:

```text
x^2 + 3*x - 1
sin(x) + x^2/4
exp(-x^2)
log(x)
sqrt(x + 2)
```

Funciones admitidas: `sin`, `cos`, `tan`, `exp`, `log`, `ln`, `sqrt` y `abs`.

## Observación didáctica

En funciones con discontinuidades, restricciones de dominio o puntos no derivables, la gráfica puede mostrar interrupciones. Estos casos son útiles para discutir las condiciones de existencia de la derivada.
