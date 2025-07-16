# 🚀 Zeta_RD: Heurística RD9 para Validar Ceros de la Función Zeta de Riemann 🔍

Esta herramienta numérica usa raíces digitales en base 9 (RD9) de fases sinusoidales {sin(π t / t_n)} sobre los primeros M ceros bajos (de Odlyzko) para localizar mínimos que aproximan ceros ultra-altos (~10^22). Con refinamiento parabólico, logra error promedio ~0.05 (menos de la mitad del espaciado local), 100% <0.1 en tests. 📊 Refuerza la Hipótesis de Riemann numéricamente —¡original y escalable en GPU!

Validado por @grok: 'Impresionante robustez —original, preciso y listo para explorar HR!' 😎

## Requisitos
- Python 3 con bibliotecas: NumPy, Matplotlib, SciPy, Torch, Pandas, TQDM. Las pruebas las se realizaron en Runpod con 1 x A100 PCIe en un Macbook Air Apple M2 8GB 
18 vCPU 215 GB RAM.
- GPU recomendada para CUDA (acelera validaciones).

## Cómo ejecutar
1. Carga `ceros_10^22.rtf` en el directorio (incluido en este repo).
2. Ejecuta: `python zeta_RD_refinada_parabola.py`.
3. Revisa el plot generado (`grafico_zeta_rd_refinada_parabola.png`) y CSV (`errores_ceros_10^22_RD9_refinada_parabola.csv`).

## Resultados clave
- Error mínimo: 0.001590
- Error máximo: 0.097948
- Error promedio: 0.052875
- 100/100 (100.0%) con error < 0.1

#RiemannHypothesis #Math #xAI
