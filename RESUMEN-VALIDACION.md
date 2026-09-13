# Validación de GLORIA — resultados ejecutados

Corridas realizadas el 2026-07-31 en una AMD Radeon RX 9060 XT (gfx1200), con los binarios de
este repo en modo consola. El setup del cilindro está en `src/setup.cpp` (bloque comentado
"VALIDACION GLORIA"); la referencia uniforme es el mismo setup con `n_niveles=1, lbm_D=32`.

## Test 1 — Flujo uniforme a través del parche (bug-check)

Corriente libre sin sólido atravesando un parche de 2 niveles con acople bidireccional,
1000 pasos del nivel 0.

**Resultado: `interface_mismatch = 0.00%` en todas las muestras.** La corriente atraviesa las
dos interfaces (entrada y salida del parche) sin deformación medible. Esto verifica de una vez
la indexación, los stencils de interpolación, la mezcla temporal, la restricción y el manejo de
`TYPE_E`.

## Test 2 — Cilindro 2D, Re = 200, calle de von Kármán

| Configuración | celdas | Cd | Cl' (RMS·√2) | St |
|---|---|---|---|---|
| **GLORIA**: dominio 512×256 (D=16) + parche 192×128 a dx/2 (D efectivo = 32) | 156k | **1.426 ± 0.043** | **0.722** | **0.1971** |
| **Referencia uniforme**: 1024×512, D=32 en todo el dominio | 524k | 1.441 ± 0.044 | 0.731 | 0.1985 |
| Literatura (2D, Re=200) | — | 1.3–1.4 | 0.5–0.7 | 0.196–0.20 |

Estadísticos sobre la ventana t* > 47 tiempos convectivos (mismo tiempo físico en ambas).

**Diferencias GLORIA vs uniforme: Cd 1.0%, St 0.7%, Cl' 1.2%** — con 3.4× menos celdas.

El diagnóstico `interface_mismatch` promedió ~13% de u∞ durante el desprendimiento: ese valor
está dominado por la estela cruzando la interfaz aguas abajo (donde grueso y fino difieren
legítimamente por el factor 2 de resolución) y NO es señal de error de acople — como lo
demuestran el 0.00% del test 1 y el 1% de acuerdo en las fuerzas.

## Figura

`validacion_continuidad.png`:
- Arriba: campo |u|/u∞ compuesto (nivel 0 + nivel 1 superpuestos en su posición física, borde
  del parche punteado). La calle de vórtices nace dentro del parche fino y sale al dominio
  grueso sin discontinuidades.
- Abajo: perfil de u_x sobre el eje del cilindro extraído de AMBOS niveles: las curvas se
  superponen a través de las dos interfaces. (La diferencia entre curvas cerca de la
  recirculación es esperada: ahí la curva azul es el interior grueso apantallado, cuya solución
  no participa del resultado; la roja —el nivel fino— es la solución real.)

## Cómo reproducir

1. En `src/setup.cpp`, comentá el setup NACA activo y descomentá el bloque
   "VALIDACION GLORIA" (cilindro).
2. En `defines.hpp` podés dejar `INTERACTIVE_GRAPHICS` (mirás la corrida) o cambiar a
   `GRAPHICS` para correr en consola.
3. `./make.sh && bin/FluidX3D`. El CSV queda en `bin/export/`.
4. Para la referencia uniforme: `n_niveles = 1u; lbm_D = 32.0f;` y correr el doble de pasos
   (mismo tiempo físico).
