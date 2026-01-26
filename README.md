# Astropanoptes

Astropanoptes es un prototipo de control para astrofotografía que integra:
- Captura desde cámaras Player One (SDK vía `pyPOACamera`).
- Control de montura con firmware Arduino (comandos de microsteps, rate y move).
- Tracking por correlación de fase (OpenCV) en un loop de control en tiempo real.
- Una UI en Jupyter/ipywidgets para operación básica.

Este README describe **toda la estructura del repositorio** y explica cada módulo, incluyendo los aún pendientes de implementar.

## Estructura del repositorio

> Ruta raíz: `/workspace/astropanoptes`

- `README.md`: esta documentación.
- `LICENSE`: licencia del proyecto.
- `app.ipynb`: notebook de ejemplo para levantar UI y ejecutar el runner.
- `app_runner.py`: orquestador principal de runtime; loop de control, captura, preview y tracking.
- `ui.py`: UI en ipywidgets (estado, preview, control manual de montura, parámetros de cámara).
- `actions.py`: definición de acciones y factories (connect, set params, tracking, stacking, platesolve, goto).
- `ap_types.py`: tipos compartidos (ejes, modos, `Frame`, `SystemState`).
- `config.py`: configuración de cámara, preview, montura, tracking, stacking, platesolve y app.
- `camera_poa.py`: wrapper de alto nivel para cámara Player One (I/O, configuración, stream).
- `pyPOACamera.py`: wrapper ctypes del SDK Player One (loader multiplataforma + constantes/structs).
- `libPlayerOneCamera.3.9.0.dylib`: binario del SDK (macOS). En Linux/Windows se esperan `.so`/`.dll`.
- `PlayerOneCamera.h`: header del SDK (referencia de API).
- `imaging.py`: utilidades de imagen (downsample, stretch rápido, preview JPEG, canal verde de Bayer).
- `tracking.py`: pipeline de tracking (preprocesado, correlación de fase, control PI, auto-calibración, rate limiter).
- `mount_arduino.py`: driver de montura vía serial (protocolo Arduino, comandos RATE/MOVE/MS/STOP).
- `mount_firmware.ino`: firmware Arduino para la montura (lado microcontrolador).
- `logging_utils.py`: logging liviano a stdout o `ipywidgets.Output`.

## Módulos actuales (qué hacen)

### 1) Orquestación y UI
- **`app_runner.py`**
  - Controla el lifecycle de cámara, stream y montura.
  - Ejecuta el loop de control a `control_hz`, genera previews y aplica tracking.
  - Mantiene el estado global (`SystemState`) para la UI.

- **`ui.py`**
  - Construye la UI en Jupyter (botones de conexión, estados, live view).
  - Incluye controles manuales de montura (microsteps, move, stop).
  - Refleja métricas de tracking cuando está activo.

### 2) Tipos, acciones y configuración
- **`ap_types.py`**
  - Enum de ejes (`Axis`), modos de display y estructura de `Frame`.
  - `SystemState` agrupa indicadores de estado y métricas (FPS, tracking, stacking, platesolve).

- **`actions.py`**
  - Enum `ActionType` y factories para eventos (cámara, montura, tracking, stacking, platesolve, goto).
  - Sirve como contrato entre UI y runner (todo pasa por cola de acciones).

- **`config.py`**
  - Configs declarativas para cámara, preview, montura, tracking, stacking y platesolve.
  - `AppConfig` agrega todo en una sola estructura.

### 3) Cámara y procesamiento de imagen
- **`camera_poa.py`**
  - Envoltura de la cámara Player One: configuración de ROI/binning/formato, inicio de exposición y lectura.
  - Ofrece `CameraStream` para captura continua y preview.

- **`pyPOACamera.py`**
  - Binding ctypes con el SDK oficial (carga dinámica según plataforma).
  - Define enums, structs y funciones del driver.

- **`imaging.py`**
  - Utilidades rápidas para preview: downsample por stride, stretch por percentiles, JPEG encode.
  - Extracción rápida del canal verde desde Bayer RAW16.

### 4) Tracking y control de montura
- **`tracking.py`**
  - Tracking incremental por correlación de fase.
  - Control PI y rate limiter para generar velocidades de montura (µsteps/s).
  - Soporte de calibración manual + auto-cal (RLS) y bootstrap.

- **`mount_arduino.py`**
  - Conexión serial y protocolo con el firmware Arduino.
  - Comandos: `PING`, `ENABLE`, `STOP`, `RATE`, `MS`, `MOVE`, `STATUS`.

- **`mount_firmware.ino`**
  - Firmware del microcontrolador compatible con el protocolo anterior.

### 5) Logging
- **`logging_utils.py`**
  - Abstracción simple de logs para consola o widgets.

## Módulos pendientes (por implementar)

> Estos componentes están **declarados en tipos/configs/actions**, pero aún no existen como módulos dedicados o no están integrados en el runner/UI:

1) **Stacking (apilado)**
   - `actions.py` define `STACKING_*` y `SystemState` tiene métricas de stacking.
   - En `ui.py` hay botones/estados deshabilitados para stacking.
   - Falta un módulo que: alinee frames, integre/guarde resultados, y exponga métricas.

2) **Gestión de calibraciones persistentes**
   - `tracking.py` soporta autocal y bootstrap en memoria.
   - No existe aún persistencia a disco ni herramientas de export/import.

## Flujo general (alto nivel)

1. **UI** genera acciones (`actions.py`).
2. **AppRunner** consume acciones y coordina cámara, preview y montura.
3. **Tracking** procesa frames y emite rates a la montura.
4. **Estado** se refleja en `SystemState` y vuelve a la UI.

## Requisitos (implícitos)

- Python con `numpy`, `opencv-python`, `ipywidgets`, `pyserial`.
- SDK de Player One Camera disponible en la plataforma (binarios `.dll/.so/.dylib`).
- Arduino con firmware de `mount_firmware.ino` cargado.

---

Si necesitas ampliar este README con pasos de instalación o documentación de la API, indícalo y lo agrego.
