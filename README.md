# Astropanoptes

Astropanoptes es un prototipo de control para astrofotografía que integra:
- Captura desde cámaras Player One (SDK vía `pyPOACamera`).
- Control de montura con firmware ESP32 (Bluetooth Classic SPP) para TMC2209 STEP/DIR.
- Tracking por correlación de fase (OpenCV) en un loop de control en tiempo real.
- Una UI en PyQt6 para operación básica.

Este README describe **toda la estructura del repositorio** y explica cada módulo, incluyendo los aún pendientes de implementar.

## Estructura del repositorio

> Ruta raíz: `/workspace/astropanoptes`

- `README.md`: esta documentación.
- `LICENSE`: licencia del proyecto.
- `app_runner.py`: orquestador principal de runtime; loop de control, captura, preview y tracking.
- `app.py`: entrypoint principal; inicia la UI PyQt6 o el modo terminal.
- `terminal_app.py`: consola interactiva y automatizable para operación y diagnóstico.
- `ui/pyqt6_app.py`: implementación principal de la UI de escritorio.
- `actions.py`: definición de acciones y factories (connect, set params, tracking, stacking, platesolving, goto).
- `ap_types.py`: tipos compartidos (ejes, modos, `Frame`, `AppState`).
- `config.py`: configuración de cámara, preview, montura, tracking, stacking, platesolving y app.
- `camera_poa.py`: wrapper de alto nivel para cámara Player One (I/O, configuración, stream).
- `pyPOACamera.py`: wrapper ctypes del SDK Player One (loader multiplataforma + constantes/structs).
- `libPlayerOneCamera.3.9.0.dylib`: binario del SDK (macOS). En Linux/Windows se esperan `.so`/`.dll`.
- `PlayerOneCamera.h`: header del SDK (referencia de API).
- `imaging.py`: utilidades de imagen (stretch rápido, preview JPEG, canal verde de Bayer).
- `tracking.py`: pipeline de tracking (preprocesado, correlación de fase, control PI, auto-calibración, rate limiter).
- `stacking.py`: live stacking con alineación, drizzle opcional, salida mono/RGB y preview JPEG.
- `platesolving.py`: plate solving contra Gaia/SIMBAD, overlay/debug y worker asíncrono.
- `goto.py`: modelo de apuntado, sync, GoTo y rutinas de calibración/autocalibración.
- `gaia_cache.py`: catálogo combinado Gaia DR3 + Hipparcos/Tycho-2, caché HEALPix,
  credenciales y resolución de nombres.
- `simulation.py`: modo demo sin hardware; simula cámara, montura, tracking/GoTo y campos estelares Gaia.
- `mount_arduino.py`: driver de montura vía serial (puerto SPP), comandos `PING/ENABLE/STOP/MOVE/STATUS`; `MS 64` se conserva como handshake compatible.
- `mount_firmware/mount_firmware.ino`: firmware ESP32 para la montura (lado microcontrolador).
- `logging_utils.py`: logging liviano a stdout o sink global de la UI.

## Módulos actuales (qué hacen)

### 1) Orquestación, UI y terminal
- **`app_runner.py`**
  - Controla el lifecycle de cámara, stream y montura.
  - Ejecuta el loop de control a `control_hz`, genera previews y aplica tracking.
  - Mantiene el estado global (`AppState`) para la UI.

- **`app.py` + `ui/pyqt6_app.py`**
  - Construye la UI en PyQt6 (botones de conexión, estados, live view).
- Incluye controles manuales de montura (move y stop). El microstepping está cableado de forma fija a 1/64.
  - Refleja métricas de tracking cuando está activo.

- **`terminal_app.py`**
  - Controla el mismo `AppRunner` sin cargar PyQt6 mediante `python app.py --cli`.
  - Ofrece consola interactiva, comandos repetibles con `-c` y archivos de sesión con `--script`.
  - Expone estado/configuración JSON, esperas por campos del estado e inyección avanzada de acciones.
  - Guarda Live, Stack y debug de Plate Solve como JPEG en `terminal_output/images/`, junto a un JSON con el estado exacto de la captura.

### 2) Tipos, acciones y configuración
- **`ap_types.py`**
  - Enum de ejes (`Axis`), modos de display y estructura de `Frame`.
  - `AppState` agrupa indicadores de estado y métricas (FPS, tracking, stacking, platesolving).

- **`actions.py`**
  - Enum `ActionType` y factories para eventos (cámara, montura, tracking, stacking, platesolving, goto).
  - Sirve como contrato entre UI y runner (todo pasa por cola de acciones).

- **`config.py`**
  - Configs declarativas para cámara, preview, montura, tracking, stacking, platesolving y simulación.
  - `AppConfig` agrega todo en una sola estructura.

### 3) Cámara y procesamiento de imagen
- **`camera_poa.py`**
  - Envoltura de la cámara Player One: configuración de ROI/binning/formato, inicio de exposición y lectura.
  - Ofrece `CameraStream` para captura continua y preview.

- **`pyPOACamera.py`**
  - Binding ctypes con el SDK oficial (carga dinámica según plataforma).
  - Define enums, structs y funciones del driver.

- **`imaging.py`**
  - Utilidades rápidas para preview: stretch por percentiles y JPEG encode.
  - Extracción rápida del canal verde desde Bayer RAW16.

### 4) Tracking y control de montura
- **`tracking.py`**
  - Tracking incremental por correlación de fase.
  - Control PI y rate limiter para generar velocidades de montura (µsteps/s).
  - Soporte de calibración manual + auto-cal (RLS) y bootstrap.

- **`mount_arduino.py`**
  - Conexión serial y protocolo con firmware ESP32 vía Bluetooth Classic SPP.
  - Comandos: `PING`, `ENABLE`, `STOP`, `MOVE`, `STATUS`, `DEBUG`; `MS 64` existe únicamente como handshake legado sin efecto físico.
  - GoTo y movimientos manuales usan un solo `MOVE` por eje para aprovechar la aceleración/frenado simétricos del firmware cargado; `delay_us` es el retardo mínimo (velocidad máxima).
  - El microstepping no es configurable por software: ambos drivers están cableados permanentemente a 1/64.

- **`mount_firmware/mount_firmware.ino`**
  - Firmware actual de ESP32 (nombre BT: `AstroPanoptes-ESP32`).
  - Pines controlados: `EN=21`, `AZ STEP/DIR=33/25`, `ALT STEP/DIR=26/27`; MS no tiene pines asignados en el firmware.

### 5) Stacking, plate solving y GoTo
- **`stacking.py`**
  - Alinea frames en vivo, acumula mosaico mono/RGB, genera preview y guarda `.npy` + `.png`.
  - Soporta drizzle x1/x2/x3 desde la UI.

- **`platesolving.py` + `gaia_cache.py`**
  - Detecta fuentes con SEP, consulta/carga Gaia, resuelve por tripletas y publica overlays/debug.
  - Usa caché en `~/.cache/gaia_cones` por defecto.
  - Hipparcos y Tycho-2 completos pueden descargarse directamente desde CDS y
    teselarse localmente sin consultas TAP:

```bash
source /Users/josue/myenv/bin/activate
python scripts/import_bright_catalogs.py --workers 6
```

- **`goto.py`**
  - Mantiene modelo de apuntado, sync desde plate solving, GoTo y calibraciones manual/auto.
  - Después de ajustar el modelo, la pestaña `GoTo` permite activar
    `Estrellas esperadas según modelo`. La vista Live dibuja en magenta las
    posiciones proyectadas desde el modelo, sin usar la última solución de
    placa; una alineación perfecta debe coincidir con las estrellas observadas.

#### Diagnóstico de plate solving y GoTo

Cada plate solve explícito y cada operación `GoTo`, `AutoCal`, estimación de
roll o ajuste del modelo crea por defecto una sesión en
`stack_output/goto_diagnostics/`. La carpeta incluye:

- los RAW16 exactos usados por SEP/plate solving (`.npy`) y los stacks de
  deriva comprimidos sin pérdida (`.npz`);
- parámetros de cámara, SEP, óptica, observador y montura;
- hashes y estadísticas de cada frame, resultados de plate solving y consenso;
- muestras y estado del modelo antes/después del fit;
- las iteraciones de planificación, pasos, delays y error previsto del GoTo;
- `timeline.jsonl` incremental y `manifest.json` final para reconstruir el flujo.

Se puede cambiar la ubicación con `GoToConfig.diagnostics_dir` y
`PlatesolvingConfig.diagnostics_dir`, o desactivar temporalmente el guardado
con `ASTROPANOPTES_DIAGNOSTICS=0`.

### 6) Modo demo / simulación
- **`simulation.py`**
  - Reemplaza cámara y montura por backends simulados cuando se activa `Demo` en la barra superior.
  - La montura física simulada arranca con inclinación aleatoria acotada por `SimulationConfig.random_mount_tilt_deg`.
  - La cámara simulada arranca con error aleatorio de roll acotado por `SimulationConfig.random_camera_roll_deg`.
  - Renderiza frames RAW16 usando Gaia DR3 hasta `G≤15` y completa el extremo brillante
    con Hipparcos/Tycho-2 hasta `V≤15`.
  - Si faltan teselas de cualquiera de los catálogos, mantiene la cámara viva pero renderiza sin estrellas y muestra error de cámara; el fallback sintético solo se usa si `SimulationConfig.allow_synthetic_fallback=True`.
  - El botón `Download Gaia field` descarga al caché las teselas Gaia e Hipparcos/Tycho-2 del campo actual; en demo usa la posición simulada real y refresca la cámara simulada al terminar.
  - Los comandos normales de tracking, GoTo, sync y calibración pasan por el mismo runner que en modo hardware.

### 7) Logging
- **`logging_utils.py`**
  - Abstracción simple de logs para consola o sink global de la UI.

## Módulos pendientes o incompletos

Quedan como mejoras de producto/operación:

1) **Gestión de calibraciones persistentes**
   - `tracking.py` soporta autocal y bootstrap en memoria.
   - No existe aún persistencia a disco ni herramientas de export/import.

2) **Empaquetado/instalación**
   - La app todavía se ejecuta como proyecto local; falta definir instalación formal y distribución multiplataforma.

## Flujo general (alto nivel)

1. **UI** genera acciones (`actions.py`).
2. **AppRunner** consume acciones y coordina cámara, preview y montura.
3. **Tracking** procesa frames y emite rates a la montura.
4. **Estado** se refleja en `AppState` y vuelve a la UI.

En modo `Demo`, el flujo es el mismo: activar el checkbox `Demo`, conectar cámara y montura desde la barra superior, y usar plate solving, tracking, GoTo y calibración como en una sesión real.

## Control y depuración desde terminal

Para abrir una sesión interactiva:

```bash
source /Users/josue/myenv/bin/activate
python app.py --cli
```

Dentro de la consola, `help` muestra todos los comandos. Por ejemplo:

```text
demo on
camera connect
mount connect
wait camera.connected true 8
tracking start
status --json
image live primera-captura
view start
health
quit
```

`image live`, `image stack` e `image platesolve` no intentan abrir una ventana. Guardan la imagen en `terminal_output/images/` y crean al lado un archivo `.json` con el estado del runtime. Se puede cambiar la carpeta con `--images-dir`.

`view start` inicia un visor de solo lectura en `http://127.0.0.1:8765/` y lo abre en el navegador. La consola permanece disponible para mover, detener y ejecutar plate solving. En paralelo se reemplaza atómicamente `terminal_output/images/live-latest.jpg`, de modo que una herramienta de diagnóstico pueda inspeccionar exactamente la imagen más reciente. `view status`, `view open` y `view stop` consultan, reabren y detienen el visor. Para scripts se puede usar `view start 8765 2 no-open`; el servidor nunca escucha fuera de `127.0.0.1`.

Los movimientos manuales aceptan dos perfiles. `smooth` (predeterminado) usa una
curva S limitada en velocidad y aceleración; `direct` aplica velocidad constante
para comparación mecánica. Ambos respetan el límite de seguridad del firmware:

```text
mount move alt 1 30000 10 smooth
mount move alt 1 30000 10 direct
```

El valor `delay_us` solicita una velocidad, pero ya no puede saltarse el límite
de 12 000 microsteps/s. Con `smooth`, la rampa comienza aproximadamente en
400 microsteps/s y limita la aceleración a 4 000 microsteps/s².

Para una sesión automatizada, se puede repetir `-c`:

```bash
python app.py --cli --demo --seed 42 --connect \
  -c "wait camera.connected true 8" \
  -c "wait mount.connected true 8" \
  -c "image live smoke 8" \
  -c "health" \
  -c "status --json"
```

Los logs internos se escriben en `stderr` y las respuestas/JSON en `stdout`, por lo que es posible guardarlos por separado. `get tracking.error_px`, por ejemplo, devuelve un único dato sin imprimir todo el estado. Una condición no satisfecha o un comando inválido hace que el modo automatizado termine con código 2.

Para operaciones asíncronas, `await platesolving` y `await goto 120` esperan específicamente la operación lanzada por esa consola, evitando carreras con un estado `busy=false` anterior. El plate solve tiene un presupuesto total configurable (`platesolving.total_timeout_s`, 120 s por defecto); si un `await platesolving SEG` más corto vence, solicita cancelación cooperativa del solver. `stop` (también `estop`) ejecuta una parada atómica: detiene la montura, cancela plate solving, GoTo y tracking, e invalida la posición sincronizada si había movimiento en curso. Por seguridad, un modelo con menos de tres muestras ajustadas no permite GoTo mayores de 3°, y ningún GoTo puede superar 10° sin cambiar explícitamente la configuración.

La cinemática nominal de la montura es fija: NEMA 17 de 200 pasos/vuelta, microstepping 1/64 y reductor cicloidal 45:1 (`1600` microsteps por grado de salida). El fit no reescribe esos valores. Aprende por separado una corrección global acotada, el roll de cámara, backlash por sentido y una componente periódica acotada de 45 lóbulos (período de 12 800 microsteps, equivalente a 8° de salida). La aceleración y frenado por curva S se ejecutan pulso a pulso en el firmware.

También hay una sesión smoke lista para ejecutar:

```bash
python app.py --cli --demo --connect --script scripts/debug_demo.cli
```

Para diagnósticos de bajo nivel, `action TIPO '{...}'` permite inyectar cualquier `ActionType` con un payload JSON. En uso normal conviene preferir los comandos de alto nivel porque validan sus argumentos.

## Requisitos

- Python con las dependencias de `requirements.txt`.
- SDK de Player One Camera disponible en la plataforma (binarios `.dll/.so/.dylib`).
- ESP32 con firmware de `mount_firmware/mount_firmware.ino` cargado.
- En macOS, `blueutil` (`brew install blueutil`) para que la app pueda olvidar, volver a emparejar y conectar automáticamente la montura Bluetooth antes de abrir el puerto SPP.

## Tests

Los tests se ejecutan con el entorno local activado:

```bash
source /Users/josue/myenv/bin/activate
python3 -m pytest -q
```

---
