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
- `tracking.py`: pipeline de tracking (firma RAW16, control PI, auto-calibración y rate limiter).
- `raw_alignment.py`: alineación rápida directamente sobre Bayer RAW16, compartida por tracking y stacking.
- `stacking.py`: live stacking con alineación RAW16, drizzle opcional, salida mono/RGB y preview JPEG.
- `platesolving.py`: plate solving contra Gaia/SIMBAD, overlay/debug y worker asíncrono.
- `goto.py`: modelo de apuntado, sync, GoTo y rutinas de calibración/autocalibración.
- `gaia_cache.py`: catálogo combinado Gaia DR3 + Hipparcos/Tycho-2, caché HEALPix,
  credenciales y resolución de nombres.
- `simulation.py`: modo demo sin hardware; simula cámara, montura, tracking/GoTo y campos estelares Gaia.
- `mount_arduino.py`: driver de montura vía serial (puerto SPP), comandos `PING/ENABLE/STOP/MOVE/TEST/STATUS`; `MS 64` se conserva como handshake compatible.
- `focuser.py`: métrica de nitidez y búsqueda automática de foco sobre el tercer motor.
- `mount_firmware/mount_firmware.ino`: firmware ESP32 para la montura (lado microcontrolador).
- `logging_utils.py`: logging liviano a stdout o sink global de la UI.

## Módulos actuales (qué hacen)

### 1) Orquestación, UI y terminal
- **`app_runner.py`**
  - Controla el lifecycle de cámara, stream y montura.
  - Ejecuta el loop de control a `control_hz`; el cálculo de tracking y preview corre en workers con política “último frame gana”.
  - Encola stacking sólo cuando cambia la secuencia de cámara y mantiene las transformaciones astronómicas en cadencias independientes.
  - Mantiene el estado global (`AppState`) para la UI.

- **`app.py` + `ui/pyqt6_app.py`**
  - Construye la UI en PyQt6 (botones de conexión, estados, live view).
- Incluye controles manuales de montura (move y stop). El microstepping está cableado de forma fija a 1/64.
  - Refleja métricas de tracking cuando está activo.

- **`terminal_app.py`**
  - Controla el mismo `AppRunner` sin cargar PyQt6 mediante `python app.py --cli`.
  - Ofrece consola interactiva, comandos repetibles con `-c` y archivos de sesión con `--script`.
  - Expone estado/configuración JSON, esperas por campos del estado e inyección avanzada de acciones.
  - `health` incluye percentiles p50/p95/p99 por sección del loop para localizar pausas.
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

- **`preview.py`**
  - Pipeline de preview/visor: stretch por percentiles + gamma (`cfg.camera.gamma`, controlable desde la pestaña Camera o `camera set gamma VALOR`) + JPEG. Solo afecta la imagen mostrada; el RAW guardado no cambia.

### 4) Tracking y control de montura
- **`tracking.py`**
  - Tracking incremental por perfiles RAW16 de precisión y detalle, sin ejecutar SEP.
  - Control PI y rate limiter para generar velocidades de montura (µsteps/s).
  - Soporte de calibración manual + auto-cal (RLS) y bootstrap.

- **`mount_arduino.py`**
  - Conexión serial y protocolo con firmware ESP32 vía Bluetooth Classic SPP.
  - Comandos: `PING`, `ENABLE`, `STOP [eje]`, `MOVE`, `TEST`, `STATUS`, `DEBUG`; `MS 64` existe únicamente como handshake legado sin efecto físico.
  - GoTo y movimientos manuales usan un solo `MOVE` por eje para aprovechar la aceleración/frenado simétricos del firmware cargado; `delay_us` es el retardo mínimo (velocidad máxima).
  - El microstepping no es configurable por software: los tres drivers están cableados permanentemente a 1/64.
  - `ArduinoMount.focus_steps()` mueve el enfocador (eje `C`) sin mandar `STOP`
    global, de modo que enfocar no aborta un slew ni el tracking.
  - `ArduinoMount.test_steps()` usa el comando `TEST`, idéntico a `MOVE` pero sin
    el tope de velocidad. Es para banco con otros motores; en el tren del
    telescopio sólo produciría pérdida de pasos.

- **`focuser.py`**
  - Métrica de nitidez y búsqueda automática del mejor foco.
  - La métrica es energía de gradiente normalizada por la **suma de cuadrados**
    de la señal. Así no cambia si se toca ganancia o exposición a mitad del
    barrido, ni si varía el número de estrellas: con N fuentes, numerador y
    denominador crecen igual y N se cancela. Normalizar por el cuadrado de la
    suma la dejaba valiendo 1/N, y bastaba que una estrella saliera del campo
    para que la nitidez subiera sola. Se mide sobre un solo plano del mosaico
    Bayer (si no, se mediría el patrón de color) y con fondo por cajas de SEP
    (si no, el gradiente de cielo de Santiago se confundiría con ruido y dejaría
    la máscara vacía).
  - La búsqueda es una V en dos etapas, grueso y fino, siempre recorridas en el
    mismo sentido para que el juego del acople no entre en la curva. Si el
    máximo cae en un extremo, el barrido se extiende hasta encerrarlo.
  - Sólo se miden frames cuya integración empezó después del movimiento; el
    frame que ya estaba en el buffer corresponde a la posición anterior.

- **`mount_firmware/mount_firmware.ino`**
  - Firmware actual de ESP32 (nombre BT: `AstroPanoptes-ESP32`), tres ejes:
    `A`=AZ, `B`=ALT, `C`=enfocador.
  - Pines controlados: `EN=21`, `AZ STEP/DIR=33/25`, `ALT STEP/DIR=26/27`,
    `FOCUS STEP/DIR=14/13`. **Verifica los pines del enfocador contra tu cableado
    del CNC shield antes de flashear**: son los únicos que no vienen de una
    sesión real y unos pines equivocados fallan en silencio.
  - `MOVE` sigue limitado a `MOVE_MAX_RATE_STEPS_S` (12000 pasos/s). `TEST` corre
    el mismo perfil de aceleración y frenado sin ese tope, para probar motores
    con otra reducción; no usarlo sobre el telescopio.
  - `STOP` sin argumento detiene los tres ejes; `STOP C` detiene sólo el
    enfocador.

### 5) Stacking, plate solving y GoTo
- **`stacking.py`**
  - Usa el mismo alineador RAW16 de tracking, acumula mosaico mono/RGB, genera preview y guarda `.npy` + `.png`.
  - Soporta drizzle x1/x2/x3 desde la UI.
  - Las grabaciones de prueba se pueden apilar a color sin cargar el `.npy`
    completo en memoria:

```bash
source /Users/josue/myenv/bin/activate
python scripts/stack_raw_recordings.py raw_output/raw_*.npy \
  --scale 2 --output-dir stack_output/raw_drizzle_x2
```

  - El apilado offline registra las muestras Bayer antes de reconstruir RGB,
    evitando ampliar el patrón de la matriz de color. x2 es el valor recomendado;
    x3 se conserva para ópticas y capturas cuyo muestreo medido realmente lo justifique.

  - `scripts/combine_raw_stacks.py` registra stacks de varias grabaciones del
    mismo campo, normaliza su respuesta fotométrica y pondera cada sesión por
    el ruido medido. Conserva PNG de 16 bits, JPEG a resolución completa,
    preview, datos lineales y mapa de cobertura.

- **`platesolving.py` + `gaia_cache.py`**
  - Detecta fuentes con SEP, consulta/carga Gaia, resuelve por tripletas y publica overlays/debug.
  - **Resolver sobre el mosaico apilado** (`platesolving source stack`): en cielos con
    mucha contaminación lumínica es preferible a alargar la exposición. Con
    exposiciones cortas las estrellas quedan puntuales en vez de convertirse en
    trazas por deriva sideral, la señal acumulada saca estrellas más débiles, y
    el mosaico cubre más cielo que un frame suelto — el detector entrega muchas
    más fuentes utilizables, que es justo lo que necesita la búsqueda de tripletas.
    El runner corrige automáticamente tres cosas al usar esta fuente. La escala de
    placa se divide por el factor drizzle. La época del solve es el instante del
    **frame de referencia** del stack (no "ahora"), porque los frames se alinean
    sobre él; usar la hora actual desplazaría el centro por toda la deriva
    acumulada durante el apilado. Y el mosaico se **re-centra sobre el frame de
    referencia**: con la montura parada el lienzo crece alejándose de él, de modo
    que su centro geométrico se aparta del apuntado por la mitad de la deriva
    acumulada, y el solver da por hecho que la imagen está centrada en el objetivo
    pedido. El re-centrado se hace rellenando de forma simétrica, nunca
    recortando, así que no se pierde nada del cielo extra que capturó el mosaico.
    Con `platesolving source live` se vuelve al frame vivo.
  - Usa caché en `~/.cache/gaia_cones` por defecto.
  - Observador por defecto: **Estación Central, Santiago** (`ObserverConfig()`); se puede cambiar desde la pestaña `Observador` o con `platesolving set observer_lat_deg=... observer_lon_deg=... observer_height_m=...`.
  - **Rendimiento de la búsqueda de tripletas.** El costo estaba dominado por
    llamadas repetidas al KD-tree, no por la geometría. Cuatro correcciones,
    medidas con perfilado y verificadas como numéricamente equivalentes:
    los anillos alrededor de `i` no dependen del vecino `j`, así que se calculan
    una vez por `(tripleta, i)` en vez de una vez por par, y los de `j` se
    memorizan; la consulta de vecinos se hace **en lote** por tripleta en lugar
    de una por estrella de catálogo (scikit-learn revalida su entrada en cada
    llamada, y ese coste dominaba el solve completo); los lados del triángulo
    se calculan con productos punto sobre los vectores unitarios que ya existen,
    sin construir objetos `SkyCoord`; y los `source_id` se extraen a un array
    numpy en vez de indexar el `DataFrame` en el bucle más interno.
    Medido sobre un campo de Santiago (9 estrellas, catálogo de 4000):
    **4,16 s → 0,98 s**. El caso que hacía timeout (radio 6°, catálogo de
    12 000) baja a 24 s.
  - Los defaults de `PlatesolvingConfig` (`search_radius_deg=3`, `N_seed=8`, `max_i_scan=5000`, `triplet_max_trials=1500`, `rotation_prior_enable=False`, `total_timeout_s=75`, `download_missing_tiles=False`, `bright_catalog_enabled=False`) son los que resolvieron campos reales de forma repetible en sesiones de observación; `platesolving download` sigue permitiendo traer teselas puntuales aunque `download_missing_tiles` esté en `False` por defecto.
  - **Confianza en cielos contaminados (Santiago).** Bajo contaminación lumínica fuerte y con este FoV angosto (~0.36°×0.20°) un cuadro suele tener solo 3–4 estrellas reales. Por eso el umbral por cuadro es bajo (`min_inliers=3`, el piso estructural: una tripleta semilla aporta 3 coincidencias por construcción, así que un campo de 3 estrellas nunca puede producir un `min_validation_inliers` mayor) y la garantía contra falsos positivos recae en el **consenso multi-cuadro** (`initial_consensus_count=3`): una coincidencia falsa de 3 estrellas contra un catálogo grande es fácil por azar en un cuadro, pero reproducir el *mismo* centro/escala/rotación en cuadros independientes no lo es. Bajar `initial_consensus_count` a 1 desactiva esa red y permite aceptar soluciones falsas.
  - Hipparcos y Tycho-2 completos pueden descargarse directamente desde CDS y
    teselarse localmente sin consultas TAP:

```bash
source /Users/josue/myenv/bin/activate
python scripts/import_bright_catalogs.py --workers 6
```

- **`transmission_error.py`**
  - Aprende el error de transmisión cicloidal **desde el propio tracking**, sin
    plate solves dedicados. El lazo visual ya estima por RLS la respuesta real
    px/µstep decenas de veces por segundo; su variación con la fase del lóbulo
    *es* el error de transmisión. El colector acumula (fase, ganancia) en bins,
    ajusta el primer armónico y lo convierte a los coeficientes que usa
    `GoToModel.periodic_coeff_deg` (la ganancia es la derivada del offset, así
    que la conversión lleva un factor `P·k/2π`).
  - `transmission status` muestra cobertura de fase y estimación actual;
    `transmission apply` la vuelca al modelo de apuntado. Se niega a ajustar sin
    cobertura suficiente en vez de inventar coeficientes.

- **`goto.py`**
  - Mantiene modelo de apuntado, sync desde plate solving, GoTo y calibraciones manual/auto.
  - El presupuesto de acoplamiento entre ejes se corrige por `cos(alt)`: `J` está
    en grados de *azimut* por paso, y cerca del cenit un grado de azimut abarca
    un ángulo mínimo en el cielo, así que un límite fijo sobre la entrada cruda
    rechazaría arriba la misma no-ortogonalidad física que acepta abajo.
  - Un fit rechazado distingue `MODEL_FIT_PHASE_COVERAGE_TOO_SHORT` de
    `MODEL_OUTSIDE_MECHANICAL_LIMITS`. `J` es la escala **media**; con
    desplazamientos mucho menores que un lóbulo se mide la pendiente local del
    error de transmisión (hasta ~20 % con los valores por defecto) y no la escala
    media, de modo que un reductor perfecto puede caer fuera del sobre. El
    mensaje indica cuántos ciclos cubrieron las muestras y cuántos pasos hace
    falta mover.
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

### Controlar la GUI ya abierta desde otra terminal

Al iniciar `python app.py` (la GUI PyQt6), la app también abre un socket de control local en `~/.astropanoptes/control.sock` (permisos `0600`, solo el usuario dueño puede conectarse). Otro proceso —una terminal del usuario, o un agente como Codex/Claude Code— puede conectarse a esa misma sesión con:

```bash
python app.py attach -c "camera connect" -c "tracking start" -c "status --json"
```

Sin `-c`/`--script` abre una consola interactiva (`python app.py attach`) con el mismo lenguaje de comandos que `--cli` (`help`, `status`, `mount move ...`, `goto ...`, etc.), pero ejecutado contra el `AppRunner` que ya está corriendo dentro de la ventana abierta, no contra uno nuevo. Los efectos se ven en vivo en la interfaz gráfica, y el usuario puede seguir usando los controles de la ventana al mismo tiempo: ambos caminos comparten el mismo estado y la misma cola de acciones. Si no hay ninguna GUI abierta, `attach` falla con un mensaje claro en vez de crear una sesión nueva (evita abrir dos conexiones a la cámara/montura a la vez). El socket se puede indicar explícitamente con `--socket RUTA` si se abrió más de una instancia con rutas distintas.

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

### Comprobar los ejes antes de empezar

Un TMC2209 en STEP/DIR no tiene realimentación: el firmware emite los pulsos y
responde `OK` aunque al otro lado no haya motor. Un cable suelto, un driver sin
corriente o un Vref a cero son **indistinguibles de un eje sano** desde el
software, y se llevan la noche entera sin que nada lo delate.

```bash
python scripts/check_axes.py
```

Mueve cada eje una cantidad conocida y mide cuánto se desplazó realmente el
campo, con el mismo alineador que usa el tracking. Tres detalles que hacen que la
medida signifique algo:

- **Descuenta la deriva sideral.** Sin seguimiento el cielo se corre 22 px/s a x1
  y 90 px/s a x5; en los segundos que dura la prueba eso puede superar al propio
  movimiento comandado, y un eje muerto parecería vivo. Se mide primero una
  referencia sin mover nada.
- **Corrige por cos(altitud).** En alt-az el azimut desplaza el campo
  Δaz·cos(alt); cerca del cenit un eje sano movería muy poco cielo y saldría como
  muerto. Se puede indicar la altitud con `--alt-deg`.
- **Dimensiona el movimiento según la escala.** Tiene que quedar solape entre las
  dos imágenes: a x1, 400 pasos corren el campo 1354 px, más que el alto del
  sensor, y no habría nada que correlacionar.

También detecta el error inverso: si el campo se mueve **más** de lo previsto, la
escala óptica configurada no es la que hay montada — típicamente el barlow puesto
no es el seleccionado en la pestaña Observador. Ese error es igual de silencioso:
el plate solving busca a una escala equivocada y falla sin decir por qué.

### Enfocador

El tercer motor del CNC shield va directo a la ruedita de foco del telescopio.
La pestaña **Enfoque** de la GUI y el comando `focus` de la consola controlan lo
mismo:

```text
focus in 300          # acercar 300 microsteps
focus out             # alejar el paso configurado
focus auto            # buscar el mejor foco (necesita la cámara capturando)
focus cancel          # parar el enfocador sin tocar la montura
focus status
focus set autofocus_frames=5 autofocus_fine_step=60
```

La búsqueda automática hace un barrido grueso alrededor de la posición actual,
lo extiende si el máximo cae en un extremo, y afina alrededor del vértice
interpolado. Todos los barridos se recorren en el mismo sentido y la posición
final se aproxima también desde ese lado, para que el juego del acople no entre
en la medida.

#### Homing aproximado y posiciones por barlow

No hay final de carrera, pero el piñón tiene dientes rotos en el extremo
retraído: al pasarse del recorrido sigue girando en banda sin forzar nada.
`focus home` usa eso — retrae `home_travel_steps + home_overshoot_steps` — y deja
un cero mecánico repetible. Es lo que hace que una posición guardada signifique
lo mismo mañana.

```text
focus home            # retrae hasta patinar; posición 0 = retraído
focus save x2         # guarda la posición actual como el foco del barlow x2
focus goto x2         # vuelve a ella
focus presets         # lista lo guardado, con su origen y su fecha
focus forget x2
```

Los presets viven en `calibration_frames/focus_presets.json` (fuera de git) y
cada uno recuerda **con qué origen** se guardó. Un preset guardado con homing no
se aplica en una sesión sin homing, y uno guardado sin homing no se aplica en
otra sesión: en ambos casos el número describiría un cero que ya no existe, y
aplicarlo movería el enfocador a cualquier sitio. La app lo dice en vez de
moverse.

El tope de recorrido también depende de esto. Con homing el rango es
`0..home_travel_steps`, que protege los dos extremos de verdad. Sin homing no se
sabe dónde está el enfocador dentro de su carrera, y sólo queda limitar
simétricamente con `max_travel_steps` alrededor del punto de partida.

`focus zero` **no** es homing: pone el cero donde esté el enfocador ahora, lo
cual sirve dentro de la sesión y nada más.

#### Búsqueda guiada

Con un preset conocido la búsqueda no barre todo el recorrido: usa una ventana
alrededor de lo que ya sabe.

```text
focus goto x2
focus auto          # deduce que el barlow puesto es x2 y busca ahí
focus auto x5       # o se le dice explícitamente
```

El foco cambia de una noche a otra —el tubo se dilata con la temperatura— así
que el nominal guardado es un punto de partida, no la respuesta. Cada autofoco
exitoso se anota en el historial del preset, y de ahí salen dos cosas:

- **El centro** de la ventana es la mediana del historial reciente, no el
  nominal. Así el prior sigue la deriva estacional solo, sin tocar el número que
  guardó el usuario.
- **El ancho** es `k · dispersión medida` en este equipo, con un mínimo. Cuánto
  se mueve el foco de una noche a otra sólo lo sabe el historial; ponerlo como
  constante sería inventarlo.

Si la ventana no acaba encerrando el máximo —el foco se movió más de lo que la
dispersión hacía esperar, o cambió algo del tren óptico— la búsqueda **cae sola
a la general**. Devolver el mejor punto de una ventana que no contiene el máximo
daría un foco malo con aire de éxito.

Sin presets, o sin homing, la búsqueda es la general de siempre.

```text
focus presets
  x1         +12800  (homed)  7 noches, centro +12870, dispersion +-140
  x2         +15200  (homed)  3 noches, centro +15245, dispersion +-90
  x5         +18400  (homed)  sin historial
```

`focus auto` necesita señal medible. Si el campo está demasiado oscuro para el
umbral de detección, se reporta como fallo en vez de aceptar un máximo de ruido.

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
