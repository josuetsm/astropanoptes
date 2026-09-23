# focuser.py
"""Enfocador motorizado: metrica de foco y busqueda automatica.

El enfocador es el tercer motor del CNC shield, acoplado directo a la ruedita
de foco del telescopio. No tiene encoder ni final de carrera, asi que todo aqui
es relativo: se cuentan pasos desde donde estaba al arrancar la sesion.

La busqueda automatica es una curva en V en dos etapas. Un barrido grueso
localiza el minimo de la V, y uno fino lo afina. Ambos recorren siempre en la
misma direccion y la posicion final se aproxima tambien desde esa direccion,
para que el juego mecanico del acople no se cuele en el resultado.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ap_types import Frame
from logging_utils import log_error, log_info
from mount_arduino import ArduinoMount, normalize_move_profile
from protocols import StatePublisherProtocol
from workers import BaseWorker


def cfa_plane(image: np.ndarray) -> np.ndarray:
    """Un solo plano del mosaico Bayer, en float32.

    Medir nitidez sobre el RAW completo mide el patron Bayer, no el foco: en un
    RGGB los pixeles vecinos son de colores distintos y el gradiente entre ellos
    domina cualquier gradiente real de la imagen. Submuestrear un unico plano
    (aqui el verde de la fila par) elimina ese damero por construccion, a costa
    de la mitad de resolucion, que sobra para una curva de foco.
    """
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = arr.mean(axis=2)
    if arr.ndim != 2:
        raise ValueError(f"se esperaba una imagen 2D, llego {arr.shape}")
    return np.asarray(arr[0::2, 1::2], dtype=np.float32)


def focus_metric(image: np.ndarray) -> float:
    """Nitidez de la imagen: mas alto es mejor foco.

    Es energia de gradiente normalizada por el flujo al cuadrado sobre los
    pixeles con senal. Esa normalizacion la hace invariante a la ganancia y a la
    exposicion (si la imagen se multiplica por k, numerador y denominador se
    multiplican por k^2), de modo que la curva sigue siendo comparable aunque se
    toque la camara a mitad del barrido. Al desenfocar, el flujo total se
    conserva pero se reparte en mas pixeles, los gradientes caen y la metrica
    baja: por eso tiene un maximo en el foco.

    Funciona igual con estrellas y con un planeta, que era el requisito: en un
    caso el gradiente lo aportan los bordes del disco y en el otro los flancos
    de las PSF.

    El fondo se estima por cajas con SEP, el mismo modelo que usa la deteccion
    de fuentes. Bajo la contaminacion luminica de Santiago el cielo no es plano
    -- el gradiente vale mas que el propio ruido de lectura -- y una mediana
    global lo confunde con ruido: el umbral se infla al doble y en un campo
    pobre no queda ni un pixel de senal que medir.
    """
    import cv2
    import sep

    plane = np.ascontiguousarray(cfa_plane(image))
    if plane.size < 4096:
        return 0.0

    try:
        bkg = sep.Background(plane, bw=32, bh=32)
        signal = plane - bkg.back()
        noise = float(bkg.globalrms)
    except Exception:
        background = float(np.median(plane))
        signal = plane - background
        mad = float(np.median(np.abs(signal)))
        noise = 1.4826 * mad if mad > 0.0 else float(np.std(signal))
    if not np.isfinite(noise) or noise <= 0.0:
        return 0.0

    mask = signal > (5.0 * noise)
    if int(mask.sum()) < 16:
        return 0.0
    # Dilatar incluye los flancos de cada fuente, que es donde vive el gradiente
    # que nos interesa; sin esto la mascara se queda solo con los nucleos.
    mask = cv2.dilate(mask.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0

    gx = cv2.Sobel(signal, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(signal, cv2.CV_32F, 0, 1, ksize=3)
    energy = gx * gx + gy * gy

    # El ruido tambien tiene gradiente, y la mascara crece al desenfocar: sin
    # descontarlo, parte de lo que se mide es cuanta area se enmascaro y no
    # cuan nitida es la senal. El nivel se estima fuera de la mascara, donde
    # solo hay cielo.
    outside = ~mask
    n_mask = float(mask.sum())
    noise_energy = float(np.mean(energy[outside])) if bool(outside.any()) else 0.0
    gradient_energy = float(np.sum(energy[mask]) - noise_energy * n_mask)
    if gradient_energy <= 0.0:
        return 0.0

    # Normalizar por la suma de cuadrados, no por el cuadrado de la suma.
    #
    # Con N fuentes iguales, el numerador crece como N. Elevar la suma de flujos
    # al cuadrado hace que el denominador crezca como N^2, y la metrica acaba
    # valiendo 1/N: basta que una estrella salga del campo a mitad del barrido
    # para que la nitidez "suba" sin que el foco haya cambiado. Sumando
    # cuadrados, denominador y numerador crecen igual y N se cancela, de modo
    # que lo que queda es la media -- ponderada por brillo -- de la nitidez de
    # cada fuente. Para una gaussiana da exactamente 1/sigma^2.
    power = float(np.sum(signal[mask] ** 2) - (noise * noise) * n_mask)
    if power <= 0.0:
        return 0.0
    return float(1.0e3 * gradient_energy / power)


def parabolic_peak(
    positions: List[int],
    metrics: List[float],
) -> Optional[float]:
    """Vertice de la parabola por los tres puntos alrededor del mejor.

    Devuelve None si el mejor cae en un extremo (todavia no se cerro la V) o si
    los tres puntos no forman un maximo.
    """
    if len(positions) < 3:
        return None
    best = int(np.argmax(metrics))
    if best == 0 or best == len(positions) - 1:
        return None
    x0, x1, x2 = (float(positions[best - 1]), float(positions[best]), float(positions[best + 1]))
    y0, y1, y2 = (float(metrics[best - 1]), float(metrics[best]), float(metrics[best + 1]))
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0.0:
        return None
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom
    if a >= 0.0:
        return None
    vertex = -b / (2.0 * a)
    lo, hi = min(x0, x2), max(x0, x2)
    if not (lo <= vertex <= hi):
        return None
    return float(vertex)


class FocusPresets:
    """Posiciones de foco con nombre, persistidas entre sesiones.

    Guardar un numero de pasos solo tiene sentido si el origen es el mismo cada
    vez, asi que cada preset recuerda si al guardarlo habia homing. Uno guardado
    sin homing describe un origen arbitrario -- el punto donde se abrio la app
    aquel dia -- y aplicarlo despues moveria el enfocador a cualquier sitio.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()

    def load(self) -> Dict[str, Dict[str, Any]]:
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        out: Dict[str, Dict[str, Any]] = {}
        for name, entry in data.items():
            if isinstance(entry, dict) and "position" in entry:
                try:
                    out[str(name)] = {
                        "position": int(entry["position"]),
                        "homed": bool(entry.get("homed", False)),
                        "session": str(entry.get("session", "")),
                        "saved_at": str(entry.get("saved_at", "")),
                        "note": str(entry.get("note", "")),
                        "history": [
                            int(h) for h in entry.get("history", [])
                            if isinstance(h, (int, float))
                        ],
                    }
                except (TypeError, ValueError):
                    continue
        return out

    def _write(self, data: Dict[str, Dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Escritura atomica: un corte a medias dejaria el archivo ilegible y se
        # perderian todos los presets, no solo el que se estaba guardando.
        handle = tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(self.path.parent),
            delete=False,
        )
        try:
            with handle:
                json.dump(data, handle, indent=2, sort_keys=True, ensure_ascii=False)
            os.replace(handle.name, self.path)
        except BaseException:
            Path(handle.name).unlink(missing_ok=True)
            raise

    def save(
        self,
        name: str,
        position: int,
        *,
        homed: bool,
        session: str = "",
        note: str = "",
    ) -> None:
        with self._lock:
            data = self.load()
            data[str(name)] = {
                "position": int(position),
                "homed": bool(homed),
                # Sin homing, el origen es el punto donde se abrio la app ese
                # dia. Anotar la sesion permite distinguir "sirve todavia" de
                # "describe un cero que ya no existe".
                "session": str(session),
                # El historial sobrevive a un re-guardado: es la medida de
                # cuanto se mueve el foco de una noche a otra, y perderla cada
                # vez que se corrige el nominal dejaria al buscador sin prior.
                "history": list(data.get(str(name), {}).get("history", [])),
                "saved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "note": str(note),
            }
            self._write(data)

    def delete(self, name: str) -> bool:
        with self._lock:
            data = self.load()
            if str(name) not in data:
                return False
            del data[str(name)]
            self._write(data)
            return True

    def get(self, name: str) -> Optional[Dict[str, Any]]:
        return self.load().get(str(name))

    def record_result(self, name: str, position: int, *, history_max: int = 20) -> None:
        """Anota donde quedo realmente el foco esta vez.

        El foco cambia de una noche a otra -- dilatacion del tubo, temperatura --
        asi que el nominal guardado es un punto de partida, no la respuesta. Lo
        que dice cuanto hay que buscar alrededor es la dispersion medida en este
        equipo, no una constante inventada.
        """
        with self._lock:
            data = self.load()
            entry = data.get(str(name))
            if entry is None:
                return
            history = [int(h) for h in entry.get("history", [])]
            history.append(int(position))
            entry["history"] = history[-int(max(1, history_max)):]
            data[str(name)] = entry
            self._write(data)

    @staticmethod
    def prior(entry: Optional[Dict[str, Any]]) -> Tuple[Optional[int], Optional[float]]:
        """Centro y dispersion a partir de un preset.

        El centro es la mediana del historial reciente en cuanto hay muestras:
        asi el prior sigue la deriva estacional sola, sin tocar el nominal que
        guardo el usuario. La dispersion es MAD, que no se deja arrastrar por
        una noche en la que el enfoque salio mal.
        """
        if entry is None:
            return None, None
        history = [int(h) for h in entry.get("history", [])]
        nominal = int(entry.get("position", 0))
        if len(history) < 2:
            return nominal, None
        arr = np.asarray(history, dtype=np.float64)
        center = float(np.median(arr))
        mad = float(np.median(np.abs(arr - center)))
        spread = 1.4826 * mad if mad > 0.0 else float(np.std(arr))
        return int(round(center)), float(spread) if np.isfinite(spread) else None


class FocuserCancelled(RuntimeError):
    """El autofoco fue cancelado por el usuario."""


class FocuserWorker(BaseWorker):
    """Movimientos del enfocador y busqueda automatica, fuera del hilo de control.

    Dependencias inyectadas:
      - get_mount(): ArduinoMount o None
      - get_cfg(): FocuserConfig
      - get_frame(): Frame mas reciente (o None)
      - publish_state(patch)
    """

    def __init__(
        self,
        *,
        get_mount: Callable[[], Optional[ArduinoMount]],
        get_cfg: Callable[[], Any],
        get_frame: Callable[[], Optional[Frame]],
        publish_state: StatePublisherProtocol,
        operation_finished: Optional[Callable[[], None]] = None,
        out_log: Any = None,
    ) -> None:
        super().__init__(name="FocuserWorker")
        self._get_mount = get_mount
        self._get_cfg = get_cfg
        self._get_frame = get_frame
        self._publish_state = publish_state
        self._operation_finished = operation_finished
        self._out_log = out_log

        self._cancel_af = threading.Event()
        self._pos_lock = threading.Lock()
        self._position = 0
        self._last_direction = 0
        self._homed = False
        self._session_id = uuid.uuid4().hex
        self._presets = FocusPresets(
            getattr(get_cfg(), "presets_path", "calibration_frames/focus_presets.json")
        )

    # -------------------------
    # Posicion relativa
    # -------------------------
    @property
    def position(self) -> int:
        with self._pos_lock:
            return int(self._position)

    @property
    def homed(self) -> bool:
        with self._pos_lock:
            return bool(self._homed)

    @property
    def presets(self) -> FocusPresets:
        return self._presets

    @property
    def session_id(self) -> str:
        return self._session_id

    def zero(self) -> None:
        """Declara la posicion actual como el cero de la sesion.

        No es un homing: el cero queda donde este el enfocador ahora, que es un
        sitio distinto cada dia. Sirve dentro de la sesion, no entre sesiones.
        """
        with self._pos_lock:
            self._position = 0
            self._homed = False
        self._publish(
            {"position": 0, "homed": False, "best_position": None, "best_metric": 0.0}
        )

    def cancel(self) -> None:
        self._cancel_af.set()

    # -------------------------
    # API
    # -------------------------
    def request(self, *, kind: str, **payload: Any) -> None:
        if str(kind) == "autofocus":
            self._cancel_af.clear()
        super().request(kind=str(kind), **payload)

    def _publish(self, patch: Dict[str, Any]) -> None:
        try:
            self._publish_state({"focuser": dict(patch)})
        except Exception as exc:  # pragma: no cover - frontera defensiva
            log_error(
                self._out_log,
                "Focuser: state publish failed",
                exc,
                throttle_s=5.0,
                throttle_key="focuser_publish",
            )

    def _handle_request(self, request: Dict[str, Any]) -> None:
        kind = str(request.get("kind", "move"))
        try:
            if kind == "move":
                self._do_move(
                    direction=int(request.get("direction", 1)),
                    steps=int(request.get("steps", 0)),
                )
            elif kind == "goto":
                self._do_goto(int(request.get("position", 0)))
            elif kind == "home":
                self._do_home()
            elif kind == "preset":
                self._do_preset(str(request.get("name", "")))
            elif kind == "autofocus":
                self._do_autofocus(dict(request.get("params") or {}))
            else:
                raise ValueError(f"peticion de enfocador desconocida: {kind!r}")
        except FocuserCancelled:
            self._publish({"moving": False, "autofocus": "cancelled", "autofocus_stage": ""})
            log_info(self._out_log, "Focuser: autofoco cancelado")
        except Exception as exc:
            self._publish(
                {
                    "moving": False,
                    "autofocus": "failed" if kind == "autofocus" else "idle",
                    "autofocus_stage": "",
                    "last_error": str(exc),
                }
            )
            log_error(self._out_log, f"Focuser: {kind} failed", exc)
        finally:
            if self._operation_finished is not None:
                self._operation_finished()

    # -------------------------
    # Movimiento
    # -------------------------
    def _require_mount(self) -> ArduinoMount:
        mount = self._get_mount()
        if mount is None or not mount.is_connected():
            raise RuntimeError("la montura no esta conectada")
        if not mount.supports_focuser():
            raise RuntimeError(
                "el firmware cargado no tiene el tercer eje; flashea "
                "mount_firmware con soporte de enfocador"
            )
        return mount

    def _travel_bounds(self) -> Tuple[Optional[int], Optional[int]]:
        """Limites del recorrido, que dependen de si hay homing.

        Con homing, el cero es el tope retraido y el rango real es 0..recorrido:
        un guardia asimetrico que si protege los dos extremos. Sin homing no se
        sabe donde esta el enfocador dentro de su recorrido, y lo unico honesto
        es limitar simetricamente alrededor del punto de partida.
        """
        cfg = self._get_cfg()
        if self.homed:
            travel = max(0, int(getattr(cfg, "home_travel_steps", 0)))
            if travel <= 0:
                return None, None
            return 0, travel
        limit = max(0, int(getattr(cfg, "max_travel_steps", 0)))
        if limit <= 0:
            return None, None
        return -limit, limit

    def _move_relative(
        self,
        mount: ArduinoMount,
        signed_steps: int,
        *,
        allow_beyond_limit: bool = False,
    ) -> None:
        """Mueve el enfocador y actualiza la posicion relativa.

        El juego del acople se compensa como en la montura: al invertir el
        sentido se manda primero el juego conocido, que no cuenta como
        recorrido util porque no mueve el foco.
        """
        cfg = self._get_cfg()
        if int(signed_steps) == 0:
            return
        direction = 1 if signed_steps > 0 else -1
        steps = abs(int(signed_steps))

        target = self.position + direction * steps
        if not allow_beyond_limit:
            lo, hi = self._travel_bounds()
            if lo is not None and target < lo:
                raise RuntimeError(
                    f"el movimiento saldria del recorrido permitido "
                    f"({target:+d} pasos, minimo {lo:+d})"
                )
            if hi is not None and target > hi:
                raise RuntimeError(
                    f"el movimiento saldria del recorrido permitido "
                    f"({target:+d} pasos, maximo {hi:+d})"
                )

        sign = -1 if bool(getattr(cfg, "invert", False)) else 1
        profile = normalize_move_profile(str(getattr(cfg, "profile", "smooth")))
        delay_us = max(1, int(getattr(cfg, "delay_us", 900)))
        backlash = max(0, int(getattr(cfg, "backlash_steps", 0)))

        self._publish({"moving": True})
        try:
            if backlash > 0 and self._last_direction in (-1, 1) and self._last_direction != direction:
                mount.focus_steps(
                    sign * direction,
                    backlash,
                    delay_us,
                    profile=profile,
                    blocking=True,
                )
            mount.focus_steps(
                sign * direction,
                steps,
                delay_us,
                profile=profile,
                blocking=True,
            )
        finally:
            self._publish({"moving": False})

        self._last_direction = direction
        with self._pos_lock:
            self._position += direction * steps
        self._publish({"position": self.position})

    def _do_move(self, *, direction: int, steps: int) -> None:
        if int(steps) <= 0:
            return
        mount = self._require_mount()
        self._move_relative(mount, (1 if direction >= 0 else -1) * int(steps))

    def _do_goto(self, position: int) -> None:
        mount = self._require_mount()
        self._move_relative(mount, int(position) - self.position)

    def _do_home(self) -> None:
        """Homing aproximado contra la zona de patinaje del pinon.

        El pinon tiene dientes rotos en un extremo: retraido del todo sigue
        girando sin arrastrar nada. Mandar mas pasos de los que caben en el
        recorrido no fuerza el mecanismo, solo patina, y deja el enfocador
        siempre en el mismo sitio fisico. Ese es el origen repetible que hace
        que una posicion guardada signifique lo mismo manana.

        No es un final de carrera: la posicion es buena a la escala del juego
        del acople, no al microstep.
        """
        cfg = self._get_cfg()
        mount = self._require_mount()
        travel = max(1, int(getattr(cfg, "home_travel_steps", 32000)))
        overshoot = max(0, int(getattr(cfg, "home_overshoot_steps", 3000)))
        extend_positive = bool(getattr(cfg, "home_extend_is_positive", True))
        retract = -1 if extend_positive else +1

        total = travel + overshoot
        self._publish({"autofocus_stage": "homing", "last_error": None})
        log_info(
            self._out_log,
            f"Focuser: homing, {total} pasos de retraccion "
            f"({travel} de recorrido + {overshoot} de patinaje)",
        )
        # allow_beyond_limit: el homing es justo la operacion que tiene que
        # pasarse del tope, porque el tope es lo que esta buscando.
        self._move_relative(mount, retract * total, allow_beyond_limit=True)

        with self._pos_lock:
            self._position = 0
            self._homed = True
        # Tras patinar contra el tope, el juego esta tomado en el sentido de
        # retraccion; declararlo evita que el primer movimiento util lo repita.
        self._last_direction = retract
        self._publish(
            {
                "position": 0,
                "homed": True,
                "autofocus_stage": "",
                "best_position": None,
                "best_metric": 0.0,
            }
        )
        log_info(self._out_log, "Focuser: homing terminado, posicion 0 = retraido")

    def _do_preset(self, name: str) -> None:
        entry = self._presets.get(name)
        if entry is None:
            raise RuntimeError(f"no hay preset de foco llamado {name!r}")
        if bool(entry.get("homed", False)) and not self.homed:
            raise RuntimeError(
                f"el preset {name!r} se guardo con homing y esta sesion no lo "
                "tiene: haz 'focus home' primero o la posicion no significa nada"
            )
        if not bool(entry.get("homed", False)):
            if self.homed:
                raise RuntimeError(
                    f"el preset {name!r} se guardo sin homing, respecto a un origen "
                    "arbitrario: vuelve a guardarlo con la sesion homed"
                )
            if str(entry.get("session", "")) != self._session_id:
                raise RuntimeError(
                    f"el preset {name!r} se guardo sin homing en otra sesion: su "
                    "cero era el punto donde se abrio la app aquel dia y ya no "
                    "existe. Haz 'focus home' y guardalo de nuevo"
                )
        log_info(
            self._out_log,
            f"Focuser: preset {name!r} -> {int(entry['position']):+d}",
        )
        self._do_goto(int(entry["position"]))

    # -------------------------
    # Autofoco
    # -------------------------
    def _check_cancel(self) -> None:
        if self._cancel_af.is_set():
            raise FocuserCancelled()

    def _measure(self, *, settle_s: float, frames: int) -> float:
        """Metrica en la posicion actual, mediana de varios frames.

        Solo cuentan frames cuya integracion empezo despues de que el enfocador
        se detuvo. El buffer de la camara casi siempre tiene un frame anterior
        al movimiento, y usarlo mide el foco de la posicion previa: la curva
        sale corrida un paso entero y el maximo cae donde no es. Por eso se
        exige ``t_capture >= t_ref + exposicion``, que vale tanto si el sello de
        tiempo marca el inicio como el final de la toma.

        La mediana entre frames amortigua el seeing, que en una sola toma puede
        superar la diferencia entre dos posiciones vecinas.
        """
        self._check_cancel()
        if settle_s > 0.0:
            deadline = time.perf_counter() + float(settle_s)
            while time.perf_counter() < deadline:
                self._check_cancel()
                time.sleep(min(0.05, max(0.0, deadline - time.perf_counter())))

        t_ref = time.perf_counter()
        wanted = max(1, int(frames))
        values: List[float] = []
        seen: Optional[float] = None
        exposure_s = 0.0
        deadline = t_ref + 10.0
        while len(values) < wanted and time.perf_counter() < deadline:
            self._check_cancel()
            fr = self._get_frame()
            if fr is None:
                time.sleep(0.02)
                continue
            if not values:
                try:
                    exposure_s = max(0.0, float(fr.meta.get("exp_ms", 0.0)) / 1000.0)
                except (TypeError, ValueError):
                    exposure_s = 0.0
                # Cada frame extra cuesta al menos una exposicion mas.
                deadline = t_ref + 10.0 + 3.0 * exposure_s * wanted
            token = float(fr.t_capture)
            if token < (t_ref + exposure_s) or token == seen:
                time.sleep(0.02)
                continue
            seen = token
            values.append(focus_metric(fr.raw))
        if not values:
            raise RuntimeError("no llegaron frames de la camara para medir el foco")
        return float(np.median(values))

    def _sweep(
        self,
        mount: ArduinoMount,
        measured: Dict[int, float],
        *,
        start: int,
        step: int,
        points: int,
        settle_s: float,
        frames: int,
        stage: str,
    ) -> None:
        """Recorre ``points`` posiciones en un solo sentido, midiendo en cada una.

        Siempre se aproxima ``start`` primero y despues se avanza: asi todas las
        medidas del barrido llegan desde el mismo lado y el juego del acople no
        se mezcla con la curva.
        """
        self._move_relative(mount, start - self.position)
        for index in range(int(points)):
            self._check_cancel()
            if index > 0:
                self._move_relative(mount, int(step))
            value = self._measure(settle_s=settle_s, frames=frames)
            measured[self.position] = value
            self._publish(
                {
                    "autofocus_stage": f"{stage} {index + 1}/{int(points)}",
                    "last_metric": float(value),
                    "samples": len(measured),
                }
            )
            log_info(
                self._out_log,
                f"Focuser: {stage} {index + 1}/{int(points)} pos={self.position:+d} "
                f"nitidez={value:.3f}",
            )

    @staticmethod
    def _best_of(measured: Dict[int, float]) -> int:
        return max(measured, key=lambda pos: measured[pos])

    @classmethod
    def _peak_estimate(cls, measured: Dict[int, float]) -> int:
        """Maximo de la curva, interpolado si los tres puntos lo permiten."""
        positions = sorted(measured)
        vertex = parabolic_peak(positions, [measured[pos] for pos in positions])
        if vertex is None:
            return cls._best_of(measured)
        return int(round(vertex))

    def _sweep_until_bracketed(
        self,
        mount: ArduinoMount,
        *,
        origin: int,
        step: int,
        points: int,
        settle_s: float,
        frames: int,
        stage: str,
        max_extensions: int = 3,
    ) -> Dict[int, float]:
        """Barrido grueso que se extiende hasta encerrar el maximo.

        Un maximo en un extremo del barrido no es un maximo: solo dice que el
        foco esta mas alla. Aceptarlo daria por bueno el punto mas lejano que se
        llego a medir, que es exactamente el error que comete un barrido de
        rango fijo cuando el enfocador arranca lejos.
        """
        measured: Dict[int, float] = {}
        self._sweep(
            mount,
            measured,
            start=origin - step * (points // 2),
            step=step,
            points=points,
            settle_s=settle_s,
            frames=frames,
            stage=stage,
        )
        extra = max(2, int(points) // 2)
        for _ in range(int(max_extensions)):
            ordered = sorted(measured)
            best = self._best_of(measured)
            if best not in (ordered[0], ordered[-1]):
                break
            forward = best == ordered[-1]
            start = (
                ordered[-1] + step
                if forward
                else ordered[0] - step * extra
            )
            try:
                self._sweep(
                    mount,
                    measured,
                    start=start,
                    step=step,
                    points=extra,
                    settle_s=settle_s,
                    frames=frames,
                    stage=f"{stage} (extension)",
                )
            except FocuserCancelled:
                raise
            except RuntimeError as exc:
                # Tipicamente el tope de recorrido: se sigue con lo medido en
                # vez de tirar todo el barrido a la basura.
                log_info(self._out_log, f"Focuser: no se pudo extender el barrido ({exc})")
                break
        return measured

    def _nearest_preset(self) -> Optional[str]:
        """Que barlow parece estar puesto, deducido de donde esta el enfocador.

        Solo con homing: sin un origen comun, comparar posiciones de sesiones
        distintas no significa nada.
        """
        if not self.homed:
            return None
        cfg = self._get_cfg()
        tolerance = max(1, int(getattr(cfg, "autofocus_prior_match_steps", 1500)))
        here = self.position
        best_name, best_dist = None, None
        for name, entry in self._presets.load().items():
            if not bool(entry.get("homed", False)):
                continue
            center, _spread = FocusPresets.prior(entry)
            if center is None:
                continue
            dist = abs(int(center) - here)
            if dist <= tolerance and (best_dist is None or dist < best_dist):
                best_name, best_dist = name, dist
        return best_name

    def _search_prior(
        self, params: Dict[str, Any]
    ) -> Tuple[Optional[str], Optional[int], Optional[int]]:
        """(nombre, centro, semiancho) de la busqueda dirigida, o Nones."""
        cfg = self._get_cfg()
        if not bool(params.get("prior_enabled", getattr(cfg, "autofocus_prior_enabled", True))):
            return None, None, None

        name = str(params.get("preset", "") or "").strip() or self._nearest_preset()
        if not name:
            return None, None, None
        entry = self._presets.get(name)
        if entry is None:
            return None, None, None
        if bool(entry.get("homed", False)) != self.homed:
            # Origen distinto: el numero guardado no describe esta sesion.
            return None, None, None

        center, spread = FocusPresets.prior(entry)
        if center is None:
            return None, None, None
        floor = max(1, int(getattr(cfg, "autofocus_prior_window_steps", 800)))
        ceiling = max(floor, int(getattr(cfg, "autofocus_prior_max_window_steps", 6000)))
        if spread is None:
            half = floor
        else:
            k = float(getattr(cfg, "autofocus_prior_window_sigma", 4.0))
            half = int(round(max(float(floor), k * float(spread))))
        return name, int(center), int(min(half, ceiling))

    @staticmethod
    def _is_bracketed(measured: Dict[int, float]) -> bool:
        """El maximo esta encerrado, no pegado a un extremo del barrido."""
        if len(measured) < 3:
            return False
        ordered = sorted(measured)
        best = max(measured, key=lambda pos: measured[pos])
        return best not in (ordered[0], ordered[-1])

    def _do_autofocus(self, params: Dict[str, Any]) -> None:
        cfg = self._get_cfg()
        mount = self._require_mount()

        coarse_step = max(1, int(params.get("coarse_step", getattr(cfg, "autofocus_coarse_step", 400))))
        coarse_points = max(3, int(params.get("coarse_points", getattr(cfg, "autofocus_coarse_points", 9))))
        fine_step = max(1, int(params.get("fine_step", getattr(cfg, "autofocus_fine_step", 100))))
        fine_points = max(3, int(params.get("fine_points", getattr(cfg, "autofocus_fine_points", 7))))
        settle_s = max(0.0, float(params.get("settle_s", getattr(cfg, "autofocus_settle_s", 0.8))))
        frames = max(1, int(params.get("frames", getattr(cfg, "autofocus_frames", 3))))

        origin = self.position
        preset_name, prior_center, prior_half = self._search_prior(params)
        self._publish(
            {
                "autofocus": "running",
                "autofocus_stage": "grueso",
                "samples": 0,
                "best_position": None,
                "best_metric": 0.0,
                "last_error": None,
            }
        )

        coarse: Dict[int, float] = {}
        if prior_center is not None and prior_half is not None:
            # Busqueda dirigida: una ventana alrededor de lo que ya se sabe, con
            # el paso justo para cubrirla en los puntos previstos.
            points = max(3, int(params.get("prior_points", getattr(cfg, "autofocus_prior_points", 7))))
            step = max(fine_step, int(round(2.0 * prior_half / max(1, points - 1))))
            log_info(
                self._out_log,
                f"Focuser: autofoco dirigido por {preset_name!r}, centro "
                f"{prior_center:+d} +-{prior_half} ({points}x{step})",
            )
            coarse = self._sweep_until_bracketed(
                mount,
                origin=prior_center,
                step=step,
                points=points,
                settle_s=settle_s,
                frames=frames,
                stage=f"dirigido:{preset_name}",
                max_extensions=2,
            )
            if not (self._is_bracketed(coarse) and max(coarse.values()) > 0.0):
                # El prior no sirvio: el foco se movio mas de lo que la
                # dispersion historica hacia esperar, o cambio algo del tren
                # optico. Se cae a la busqueda general en vez de devolver el
                # mejor punto de una ventana que no contenia el maximo.
                log_info(
                    self._out_log,
                    f"Focuser: el prior de {preset_name!r} no encerro el maximo; "
                    "pasando a busqueda general",
                )
                coarse = {}

        if not coarse:
            log_info(
                self._out_log,
                f"Focuser: autofoco general desde {origin:+d} "
                f"(grueso {coarse_points}x{coarse_step}, fino {fine_points}x{fine_step})",
            )
            coarse = self._sweep_until_bracketed(
                mount,
                origin=origin,
                step=coarse_step,
                points=coarse_points,
                settle_s=settle_s,
                frames=frames,
                stage="grueso",
            )
        if max(coarse.values()) <= 0.0:
            raise RuntimeError(
                "ninguna posicion dio senal medible; revisa exposicion y ganancia"
            )
        # Centrar el fino en el vertice interpolado del grueso y no en su punto
        # medido mas alto: con pasos gruesos el maximo real cae casi siempre
        # entre dos muestras, y centrar en la muestra deja el optimo pegado al
        # borde del barrido fino.
        coarse_peak = self._peak_estimate(coarse)

        fine = self._sweep_until_bracketed(
            mount,
            origin=coarse_peak,
            step=fine_step,
            points=fine_points,
            settle_s=settle_s,
            frames=frames,
            stage="fino",
            max_extensions=2,
        )

        best_position = self._best_of(fine)
        best_metric = float(fine[best_position])
        best_position = self._peak_estimate(fine)

        # Llegar al optimo desde debajo, igual que en los barridos, para que el
        # juego mecanico ya este tomado cuando se para.
        self._move_relative(mount, (best_position - fine_step) - self.position)
        self._move_relative(mount, fine_step)
        final_metric = self._measure(settle_s=settle_s, frames=frames)

        self._publish(
            {
                "autofocus": "done",
                "autofocus_stage": "",
                "best_position": int(self.position),
                "best_metric": float(max(best_metric, final_metric)),
                "last_metric": float(final_metric),
                "samples": len(coarse) + len(fine),
                "last_error": None,
            }
        )
        drift_text = ""
        if preset_name:
            try:
                self._presets.record_result(
                    preset_name,
                    int(self.position),
                    history_max=int(getattr(cfg, "history_max", 20)),
                )
            except OSError as exc:
                log_error(self._out_log, "Focuser: no se pudo anotar el historial", exc)
            entry = self._presets.get(preset_name)
            if entry is not None:
                drift = int(self.position) - int(entry.get("position", self.position))
                drift_text = f"; {drift:+d} respecto al nominal de {preset_name!r}"

        log_info(
            self._out_log,
            f"Focuser: mejor foco en {self.position:+d} "
            f"(nitidez {final_metric:.3f}; interpolado desde {best_metric:.3f})"
            f"{drift_text}",
        )


__all__ = [
    "FocuserWorker",
    "FocuserCancelled",
    "cfa_plane",
    "focus_metric",
    "parabolic_peak",
]
