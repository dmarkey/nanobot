"""ESPHome Voice channel implementation.

Connects to ESPHome voice satellites via the Native API (aioesphomeapi)
and orchestrates the STT -> agent -> TTS pipeline using local models
(faster-whisper for STT, piper-tts for TTS) with silero VAD for
server-side voice activity detection.
"""

from __future__ import annotations

import asyncio
import io
import tempfile
import time
import uuid
import wave
from pathlib import Path
from typing import Any, Literal

from loguru import logger
from pydantic import Field

from nanobot.bus.events import OutboundMessage
from nanobot.bus.queue import MessageBus
from nanobot.channels.base import BaseChannel
from nanobot.config.schema import Base

# ESPHome satellites send/expect 16kHz 16-bit mono PCM
_SAT_RATE = 16000
_SAT_WIDTH = 2
_SAT_CHANNELS = 1

# VAD constants
_VAD_FRAME_SAMPLES = 512  # silero expects multiples of 512 samples
_VAD_FRAME_BYTES = _VAD_FRAME_SAMPLES * _SAT_WIDTH
_SPEECH_THRESHOLD = 0.5  # probability above which we consider speech
_SILENCE_TIMEOUT = 1.5  # seconds of silence after speech to trigger end
_NO_SPEECH_TIMEOUT = 5.0  # seconds to wait for speech before giving up
_MAX_RECORDING = 30.0  # hard cap on recording duration

# Auto-cleanup TTS audio entries after this many seconds
_TTS_STORE_TTL = 30.0


# ---------------------------------------------------------------------------
# Config models
# ---------------------------------------------------------------------------

class ESPHomeSatelliteTarget(Base):
    """Connection target for an ESPHome voice satellite."""

    name: str = "default"
    host: str = "localhost"
    port: int = 6053
    password: str = ""
    encryption_key: str = ""  # Noise PSK for encrypted connections
    use_announcements: bool = False  # End pipeline and play TTS via announcement API (for single-I2S-bus devices)
    speech_threshold: float | None = None  # VAD probability threshold (overrides global)
    silence_timeout_seconds: float | None = None  # Silence after speech to trigger STT (overrides global)


class STTConfig(Base):
    """Speech-to-text configuration."""

    provider: Literal["local", "groq"] = "local"
    model: str = "distil-small.en"
    device: Literal["cpu", "cuda"] = "cpu"
    language: str | None = None


class TTSConfig(Base):
    """Text-to-speech configuration."""

    model: str = "en_US-lessac-medium"
    data_dir: str = "~/.local/share/piper-tts"
    speaker_id: int | None = None


class ESPHomeConfig(Base):
    """ESPHome voice channel configuration."""

    enabled: bool = False
    host: str = "0.0.0.0"  # IP address satellites can reach this server on
    tts_port: int = 18791  # HTTP port for serving TTS audio to satellites
    satellites: list[ESPHomeSatelliteTarget] = Field(default_factory=list)
    stt: STTConfig = Field(default_factory=STTConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    response_timeout: float = 120.0
    allow_from: list[str] = Field(default_factory=lambda: ["*"])
    reconnect_interval: float = 5.0
    silence_timeout_seconds: float = _SILENCE_TIMEOUT
    speech_threshold: float = _SPEECH_THRESHOLD


# ---------------------------------------------------------------------------
# Channel
# ---------------------------------------------------------------------------

class ESPHomeChannel(BaseChannel):
    """ESPHome Voice channel.

    Connects to ESPHome voice satellites, runs local STT (faster-whisper)
    and TTS (piper), and routes transcripts through the nanobot agent.
    Uses silero VAD for server-side voice activity detection.
    """

    name = "esphome"
    display_name = "ESPHome Voice"

    @classmethod
    def default_config(cls) -> dict[str, Any]:
        return ESPHomeConfig().model_dump(by_alias=True)

    def __init__(self, config: Any, bus: MessageBus):
        if isinstance(config, dict):
            config = ESPHomeConfig.model_validate(config)
        super().__init__(config, bus)
        self.config: ESPHomeConfig = config
        self._satellite_tasks: list[asyncio.Task] = []
        self._pending: dict[str, asyncio.Future[str]] = {}
        # Cache late agent responses per satellite
        self._deferred: dict[str, str] = {}
        # Lazy-loaded models (shared across satellites, loaded once)
        self._whisper_model: Any = None
        self._piper_voice: Any = None
        self._vad_model: Any = None
        # TTS audio serving — satellites fetch TTS via URL
        self._tts_audio_store: dict[str, bytes] = {}  # id -> wav bytes
        self._http_runner: Any = None
        self._http_port = config.tts_port
        # Media player key and audio format cache per satellite
        self._media_player_keys: dict[str, int] = {}
        self._satellite_sample_rates: dict[str, int] = {}  # {name: sample_rate}
        # Feedback sounds (16kHz WAV)
        self._thinking_sound: bytes = b""
        self._dismiss_sound: bytes = b""
        _res = Path(__file__).parent.parent / "resources"
        if (_res / "processing.wav").exists():
            self._thinking_sound = (_res / "processing.wav").read_bytes()
        if (_res / "dismiss.wav").exists():
            self._dismiss_sound = (_res / "dismiss.wav").read_bytes()

    # ------------------------------------------------------------------
    # Media player helpers
    # ------------------------------------------------------------------

    async def _get_media_player_key(self, client: Any, sat_name: str) -> int | None:
        """Get the media player entity key, caching it and the audio format per satellite."""
        if sat_name in self._media_player_keys:
            return self._media_player_keys[sat_name]
        try:
            from aioesphomeapi.model import MediaPlayerInfo, MediaPlayerFormatPurpose
            entities, _ = await client.list_entities_services()
            for ent in entities:
                if isinstance(ent, MediaPlayerInfo):
                    self._media_player_keys[sat_name] = ent.key
                    # Cache the announcement sample rate
                    for fmt in ent.supported_formats:
                        if fmt.purpose == MediaPlayerFormatPurpose.ANNOUNCEMENT and fmt.sample_rate:
                            self._satellite_sample_rates[sat_name] = fmt.sample_rate
                            logger.info(
                                "ESPHome: '{}' announcement format: {} {}Hz {}ch",
                                sat_name, fmt.format, fmt.sample_rate, fmt.num_channels,
                            )
                            break
                    return ent.key
        except Exception:
            logger.debug("ESPHome: could not find media player on '{}'", sat_name)
        return None

    async def _play_via_media_player(
        self, client: Any, url: str, wav_data: bytes, sat_name: str = "",
    ) -> None:
        """Play a WAV URL via media_player_command and wait for estimated duration."""
        key = await self._get_media_player_key(client, sat_name)
        if key is None:
            logger.warning("ESPHome: no media player found on '{}'", sat_name)
            return
        logger.debug("ESPHome: playing via media player: {}", url)
        client.media_player_command(key, media_url=url)
        # Estimate duration from WAV data and wait
        try:
            wav_buf = io.BytesIO(wav_data)
            with wave.open(wav_buf, "rb") as wf:
                duration = wf.getnframes() / wf.getframerate()
            await asyncio.sleep(duration + 0.5)  # extra buffer for decode/startup
        except Exception:
            await asyncio.sleep(3.0)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def _start_tts_server(self) -> None:
        """Start a minimal HTTP server to serve TTS audio files to satellites."""
        from aiohttp import web

        async def _handle_tts(request: web.Request) -> web.Response:
            audio_id = request.match_info["audio_id"]
            wav_data = self._tts_audio_store.get(audio_id)
            if wav_data is None:
                return web.Response(status=404)
            # Remove after serving (delayed to handle HEAD+GET from mpv)
            asyncio.get_running_loop().call_later(
                5.0, self._tts_audio_store.pop, audio_id, None
            )
            return web.Response(
                body=wav_data,
                content_type="audio/wav",
                headers={"Content-Length": str(len(wav_data))},
            )

        app = web.Application()
        app.router.add_get("/tts/{audio_id}.wav", _handle_tts)
        self._http_runner = web.AppRunner(app)
        await self._http_runner.setup()
        site = web.TCPSite(self._http_runner, "0.0.0.0", self._http_port)
        await site.start()
        logger.info("ESPHome: TTS audio server listening on port {}", self._http_port)

    async def start(self) -> None:
        try:
            import aioesphomeapi  # noqa: F401
        except ImportError:
            logger.error(
                "ESPHome channel requires 'aioesphomeapi'. "
                "Install with: uv pip install aioesphomeapi"
            )
            return

        if not self.config.satellites:
            logger.warning("ESPHome: no satellites configured")
            return

        # Pre-load models in a thread so we don't block the event loop
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._load_models)
        except Exception:
            logger.exception("ESPHome: failed to load models, channel will not start")
            return

        # Start TTS audio HTTP server
        await self._start_tts_server()

        self._running = True
        logger.info(
            "ESPHome voice channel started with {} satellite(s) "
            "(STT: {} / {}, TTS: piper / {})",
            len(self.config.satellites),
            self.config.stt.provider,
            self.config.stt.model,
            self.config.tts.model,
        )

        for target in self.config.satellites:
            task = asyncio.create_task(
                self._satellite_loop(target), name=f"esphome-{target.name}"
            )
            self._satellite_tasks.append(task)

        while self._running:
            await asyncio.sleep(1)

    async def stop(self) -> None:
        self._running = False
        for fut in self._pending.values():
            if not fut.done():
                fut.cancel()
        self._pending.clear()
        for task in self._satellite_tasks:
            task.cancel()
        for task in self._satellite_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._satellite_tasks.clear()
        if self._http_runner:
            await self._http_runner.cleanup()
            self._http_runner = None
        self._tts_audio_store.clear()
        logger.info("ESPHome channel stopped")

    async def send(self, msg: OutboundMessage) -> None:
        """Resolve the pending future so the pipeline can return the response."""
        if msg.metadata.get("_progress"):
            return
        sat_name = msg.metadata.get("esphome_satellite") or msg.chat_id
        fut = self._pending.get(sat_name)
        if fut and not fut.done():
            fut.set_result(msg.content)
        else:
            # Agent finished after we already timed out — cache for next request.
            logger.info("ESPHome: caching deferred response for '{}'", sat_name)
            self._deferred[sat_name] = msg.content

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Load STT, TTS, and VAD models (called once, in a thread)."""
        # VAD (silero, bundled with faster-whisper)
        from faster_whisper.vad import get_vad_model

        logger.info("Loading silero VAD model...")
        self._vad_model = get_vad_model()
        logger.info("Silero VAD model loaded")

        # STT
        if self.config.stt.provider == "local":
            from faster_whisper import WhisperModel

            logger.info("Loading faster-whisper model '{}'...", self.config.stt.model)
            self._whisper_model = WhisperModel(
                self.config.stt.model,
                device=self.config.stt.device,
                compute_type="int8" if self.config.stt.device == "cpu" else "float16",
            )
            logger.info("faster-whisper model loaded")

        # TTS
        from piper import PiperVoice

        data_dir = Path(self.config.tts.data_dir).expanduser()
        data_dir.mkdir(parents=True, exist_ok=True)
        model_path = data_dir / f"{self.config.tts.model}.onnx"

        if not model_path.exists():
            alt = Path("~/.local/share/wyoming-piper").expanduser() / f"{self.config.tts.model}.onnx"
            if alt.exists():
                model_path = alt
            else:
                self._download_piper_voice(self.config.tts.model, data_dir)

        logger.info("Loading piper voice '{}'...", model_path.name)
        self._piper_voice = PiperVoice.load(str(model_path))
        logger.info("Piper voice loaded (sample_rate={})", self._piper_voice.config.sample_rate)

    def _serve_tts_url(self, client: Any, wav_data: bytes) -> None:
        """Store WAV data and send TTS_END with URL to the satellite."""
        from aioesphomeapi import VoiceAssistantEventType

        audio_id = uuid.uuid4().hex[:12]
        self._tts_audio_store[audio_id] = wav_data
        asyncio.get_running_loop().call_later(
            _TTS_STORE_TTL, self._tts_audio_store.pop, audio_id, None
        )
        tts_url = f"http://{self.config.host}:{self._http_port}/tts/{audio_id}.wav"
        client.send_voice_assistant_event(
            VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END,
            {"url": tts_url},
        )

    async def _stream_tts_audio(self, client: Any, wav_data: bytes) -> None:
        """Stream TTS audio to the satellite via the API, paced for playback.

        Mirrors Home Assistant's approach: send TTS_STREAM_START, stream
        16kHz 16-bit mono PCM in small chunks with sleeps to match playback
        rate, then send TTS_STREAM_END.  This lets the firmware release the
        mic/I2S bus and play audio without needing a separate HTTP fetch.
        """
        from aioesphomeapi import VoiceAssistantEventType

        client.send_voice_assistant_event(
            VoiceAssistantEventType.VOICE_ASSISTANT_TTS_STREAM_START, {},
        )

        try:
            # Parse WAV to get raw PCM frames
            wav_buf = io.BytesIO(wav_data)
            with wave.open(wav_buf, "rb") as wf:
                sample_rate = wf.getframerate()
                sample_width = wf.getsampwidth()
                n_channels = wf.getnchannels()
                samples_per_chunk = 512

                logger.debug(
                    "ESPHome: streaming {} audio samples ({}Hz {}bit {}ch)",
                    wf.getnframes(), sample_rate, sample_width * 8, n_channels,
                )

                while True:
                    chunk = wf.readframes(samples_per_chunk)
                    if not chunk:
                        break
                    client.send_voice_assistant_audio(chunk)
                    # Pace sending at ~90% of real-time to avoid overrunning
                    # the device buffer (matches HA's approach)
                    samples_in_chunk = len(chunk) // (sample_width * n_channels)
                    seconds_in_chunk = samples_in_chunk / sample_rate
                    await asyncio.sleep(seconds_in_chunk * 0.9)
        finally:
            client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_TTS_STREAM_END, {},
            )

    def _make_tts_url(self, wav_data: bytes) -> str:
        """Store WAV data and return the URL to access it."""
        audio_id = uuid.uuid4().hex[:12]
        self._tts_audio_store[audio_id] = wav_data
        asyncio.get_running_loop().call_later(
            _TTS_STORE_TTL, self._tts_audio_store.pop, audio_id, None
        )
        return f"http://{self.config.host}:{self._http_port}/tts/{audio_id}.wav"

    async def _deliver_tts(self, client: Any, wav_data: bytes, use_announcements: bool, sat_name: str = "") -> None:
        """Deliver TTS audio to the satellite using the appropriate method.

        URL mode (default): send TTS_END with URL — the satellite fetches and
        plays inline.  Works for linux-voice-assistant and devices that can
        play while the mic is active.

        Announcement mode: end the voice pipeline first (RUN_END), then play
        via media_player_command.  The firmware's on_announcement handler stops
        the mic, plays the audio, and on_idle restarts wake word.
        """
        tts_url = self._make_tts_url(wav_data)

        if use_announcements:
            from aioesphomeapi import VoiceAssistantEventType

            # End the pipeline so firmware releases the I2S bus / mic
            # (may already be ended by thinking sound — duplicates are harmless)
            client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END, {},
            )
            client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, None,
            )
            # Wait briefly then play — if thinking sound is still playing,
            # on_announcement will handle mic stop; if it finished, on_idle
            # restarted mic but on_announcement from TTS will stop it again
            await asyncio.sleep(0.3)
            await self._play_via_media_player(client, tts_url, wav_data, sat_name)
        else:
            self._serve_tts_url(client, wav_data)

    @staticmethod
    def _download_piper_voice(model_name: str, dest_dir: Path) -> None:
        """Download a piper voice model from HuggingFace."""
        import urllib.request

        # Model name format: en_GB-cori-medium -> en/en_GB/cori/medium/
        parts = model_name.split("-")
        if len(parts) != 3:
            raise ValueError(
                f"Cannot auto-download piper model '{model_name}': "
                f"expected format 'lang_COUNTRY-name-quality' (e.g. en_GB-cori-medium)"
            )
        lang_country, voice_name, quality = parts
        lang = lang_country.split("_")[0]
        base = (
            f"https://huggingface.co/rhasspy/piper-voices/resolve/main"
            f"/{lang}/{lang_country}/{voice_name}/{quality}/{model_name}"
        )

        for ext in (".onnx", ".onnx.json"):
            url = f"{base}{ext}"
            dest = dest_dir / f"{model_name}{ext}"
            logger.info("Downloading piper model: {} -> {}", url, dest)
            urllib.request.urlretrieve(url, dest)

        logger.info("Piper model '{}' downloaded to {}", model_name, dest_dir)

    # ------------------------------------------------------------------
    # Satellite connection loop
    # ------------------------------------------------------------------

    async def _satellite_loop(self, target: ESPHomeSatelliteTarget) -> None:
        """Maintain a persistent connection to a single ESPHome satellite."""
        from aioesphomeapi import APIClient, VoiceAssistantEventType
        from aioesphomeapi.core import APIConnectionError

        while self._running:
            client: APIClient | None = None
            pipeline_task: asyncio.Task | None = None
            vad_timeout_task: asyncio.Task | None = None
            try:
                logger.info(
                    "ESPHome: connecting to '{}' at {}:{}",
                    target.name, target.host, target.port,
                )
                client = APIClient(
                    address=target.host,
                    port=target.port,
                    password=target.password or "",
                    client_info="nanobot",
                    noise_psk=target.encryption_key or None,
                )

                disconnect_event = asyncio.Event()

                async def _on_disconnect(expected: bool) -> None:
                    disconnect_event.set()

                await client.connect(on_stop=_on_disconnect)
                logger.info("ESPHome: connected to '{}'", target.name)

                # Cache media player key for this satellite
                key = await self._get_media_player_key(client, target.name)
                if key is not None:
                    logger.info("ESPHome: media player key {} cached for '{}'", key, target.name)
                else:
                    logger.info("ESPHome: no media player found on '{}'", target.name)

                # Per-satellite state
                audio_buffer = bytearray()
                vad_buffer = bytearray()
                pipeline_active = False
                speech_detected = False
                pipeline_start_time = 0.0
                last_speech_time = 0.0
                use_announcements = False  # True when TTS should use announcement API

                async def _vad_silence_monitor() -> None:
                    """Monitor for silence after speech, or no speech / max recording."""
                    nonlocal pipeline_active
                    while pipeline_active:
                        await asyncio.sleep(0.1)
                        if not pipeline_active:
                            return
                        elapsed = time.monotonic() - pipeline_start_time

                        # Hard cap on total recording time
                        if elapsed > _MAX_RECORDING:
                            logger.info(
                                "ESPHome: max recording time reached on '{}' ({:.0f}s)",
                                target.name, _MAX_RECORDING,
                            )
                            await handle_stop(False)
                            return

                        # No speech detected within timeout — give up
                        if not speech_detected and elapsed > _NO_SPEECH_TIMEOUT:
                            logger.info(
                                "ESPHome: no speech detected on '{}' after {:.0f}s, ending",
                                target.name, _NO_SPEECH_TIMEOUT,
                            )
                            await handle_stop(True)
                            return

                        # Silence after speech — user finished talking
                        _sat_silence = target.silence_timeout_seconds if target.silence_timeout_seconds is not None else self.config.silence_timeout_seconds
                        if (
                            speech_detected
                            and last_speech_time > 0
                            and (time.monotonic() - last_speech_time)
                            > _sat_silence
                        ):
                            logger.info(
                                "ESPHome: VAD silence timeout on '{}' "
                                "({:.1f}s), ending audio",
                                target.name,
                                _sat_silence,
                            )
                            client.send_voice_assistant_event(
                                VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_END,
                                None,
                            )
                            await handle_stop(False)
                            return

                async def handle_start(
                    conversation_id: str,
                    flags: int,
                    audio_settings: Any,
                    wake_word_phrase: str | None,
                ) -> int:
                    nonlocal pipeline_active, speech_detected, last_speech_time
                    nonlocal vad_timeout_task, pipeline_start_time, use_announcements
                    from aioesphomeapi import VoiceAssistantFeature
                    use_announcements = target.use_announcements or bool(flags & VoiceAssistantFeature.SPEAKER)
                    audio_buffer.clear()
                    vad_buffer.clear()
                    pipeline_active = True
                    speech_detected = False
                    last_speech_time = 0.0
                    pipeline_start_time = time.monotonic()
                    logger.info(
                        "ESPHome: pipeline started on '{}' (wake: {}, flags={}, continued={}, api_audio={})",
                        target.name, wake_word_phrase or "none", flags,
                        wake_word_phrase is None, use_announcements,
                    )
                    client.send_voice_assistant_event(
                        VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START, None
                    )
                    # Start the monitor immediately so it can detect
                    # no-speech and max-recording timeouts.
                    vad_timeout_task = asyncio.create_task(_vad_silence_monitor())
                    return 0  # API audio mode

                async def handle_stop(abort: bool) -> None:
                    nonlocal pipeline_active, pipeline_task, vad_timeout_task
                    nonlocal speech_detected
                    if not pipeline_active:
                        return
                    pipeline_active = False
                    speech_detected = False
                    if vad_timeout_task and not vad_timeout_task.done():
                        vad_timeout_task.cancel()
                        vad_timeout_task = None
                    if abort:
                        logger.info("ESPHome: pipeline aborted on '{}'", target.name)
                        if pipeline_task and not pipeline_task.done():
                            pipeline_task.cancel()

                        # Play dismissal sound
                        if self._dismiss_sound:
                            dismiss_url = self._make_tts_url(self._dismiss_sound)
                            key = await self._get_media_player_key(client, target.name)
                            if key is not None:
                                # Hardware satellite — end pipeline first, wait for I2S bus release
                                client.send_voice_assistant_event(
                                    VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, None
                                )
                                await asyncio.sleep(2.0)
                                client.media_player_command(key, media_url=dismiss_url)
                            else:
                                # No media player (e.g. LVA) — can't play dismiss sound
                                client.send_voice_assistant_event(
                                    VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, None
                                )
                        else:
                            client.send_voice_assistant_event(
                                VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, None
                            )
                        return
                    audio = bytes(audio_buffer)
                    audio_buffer.clear()
                    vad_buffer.clear()
                    logger.info(
                        "ESPHome: received {:.1f}s of audio from '{}'",
                        len(audio) / (_SAT_RATE * _SAT_WIDTH), target.name,
                    )
                    pipeline_task = asyncio.create_task(
                        self._run_pipeline(target, client, audio, use_announcements)
                    )

                async def handle_audio(data: bytes) -> None:
                    nonlocal speech_detected, last_speech_time, vad_timeout_task
                    if not pipeline_active:
                        return
                    audio_buffer.extend(data)
                    vad_buffer.extend(data)

                    # Run VAD on complete frames (skip first 1s to avoid TTS echo on continued conversations)
                    echo_guard = (time.monotonic() - pipeline_start_time) < 1.0
                    while len(vad_buffer) >= _VAD_FRAME_BYTES:
                        frame = bytes(vad_buffer[:_VAD_FRAME_BYTES])
                        del vad_buffer[:_VAD_FRAME_BYTES]

                        if echo_guard:
                            continue

                        prob = self._run_vad(frame)
                        _sat_threshold = target.speech_threshold if target.speech_threshold is not None else self.config.speech_threshold
                        if prob >= _sat_threshold:
                            if not speech_detected:
                                speech_detected = True
                                logger.info(
                                    "ESPHome: speech detected on '{}' (prob={:.2f})",
                                    target.name, prob,
                                )
                                client.send_voice_assistant_event(
                                    VoiceAssistantEventType.VOICE_ASSISTANT_STT_VAD_START,
                                    None,
                                )
                            last_speech_time = time.monotonic()

                client.subscribe_voice_assistant(
                    handle_start=handle_start,
                    handle_stop=handle_stop,
                    handle_audio=handle_audio,
                )
                logger.info("ESPHome: subscribed to voice assistant on '{}'", target.name)

                # Stay alive until disconnected or stopped
                while self._running and not disconnect_event.is_set():
                    await asyncio.sleep(1)

            except asyncio.CancelledError:
                raise
            except (OSError, ConnectionError, APIConnectionError) as exc:
                logger.warning(
                    "ESPHome: '{}' connection failed: {}", target.name, exc
                )
            except Exception:
                logger.exception(
                    "ESPHome: error in satellite loop for '{}'", target.name
                )
            finally:
                # Clean up in-flight tasks and stale pending futures
                if vad_timeout_task and not vad_timeout_task.done():
                    vad_timeout_task.cancel()
                if pipeline_task and not pipeline_task.done():
                    pipeline_task.cancel()
                old_fut = self._pending.pop(target.name, None)
                if old_fut and not old_fut.done():
                    old_fut.cancel()
                if client:
                    try:
                        await client.disconnect()
                    except Exception:
                        pass

            if self._running:
                logger.info(
                    "ESPHome: reconnecting to '{}' in {}s",
                    target.name, self.config.reconnect_interval,
                )
                await asyncio.sleep(self.config.reconnect_interval)

    def _run_vad(self, frame_pcm: bytes) -> float:
        """Run silero VAD on a single 512-sample frame. Returns speech probability."""
        import numpy as np

        samples = np.frombuffer(frame_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        probs = self._vad_model(samples, num_samples=_VAD_FRAME_SAMPLES)
        return float(probs.flat[0])

    # ------------------------------------------------------------------
    # Voice pipeline: STT -> agent -> TTS -> satellite
    # ------------------------------------------------------------------

    async def _run_pipeline(
        self,
        target: ESPHomeSatelliteTarget,
        client: Any,
        audio: bytes,
        use_announcements: bool = False,
    ) -> None:
        """Run the full voice pipeline for one utterance."""
        from aioesphomeapi import VoiceAssistantEventType

        # Get the satellite's preferred sample rate (0 = use native TTS rate)
        tts_rate = self._satellite_sample_rates.get(target.name, 0)

        try:
            # 1. STT
            client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_START, None
            )
            transcript = await self._do_stt(audio)

            if not transcript.strip():
                logger.debug("ESPHome: empty transcript from '{}', ignoring", target.name)
                client.send_voice_assistant_event(
                    VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, None
                )
                return

            client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_STT_END,
                {"text": transcript},
            )
            logger.info("ESPHome: '{}' said: {}", target.name, transcript)

            # Map voice commands to slash commands
            normalised = transcript.strip().lower().rstrip(".")
            voice_command = None
            if normalised in ("new conversation", "new session", "start over", "reset"):
                voice_command = "/new"
            elif normalised in ("stop", "cancel", "nevermind", "never mind"):
                voice_command = "/stop"

            if voice_command:
                await self._handle_message(
                    sender_id=target.name,
                    chat_id=target.name,
                    content=voice_command,
                    metadata={"esphome_satellite": target.name},
                )
                confirmation = "Done." if voice_command == "/stop" else "New conversation started."
                wav_data = await self._tts_to_wav(confirmation, tts_rate)
                if wav_data:
                    await self._deliver_tts(client, wav_data, use_announcements, target.name)
                if not use_announcements:
                    client.send_voice_assistant_event(
                        VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, None
                    )
                return

            # 2. Agent
            client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_START, None
            )

            # Play thinking sound after STT (mic is released at this point)
            if self._thinking_sound:
                thinking_url = self._make_tts_url(self._thinking_sound)
                key = await self._get_media_player_key(client, target.name)
                if key is not None:
                    client.media_player_command(key, media_url=thinking_url)

            # Check for a deferred response from a previous timed-out request.
            deferred = self._deferred.pop(target.name, None)
            if deferred:
                logger.info("ESPHome: delivering deferred response to '{}'", target.name)
                response_text = deferred
            else:
                loop = asyncio.get_running_loop()
                fut: asyncio.Future[str] = loop.create_future()
                self._pending[target.name] = fut

                try:
                    await self._handle_message(
                        sender_id=target.name,
                        chat_id=target.name,
                        content=transcript,
                        metadata={"esphome_satellite": target.name},
                    )
                    # Wait up to 30s for the first response, then say "still working"
                    # and keep waiting up to the full timeout.
                    _INTERIM_TIMEOUT = 30.0
                    try:
                        response_text = await asyncio.wait_for(
                            fut, timeout=_INTERIM_TIMEOUT
                        )
                    except asyncio.TimeoutError:
                        # Say "still working" — end this pipeline cleanly,
                        # then wait for the real answer and deliver it as a new pipeline.
                        logger.info("ESPHome: interim timeout on '{}', sending progress", target.name)
                        interim_text = "I'm still working on that. Just a moment."
                        client.send_voice_assistant_event(
                            VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_END,
                            {"conversation_id": target.name, "continue_conversation": "0"},
                        )
                        client.send_voice_assistant_event(
                            VoiceAssistantEventType.VOICE_ASSISTANT_TTS_START,
                            {"text": interim_text},
                        )
                        interim_wav = await self._tts_to_wav(interim_text, tts_rate)
                        if interim_wav:
                            await self._deliver_tts(client, interim_wav, use_announcements, target.name)
                        else:
                            client.send_voice_assistant_event(
                                VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, None
                            )

                        # Wait for the real answer
                        remaining = self.config.response_timeout - _INTERIM_TIMEOUT
                        fut2: asyncio.Future[str] = loop.create_future()
                        self._pending[target.name] = fut2
                        try:
                            response_text = await asyncio.wait_for(
                                fut2, timeout=remaining
                            )
                        except asyncio.TimeoutError:
                            logger.warning("ESPHome: agent response timed out for '{}'", target.name)
                            response_text = "Sorry, I took too long to respond. Ask me again in a moment."

                        # Deliver the real answer as a fresh pipeline
                        # so the satellite plays it properly.
                        await asyncio.sleep(1.0)  # wait for interim TTS to finish
                        client.send_voice_assistant_event(
                            VoiceAssistantEventType.VOICE_ASSISTANT_RUN_START, None
                        )
                except asyncio.CancelledError:
                    raise
                finally:
                    self._pending.pop(target.name, None)

            # Always continue conversation — let the no-speech timeout
            # naturally end the session so the user knows when it's over.
            continue_conv = "1"
            client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_INTENT_END,
                {"conversation_id": target.name, "continue_conversation": continue_conv},
            )
            logger.info("ESPHome: responding to '{}': {}", target.name, response_text)

            if not response_text.strip():
                client.send_voice_assistant_event(
                    VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END, None
                )
                client.send_voice_assistant_event(
                    VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, None
                )
                return

            client.send_voice_assistant_event(
                VoiceAssistantEventType.VOICE_ASSISTANT_TTS_START,
                {"text": response_text},
            )

            wav_data = await self._tts_to_wav(response_text, tts_rate)
            if wav_data:
                await self._deliver_tts(client, wav_data, use_announcements, target.name)
                # Don't send RUN_END here — the satellite's _tts_finished()
                # callback (fired when mpv completes playback) handles
                # end-of-pipeline and continue_conversation correctly.
                # Sending RUN_END early resets satellite state while mpv
                # is still playing, causing audio truncation.
            else:
                client.send_voice_assistant_event(
                    VoiceAssistantEventType.VOICE_ASSISTANT_TTS_END, None
                )
                client.send_voice_assistant_event(
                    VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, None
                )

        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("ESPHome: pipeline error for '{}'", target.name)
            try:
                client.send_voice_assistant_event(
                    VoiceAssistantEventType.VOICE_ASSISTANT_ERROR,
                    {"code": "pipeline_error", "message": "Pipeline failed"},
                )
                client.send_voice_assistant_event(
                    VoiceAssistantEventType.VOICE_ASSISTANT_RUN_END, None
                )
            except Exception:
                pass

    # ------------------------------------------------------------------
    # STT
    # ------------------------------------------------------------------

    async def _do_stt(self, audio_pcm: bytes) -> str:
        """Transcribe 16kHz 16-bit mono PCM audio to text."""
        if self.config.stt.provider == "groq":
            return await self._do_stt_groq(audio_pcm)
        return await self._do_stt_local(audio_pcm)

    async def _do_stt_local(self, audio_pcm: bytes) -> str:
        """Transcribe using local faster-whisper model."""
        import numpy as np

        def _transcribe() -> str:
            samples = np.frombuffer(audio_pcm, dtype=np.int16).astype(np.float32) / 32768.0
            segments, _info = self._whisper_model.transcribe(
                samples,
                language=self.config.stt.language,
                beam_size=5,
                vad_filter=False,
            )
            return " ".join(seg.text.strip() for seg in segments)

        return await asyncio.get_running_loop().run_in_executor(None, _transcribe)

    async def _do_stt_groq(self, audio_pcm: bytes) -> str:
        """Transcribe using Groq cloud Whisper API."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            with wave.open(f, "wb") as wav:
                wav.setnchannels(_SAT_CHANNELS)
                wav.setsampwidth(_SAT_WIDTH)
                wav.setframerate(_SAT_RATE)
                wav.writeframes(audio_pcm)

        try:
            return await self.transcribe_audio(tmp_path, language=self.config.stt.language) or ""
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # TTS
    # ------------------------------------------------------------------

    async def _do_tts(self, text: str) -> tuple[bytes, int]:
        """Synthesize text to (pcm_audio_bytes, sample_rate)."""

        def _synthesize() -> tuple[bytes, int]:
            from piper.config import SynthesisConfig

            syn_config = SynthesisConfig()
            if self.config.tts.speaker_id is not None:
                syn_config.speaker_id = self.config.tts.speaker_id

            audio = bytearray()
            rate = 0
            for chunk in self._piper_voice.synthesize(text, syn_config):
                audio.extend(chunk.audio_int16_bytes)
                rate = chunk.sample_rate
            return bytes(audio), rate

        return await asyncio.get_running_loop().run_in_executor(None, _synthesize)

    async def _tts_to_wav(self, text: str, target_rate: int = 0) -> bytes:
        """Synthesize text and return WAV bytes, resampled to target_rate if set."""
        import numpy as np

        tts_audio, tts_rate = await self._do_tts(text)
        if not tts_audio:
            return b""

        # Resample if target rate specified and different from TTS output
        if target_rate and tts_rate != target_rate:
            samples = np.frombuffer(tts_audio, dtype=np.int16).astype(np.float32)
            duration = len(samples) / tts_rate
            target_len = int(duration * target_rate)
            indices = np.linspace(0, len(samples) - 1, target_len)
            resampled = np.interp(indices, np.arange(len(samples)), samples)
            tts_audio = resampled.astype(np.int16).tobytes()
            tts_rate = target_rate

        wav_buf = io.BytesIO()
        with wave.open(wav_buf, "wb") as wf:
            wf.setnchannels(_SAT_CHANNELS)
            wf.setsampwidth(_SAT_WIDTH)
            wf.setframerate(tts_rate)
            wf.writeframes(tts_audio)
        return wav_buf.getvalue()
