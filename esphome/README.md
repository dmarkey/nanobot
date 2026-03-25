# ESPHome Satellite Firmware

Custom ESPHome firmware for voice satellites that work with nanobot's ESPHome voice channel.

## Supported Devices

| Device | Config | Notes |
|--------|--------|-------|
| [M5Stack ATOM Echo](https://thepihut.com/products/atom-echo-smart-speaker-dev-kit) (ESP32) | `atom-echo-office.yaml` | Original Echo, single I2S bus, RGB LED, PDM mic |
| [M5Stack ATOM Echo S3R](https://thepihut.com/products/atom-echos3r-smart-speaker-dev-kit) (ESP32-S3) | `atom-echos3r-office.yaml` | ES8311 codec, 8MB PSRAM, better audio quality |

Both devices share a single I2S bus between mic and speaker — they cannot play audio while the mic is active.

## Prerequisites

Install ESPHome 2025.11.3+:

```bash
uv tool install esphome --from "git+https://github.com/esphome/esphome.git@2025.11.3" --python 3.13
```

On Apple Silicon Macs, you may also need a native `ninja` build:

```bash
brew install ninja
cp $(which ninja) ~/.platformio/packages/tool-ninja/ninja
```

## Setup

1. Copy the secrets example and fill in your WiFi credentials:

```bash
cd esphome
cp secrets.yaml.example secrets.yaml
# Edit secrets.yaml with your WiFi SSID and password
```

2. Place your custom wake word model files (`.tflite` and `.json`) in this directory. The configs reference `my_nano.json` by default — update the `micro_wake_word` `models` section if using a different wake word.

## Compile

```bash
cd esphome
esphome compile atom-echos3r-office.yaml   # S3R
esphome compile atom-echo-office.yaml       # Original Echo
```

## Flash

### First flash (USB)

**ATOM Echo (original):**

```bash
esphome upload atom-echo-office.yaml --device /dev/cu.usbserial-*
```

**ATOM Echo S3R:**

The S3R uses native USB (USB_SERIAL_JTAG). First flash requires a full factory image with explicit flash parameters:

```bash
# Erase flash first
esptool.py --chip esp32s3 --port /dev/cu.usbmodem* erase_flash

# Flash the factory image
esptool.py --chip esp32s3 --port /dev/cu.usbmodem* \
  write_flash --flash_mode dio --flash_size 8MB 0x0 \
  .esphome/build/atom-echos3r-office/.pioenvs/atom-echos3r-office/firmware.factory.bin
```

> **Note:** The S3R won't output serial logs over USB until the firmware boots. If the device appears dead after flashing, check your router for a new DHCP lease — it may have connected to WiFi successfully.

### Subsequent flashes (OTA)

Once the device is on WiFi, flash over the air:

```bash
esphome upload atom-echos3r-office.yaml --device <IP_ADDRESS>
esphome upload atom-echo-office.yaml --device <IP_ADDRESS>
```

## Nanobot Configuration

Add satellites to your nanobot `config.json` under `channels.esphome.satellites`:

**ATOM Echo S3R** (recommended — no workarounds needed):

```json
{
  "name": "office-s3r",
  "host": "192.168.1.139",
  "port": 6053,
  "speechThreshold": 0.7,
  "silenceTimeoutSeconds": 2.0
}
```

**ATOM Echo (original)** (requires `useAnnouncements` workaround):

```json
{
  "name": "office",
  "host": "192.168.1.194",
  "port": 6053,
  "useAnnouncements": true,
  "speechThreshold": 0.65,
  "silenceTimeoutSeconds": 1.2
}
```

## Viewing Logs

```bash
# Over WiFi (API)
esphome logs atom-echos3r-office.yaml --device <IP_ADDRESS>

# Over USB serial (original Echo only)
esphome logs atom-echo-office.yaml --device /dev/cu.usbserial-*
```

## Custom Wake Words

Train a model at [openWakeWord](https://openwakeword.com/) or use [microWakeWord-Trainer](https://github.com/TaterTotterson/microWakeWord-Trainer-AppleSilicon). Place the `.tflite` and `.json` files in this directory and update the `models` section in the YAML config.

## Files

| File | Purpose |
|------|---------|
| `atom-echo-office.yaml` | Firmware config for original ATOM Echo |
| `atom-echos3r-office.yaml` | Firmware config for ATOM Echo S3R |
| `my_nano.json` | Custom wake word manifest |
| `my_nano.tflite` | Custom wake word model |
| `processing.wav` | Thinking/processing feedback sound (16kHz mono) |
| `secrets.yaml.example` | WiFi credentials template |
| `secrets.yaml` | Your WiFi credentials (git-ignored) |
