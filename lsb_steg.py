from __future__ import annotations

import numpy as np
from PIL import Image

HEADER_BITS = 32


# function for converting raw bytes into a flat bit array (MSB-first)
def _bytes_to_bits(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)


# function for packing a bit array back into bytes
def _bits_to_bytes(bits: np.ndarray) -> bytes:
    return np.packbits(bits).tobytes()


# function for writing an integer as a big-endian bit array of fixed length
def _int_to_bits(value: int, n_bits: int) -> np.ndarray:
    return np.array([(value >> (n_bits - 1 - i)) & 1 for i in range(n_bits)], dtype=np.uint8)


# function for reading a big-endian bit array back into an integer
def _bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for b in bits:
        out = (out << 1) | int(b)
    return out


# function for computing how many message bits an image can hold (header excluded)
def capacity_bits(image_path: str) -> int:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
    return w * h * 3 - HEADER_BITS


# function for computing capacity when using the n lowest bits of each channel
def capacity_bits_n(image_path: str, n: int) -> int:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        w, h = im.size
    return w * h * 3 * n - HEADER_BITS


# function for embedding a message into the LSBs of an image and saving the result
def encode(image_in: str, message: str | bytes, image_out: str) -> None:
    if isinstance(message, str):
        payload = message.encode("utf-8")
    else:
        payload = message

    with Image.open(image_in) as im:
        im = im.convert("RGB")
        pixels = np.array(im, dtype=np.uint8)

    flat = pixels.reshape(-1)
    msg_bits = _bytes_to_bits(payload)
    header = _int_to_bits(len(msg_bits), HEADER_BITS)
    all_bits = np.concatenate([header, msg_bits])

    if all_bits.size > flat.size:
        raise ValueError(
            f"Poruka prevelika: treba {all_bits.size} bita, slika nudi {flat.size} bita."
        )

    flat[: all_bits.size] = (flat[: all_bits.size] & 0xFE) | all_bits

    new_pixels = flat.reshape(pixels.shape)
    Image.fromarray(new_pixels).save(image_out)


# function for extracting raw message bytes from a stego image
def decode(image_in: str) -> bytes:
    with Image.open(image_in) as im:
        im = im.convert("RGB")
        pixels = np.array(im, dtype=np.uint8)

    flat = pixels.reshape(-1)
    lsb = flat & 1

    n_bits = _bits_to_int(lsb[:HEADER_BITS])
    if n_bits <= 0 or n_bits > lsb.size - HEADER_BITS:
        raise ValueError(f"Neispravan header (duzina={n_bits}).")
    if n_bits % 8 != 0:
        raise ValueError("Duzina poruke nije visekratnik od 8 bitova.")

    msg_bits = lsb[HEADER_BITS : HEADER_BITS + n_bits]
    return _bits_to_bytes(msg_bits)


# function for decoding a stego image directly as text
def decode_text(image_in: str, encoding: str = "utf-8") -> str:
    return decode(image_in).decode(encoding)


# function for embedding a message into the n lowest bits of each channel
def encode_n_lsb(image_in: str, message: str | bytes, image_out: str, n: int = 1) -> None:
    if not 1 <= n <= 8:
        raise ValueError("n mora biti u opsegu 1..8.")
    if isinstance(message, str):
        payload = message.encode("utf-8")
    else:
        payload = message

    with Image.open(image_in) as im:
        im = im.convert("RGB")
        pixels = np.array(im, dtype=np.uint8)

    flat = pixels.reshape(-1)
    msg_bits = _bytes_to_bits(payload)
    header = _int_to_bits(len(msg_bits), HEADER_BITS)
    all_bits = np.concatenate([header, msg_bits])

    pad = (-all_bits.size) % n
    if pad:
        all_bits = np.concatenate([all_bits, np.zeros(pad, dtype=np.uint8)])

    n_channels = all_bits.size // n
    if n_channels > flat.size:
        raise ValueError(
            f"Poruka prevelika za n={n}: treba {n_channels} kanala, slika ima {flat.size}."
        )

    groups = all_bits.reshape(n_channels, n)
    values = np.zeros(n_channels, dtype=np.uint8)
    for i in range(n):
        values |= (groups[:, i].astype(np.uint8) << (n - 1 - i))

    clear_mask = np.uint8(0xFF ^ ((1 << n) - 1))
    flat[:n_channels] = (flat[:n_channels] & clear_mask) | values

    new_pixels = flat.reshape(pixels.shape)
    Image.fromarray(new_pixels).save(image_out)


# function for extracting a message hidden in the n lowest bits of each channel
def decode_n_lsb(image_in: str, n: int = 1) -> bytes:
    if not 1 <= n <= 8:
        raise ValueError("n mora biti u opsegu 1..8.")
    with Image.open(image_in) as im:
        im = im.convert("RGB")
        pixels = np.array(im, dtype=np.uint8)

    flat = pixels.reshape(-1)
    take_mask = np.uint8((1 << n) - 1)
    vals = flat & take_mask

    all_bits = np.zeros(flat.size * n, dtype=np.uint8)
    for i in range(n):
        all_bits[i::n] = (vals >> (n - 1 - i)) & 1

    n_bits = _bits_to_int(all_bits[:HEADER_BITS])
    if n_bits <= 0 or n_bits > all_bits.size - HEADER_BITS:
        raise ValueError(f"Neispravan header (duzina={n_bits}).")
    if n_bits % 8 != 0:
        raise ValueError("Duzina poruke nije visekratnik od 8 bitova.")

    msg_bits = all_bits[HEADER_BITS : HEADER_BITS + n_bits]
    return _bits_to_bytes(msg_bits)


# function for comparing cover and stego images (changed channels, PSNR, MSE)
def diff_report(original: str, stego: str) -> dict:
    with Image.open(original) as a, Image.open(stego) as b:
        a_arr = np.array(a.convert("RGB"), dtype=np.int16)
        b_arr = np.array(b.convert("RGB"), dtype=np.int16)

    if a_arr.shape != b_arr.shape:
        raise ValueError("Slike nemaju iste dimenzije.")

    diff = a_arr - b_arr
    changed_channels = int(np.count_nonzero(diff))
    total_channels = diff.size
    max_abs_diff = int(np.max(np.abs(diff)))
    mse = float(np.mean(diff.astype(np.float64) ** 2))
    psnr = float("inf") if mse == 0 else 10 * np.log10((255.0 ** 2) / mse)

    return {
        "changed_channels": changed_channels,
        "total_channels": total_channels,
        "changed_pct": 100.0 * changed_channels / total_channels,
        "max_abs_diff_per_channel": max_abs_diff,
        "mse": mse,
        "psnr_db": psnr,
    }
