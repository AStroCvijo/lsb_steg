from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from lsb_steg import capacity_bits, decode_text, diff_report, encode

OUT_DIR = Path(__file__).parent / "samples"


# function for building a smooth RGB gradient cover image
def make_gradient(path: Path, size=(512, 384)) -> None:
    w, h = size
    x = np.linspace(0, 1, w, dtype=np.float32)
    y = np.linspace(0, 1, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    r = (255 * xx).astype(np.uint8)
    g = (255 * yy).astype(np.uint8)
    b = (255 * (1 - xx * yy)).astype(np.uint8)
    img = np.stack([r, g, b], axis=-1)
    Image.fromarray(img).save(path)


# function for building a checkerboard cover image with sharp edges
def make_checker(path: Path, size=(512, 384), tile=32) -> None:
    w, h = size
    xs = (np.arange(w) // tile) % 2
    ys = (np.arange(h) // tile) % 2
    pattern = (xs[None, :] ^ ys[:, None]).astype(np.uint8) * 255
    img = np.stack([pattern, pattern // 2, 255 - pattern], axis=-1)
    Image.fromarray(img).save(path)


# function for building a random-noise cover image (hardest to detect LSB changes)
def make_noise(path: Path, size=(512, 384), seed=42) -> None:
    rng = np.random.default_rng(seed)
    w, h = size
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    Image.fromarray(img).save(path)


# function for building a photo-like cover image (background, circle, slight noise)
def make_photo_like(path: Path, size=(512, 384), seed=7) -> None:
    rng = np.random.default_rng(seed)
    w, h = size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    bg_r = 80 + 100 * (xx / w)
    bg_g = 120 + 80 * (yy / h)
    bg_b = 180 - 60 * ((xx + yy) / (w + h))

    cx, cy, rad = w * 0.6, h * 0.55, min(w, h) * 0.25
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= rad ** 2

    r = np.where(mask, 220.0, bg_r)
    g = np.where(mask, 90.0, bg_g)
    b = np.where(mask, 70.0, bg_b)

    noise = rng.normal(0, 4, size=(h, w, 3))
    img = np.stack([r, g, b], axis=-1) + noise
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)


# function for running one encode/decode example and printing its statistics
def run_example(name: str, cover_path: Path, message: str) -> None:
    stego_path = OUT_DIR / f"{name}_stego.png"
    encode(str(cover_path), message, str(stego_path))
    recovered = decode_text(str(stego_path))
    report = diff_report(str(cover_path), str(stego_path))

    assert recovered == message, f"Mismatch u primeru {name}!"

    print(f"\n=== Primer: {name} ===")
    print(f"  cover:  {cover_path.name}")
    print(f"  stego:  {stego_path.name}")
    print(f"  kapacitet: {capacity_bits(str(cover_path))} bita "
          f"(~{capacity_bits(str(cover_path)) // 8} bajtova)")
    print(f"  poruka ({len(message)} char): {message[:60]}{'...' if len(message) > 60 else ''}")
    print(f"  dekodirana == originalna: OK")
    print(f"  promenjeno kanala: {report['changed_channels']} / {report['total_channels']} "
          f"({report['changed_pct']:.3f}%)")
    print(f"  max razlika po kanalu: {report['max_abs_diff_per_channel']} (ocekivano <=1)")
    print(f"  PSNR: {report['psnr_db']:.2f} dB  (vise od ~50 dB = covek ne vidi razliku)")


# function for running an example on a real photograph (JPEG in, PNG stego out)
def run_real_photo_example(name: str, photo_path: Path, message: str) -> None:
    stego_path = OUT_DIR / f"{name}_stego.png"
    encode(str(photo_path), message, str(stego_path))
    recovered = decode_text(str(stego_path))
    report = diff_report(str(photo_path), str(stego_path))

    assert recovered == message, f"Mismatch u primeru {name}!"

    from PIL import Image
    with Image.open(photo_path) as im:
        w, h = im.size
    cap_chars = capacity_bits(str(photo_path)) // 8

    print(f"\n=== Primer (PRAVA FOTOGRAFIJA): {name} ===")
    print(f"  cover:  {photo_path.name}  ({w}x{h} px)")
    print(f"  stego:  {stego_path.name}")
    print(f"  kapacitet: ~{cap_chars} bajtova (~{cap_chars // 1024} kB teksta)")
    print(f"  poruka ({len(message)} char): {message[:60]}{'...' if len(message) > 60 else ''}")
    print(f"  dekodirana == originalna: OK")
    print(f"  promenjeno kanala: {report['changed_channels']} / {report['total_channels']} "
          f"({report['changed_pct']:.5f}%)")
    print(f"  max razlika po kanalu: {report['max_abs_diff_per_channel']} (ocekivano <=1)")
    print(f"  PSNR: {report['psnr_db']:.2f} dB  (vise od ~50 dB = covek ne vidi razliku)")


# function for generating all cover images and running every example
def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    covers = {
        "gradient": OUT_DIR / "cover_gradient.png",
        "checker":  OUT_DIR / "cover_checker.png",
        "noise":    OUT_DIR / "cover_noise.png",
        "photo":    OUT_DIR / "cover_photo.png",
    }
    make_gradient(covers["gradient"])
    make_checker(covers["checker"])
    make_noise(covers["noise"])
    make_photo_like(covers["photo"])

    short_msg = "Tajna poruka: sastanak u 21h."
    long_msg = (
        "LSB steganografija krije podatke u najnizem bitu svakog kanala piksela. "
        "Promena je +/-1 po kanalu, sto je za ljudsko oko nevidljivo na fotografijama "
        "sa prirodnim teksturama. Ali ako sliku rekomprimujes u JPEG, poruka nestaje, "
        "jer JPEG menja LSB-ove pri kvantizaciji. " * 3
    )
    binary_msg = bytes(range(256)) * 4

    run_example("gradient_short", covers["gradient"], short_msg)
    run_example("checker_long",   covers["checker"],  long_msg)
    run_example("noise_long",     covers["noise"],    long_msg)
    run_example("photo_short",    covers["photo"],    short_msg)

    from lsb_steg import decode, encode as enc
    stego_bin = OUT_DIR / "photo_binary_stego.png"
    enc(str(covers["photo"]), binary_msg, str(stego_bin))
    rec = decode(str(stego_bin))
    assert rec == binary_msg
    rep = diff_report(str(covers["photo"]), str(stego_bin))
    print(f"\n=== Primer: photo_binary (1024 bajta proizvoljnih byte vrednosti) ===")
    print(f"  dekodirano OK, promenjeno {rep['changed_channels']} kanala, "
          f"PSNR={rep['psnr_db']:.2f} dB")

    real_msg = (
        "Tajna poruka skrivena u pravoj fotografiji pomocu LSB steganografije. "
        "Iako fotografija izgleda potpuno nepromenjeno, u najnize bitove piksela "
        "upisan je ovaj tekst. Posto fotografija ima milione piksela, promena je "
        "potpuno nevidljiva golim okom, a kapacitet je dovoljan za cele stranice teksta."
    )
    real_photos = {
        "real_photo1": OUT_DIR / "image1.jpg",
        "real_photo2": OUT_DIR / "image2.jpg",
    }
    for name, path in real_photos.items():
        if path.exists():
            run_real_photo_example(name, path, real_msg)
        else:
            print(f"\n(preskocena prava fotografija: {path.name} ne postoji)")

    print(f"\nSve slike su u: {OUT_DIR}")


if __name__ == "__main__":
    main()
