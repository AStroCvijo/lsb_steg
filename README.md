# LSB steganografija

Implementacija algoritma za skrivanje poruka u najnizi bit (LSB - *Least Significant Bit*) RGB piksela slike.

## Instalacija

```
pip install Pillow numpy
```

## Pokretanje primera

```
python generate_samples.py
```

Skripta generise 4 razlicite cover slike, upisuje u svaku po jednu poruku, dekodira je nazad i prikazuje statistiku.

## Kako algoritam radi

1. Poruka -> UTF-8 bajtovi -> niz bitova.
2. Pred poruku dodajemo 32-bitni header sa duzinom (u bitovima).
3. Pikseli slike se citaju u redosledu R, G, B, R, G, B, ...; u najnizi bit svakog kanala upisujemo po jedan bit `(kanal & 0xFE) | bit`.
4. Pri dekodiranju citamo prvih 32 bita -> duzina, zatim toliko bita -> poruka.

Kapacitet: `sirina * visina * 3 - 32` bita. Na slici 512x384 to je ~589 kbita ~= 73 kB teksta.

## Kriptoanaliza

```
python cryptanalysis.py
```

Pokrece blind decode stego slike, snima vizualizaciju LSB ravni u `analysis/`, racuna chi-square test za detekciju, i tabelarno prikazuje PSNR po popunjenosti slike i po dubini LSB-ova (n=1..4). Prethodno mora biti pokrenut `generate_samples.py`.