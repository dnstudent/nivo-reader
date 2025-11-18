## Utilizzo

```bash
python process_nivo_images.py \
  folder/ \
  --output-dir ./output \
  --debug-dir ./debug \
  --clips 260 0 0 0 \
  --table-shape 2050 1385 \
  --anagrafica-file .../Anagrafica_Genova.xlsx \
  [opzioni aggiuntive]
```

### Opzioni

#### Obbligatorie

- `images_dir`: Directory contenente le immagini da processare
- `-o, --output-dir`: Directory output per file Excel
- `--clips`: Quattro interi (UP DOWN LEFT RIGHT) per clipping (zone da rimuovere dalla tabella rispettivamente da sopra, sotto, sinistra, destra per isolare la sottotabella contenente solo nomi stazione e dati). Un buon default è "260 0 0 0"
- `--table-shape`: Due interi (WIDTH HEIGHT) per dimensioni tabella (approssimative). Un buon default è "2050 1385"
- `--anagrafica-file`: File excel con nomi stazioni (in una colonna chiamata esattamente "Stazione")

#### Opzionali

- `-d, --debug-dir`: Directory base per artefatti debug (celle rilevate ecc)
- `--station-char-shape`: Dimensioni caratteri stazioni (default: 12 10)
- `--number-char-shape`: Dimensioni caratteri numeri (default: 12 20)
- `--roi-padding`: Padding ROI in pixel (default: 3)
- `--nchars-threshold`: Soglia di caratteri (default: 30)
- `--extra-width`: Extra larghezza celle (default: 6)
- `--low-confidence-threshold`: Soglia confidenza (default: 0.7)
- `--image-formats`: Formati immagine da processare, separati da virgola (default: png,jpg,jpeg,gif)
- `--overwrite`: Sovrascrivi file già processati

### Requisiti

Installabili tramite

```bash
python3 -m pip install -r requirements.txt
```

## Aknowledgements

Parte del codice è preso o ispirato da MeteoSaver: [https://github.com/VUB-HYDR/MeteoSaver](https://github.com/VUB-HYDR/MeteoSaver)
