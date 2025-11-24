## Installazione

Il pacchetto può essere installato usando `uv` direttamente dal repository GitHub:

```bash
uv pip git+https://github.com/dnstudent/nivo-reader.git
```

oppure con `pip`:

```bash
pip install git+https://github.com/dnstudent/EasyOCR git+https://github.com/dnstudent/nivo-reader.git
```

Oppure, se si desidera installare in modalità sviluppo:

```bash
pip install git+https://github.com/dnstudent/EasyOCR
git clone https://github.com/dnstudent/nivo-reader.git
cd nivo-reader
pip install -e .
```

## Utilizzo

```bash
nivo-reader \
  folder/ \
  --output-dir ./output \
  --debug-dir ./debug \
  --clips 280 0 0 0 \
  --table-shape 2050 1385 \
  --anagrafica-file .../Anagrafica_Genova.xlsx \
  [opzioni aggiuntive]
```

### Opzioni

#### Obbligatorie

- `images_dir`: Directory contenente le immagini da processare
- `-o, --output-dir`: Directory output per file Excel
- `--clips`: Quattro interi (UP DOWN LEFT RIGHT) per clipping (zone da rimuovere dalla tabella rispettivamente da sopra, sotto, sinistra, destra per isolare la sottotabella contenente solo nomi stazione e dati). Un buon default è "280 0 0 0"
- `--table-shape`: Due interi (WIDTH HEIGHT) per dimensioni tabella (approssimative). Un buon default è "2050 1385"
- `--anagrafica-file`: File excel con nomi stazioni (in una colonna chiamata esattamente "Stazione")

#### Opzionali

- `-d, --debug-dir`: Directory base per artefatti debug (celle rilevate ecc)
- `--station-char-shape`: Dimensioni caratteri stazioni (default: 12 10)
- `--number-char-shape`: Dimensioni caratteri numeri (default: 12 20)
- `--low-confidence-threshold`: Soglia confidenza (default: 0.7)
- `--image-formats`: Formati immagine da processare, separati da virgola (default: png,jpg,jpeg,gif)
- `--overwrite`: Sovrascrivi file già processati


## Aknowledgements

Parte del codice è preso o ispirato da MeteoSaver: [https://github.com/VUB-HYDR/MeteoSaver](https://github.com/VUB-HYDR/MeteoSaver)
