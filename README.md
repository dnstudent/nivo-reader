### Utilizzo

```bash
python process_nivo_images.py \
  folder/*.jpeg \
  --output-dir ./output \
  --debug-dir ./debug \
  --clips 260 0 0 0 \
  --table-shape 2050 1385 \
  --anagrafica-file .../Anagrafica_Genova.xlsx \
  [opzioni aggiuntive]
```

### Opzioni

#### Obbligatorie

- `image_files`: Path delle immagini da processare (uno o più)
- `-o, --output-dir`: Directory output per file Excel
- `--clips`: Quattro interi (UP DOWN LEFT RIGHT) per clipping (zone da rimuvere dalla tabella rispettivamente da sopra, sotto, sinistra, destra per isolare la sottotabella contenente solo nomi stazione e dati)
- `--table-shape`: Due interi (WIDTH HEIGHT) per dimensioni tabella (approssimative)
- `--anagrafica-file`: File excel con nomi stazioni (in una colonna chiamata esattamente "Stazione")

#### Opzionali

- `-d, --debug-dir`: Directory base per artefatti debug (celle rilevate ecc)
- `--station-char-shape`: Dimensioni caratteri stazioni (default: 12 10)
- `--number-char-shape`: Dimensioni caratteri numeri (default: 12 20)
- `--roi-padding`: Padding ROI in pixel (default: 3)
- `--nchars-threshold`: Soglia caratteri (default: 30)
- `--extra-width`: Extra larghezza celle (default: 6)
- `--low-confidence-threshold`: Soglia confidenza (default: 0.7)
- `--overwrite`: Sovrascrivi file già processati

### Funzionalità

1. **Progresso**: Barra di avanzamento con tqdm
2. **Skip file**: Salta automaticamente file già processati (controllando file Excel output)
3. **Overwrite**: Flag `--overwrite` per forzare rielaborazione
4. **Gestione errori**: Continua elaborazione anche se un'immagine fallisce, con reporting
5. **OCR una sola volta**: Inicializza reader easyocr una sola volta per efficienza

## Requisiti

Installabili tramite

```bash
python3 -m pip install -r requirements.txt
```

## Aknowledgements

Parte del codice è preso o ispirato da MeteoSaver: [https://github.com/VUB-HYDR/MeteoSaver](https://github.com/VUB-HYDR/MeteoSaver)
