# NIVO Reader Library

Libreria per l'estrazione automatica di dati da tabelle NIVO (tabelle meteorologiche storiche) da immagini scansionate.

## Struttura della libreria

La libreria è organizzata in moduli specializzati per diversi aspetti del processo di estrazione:

### Moduli

- **`image_processing.py`**: Preprocessing delle immagini
  - Rotazione e normalizzazione
  - Binarizzazione e sogliatura
  - Rilevamento struttura tabella

- **`table_detection.py`**: Rilevamento della tabella e linee
  - Localizzazione del rettangolo della tabella
  - Estrazione di linee orizzontali e verticali
  - Rilevamento righe e colonne
  - Rimozione delle linee dalla tabella

- **`ocr_processing.py`**: Elaborazione OCR e riconoscimento testo
  - Rilevamento box stazioni
  - Matching nomi stazioni con anagrafica
  - Autocrop e padding ROI
  - Processamento risultati easyocr

- **`roi_utilities.py`**: Utilità per Regioni di Interesse (ROI)
  - Generazione grid di ROI
  - Clustering e etichettatura coordinate griglia
  - Verifica containment ROI

- **`excel_output.py`**: Output e visualizzazione
  - Scrittura dati a file Excel
  - Salvataggio artefatti debug (immagini intermedie)
  - Disegno bounding box e linee

- **`nivo_reader.py`**: Modulo principale
  - Funzione `read_nivo_table()` che orchester l'intero processo
  - Lettura nomi stazioni
  - Lettura valori tabella

- **`config_imports.py`**: Importazioni configurazione
  - Re-esporta classi di configurazione da `original_parameterization`

## Utilizzo della libreria

### Importazione

```python
from modules.nivo_reader import read_nivo_table
import easyocr

# Inizializzare OCR reader
ocr = easyocr.Reader(lang_list=["it"])

# Processare un'immagine
read_nivo_table(
    original_image,
    excel_out_path="output.xlsx",
    ocr=ocr,
    clips=(260, 0, 0, 0),
    table_shape=(2050, 1385),
    anagrafica=["Stazione 1", "Stazione 2", ...],
    debug_dir="./debug"
)
```

### Parametri di `read_nivo_table`

- **original_image**: Immagine numpy in formato BGR
- **excel_out_path**: Path output file Excel
- **ocr**: Reader easyocr inizializzato
- **clips**: Tuple (up, down, left, right) per clipping margini della tabella
- **table_shape**: Tuple (width, height) dimensioni attese tabella
- **anagrafica**: Lista nomi stazioni note per matching
- **station_char_shape**: Tuple (width, height) dimensioni caratteri nomi stazioni (default: 12, 10)
- **number_char_shape**: Tuple (width, height) dimensioni caratteri numeri (default: 12, 20)
- **roi_padding**: Padding intorno a ROI in pixel (default: 3)
- **nchars_threshold**: Soglia numero caratteri per rilevamento picchi (default: 30)
- **extra_width**: Extra larghezza per celle (default: 6)
- **low_confidence_threshold**: Soglia confidenza per marcatura celle (default: 0.7)
- **debug_dir**: Directory per artefatti debug (opzionale)

## Script batch `process_nivo_images.py`

Script command-line per processare multiple immagini in batch con barra di progresso.

### Utilizzo

```bash
python src/process_nivo_images.py \
  folder/*.jpeg \
  --output-dir ./output \
  --debug-dir ./debug \
  --clips 260 0 0 0 \
  --table-shape 2050 1385 \
  --anagrafica-file stazioni.txt \
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

- `cv2` (OpenCV)
- `numpy`
- `easyocr`
- `PIL` (Pillow)
- `polars`
- `openpyxl`
- `tqdm`
- `scikit-learn` (per clustering)
- `scipy`
- `img2table`

