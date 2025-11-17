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

