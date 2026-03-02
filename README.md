## Installazione

Il pacchetto può essere installato usando `uv` direttamente dal repository GitHub:

```bash
uv pip install git+https://github.com/dnstudent/nivo-reader.git
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

Da terminale lanciare un comando sulla falsariga del seguente:

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

Adattare i parametri in base alle proprie esigenze. Il roulo di ciascuno di essi è spiegato nella sezione sottostante.

### Opzioni

#### Obbligatorie

- `images_dir`: Directory contenente le immagini da processare. Possono essere distribuite in un numero arbitrario di sottocartelle; la struttura della directory viene conservata nell'output.
- `-o, --output-dir`: Directory per i file Excel in output
- `--clips`: Quattro interi (UP DOWN LEFT RIGHT) per il "clipping": lunghezze (in pixel) da rimuovere partendo dai lati dalla tabella rispettivamente da sopra, sotto, sinistra, destra per rimouvere le intestazioni ed eventuali altre decorazioni che non sono da passare all'OCR. Un buon default è "280 0 0 0"
- `--table-shape`: Due interi (WIDTH HEIGHT) che danno le dimensioni approssimative delle tabelle rappresentate nelle immagini, in pixel. Un buon default è "2050 1385"
- `--anagrafica-file`: File excel con i nomi delle stazioni (in una colonna chiamata esattamente "Stazione")

#### Opzionali

- `-d, --debug-dir`: Directory base per artefatti debug (celle rilevate ecc)
- `--station-char-shape`: Dimensioni approssimative dei caratteri dei nomi stazione, in pixel (WIDTH HEIGHT; default: 12 10)
- `--number-char-shape`: Dimensioni approssimative dei caratteri dei numeri, in pixel (WIDTH HEIGHT; default: 12 20)
<!--- `--low-confidence-threshold`: Soglia di confidenza sotto la  (default: 0.7)-->
- `--image-formats`: Formati delle immagini da processare, separati da virgola (default: png,jpg,jpeg,gif)
- `--overwrite`: Sovrascrivi file già processati; se questa flag non viene usata, le immagini già processate vengono saltate
- `--ocr-engines tesseract,easyocr,paddleocr`: OCR da usare, separati da virgola. Per ora sono supportati tesseract, easyocr e paddleocr


## Aknowledgements

Parte del codice è preso o ispirato da MeteoSaver: [https://github.com/VUB-HYDR/MeteoSaver](https://github.com/VUB-HYDR/MeteoSaver)
