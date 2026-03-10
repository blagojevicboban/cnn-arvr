# CNN 3D Vizuelizacija - Tehnička Dokumentacija
**Verzija: v1.1.0**

## Pregled
Ovaj projekat predstavlja interaktivnu 3D vizuelizaciju konvolucione neuronske mreže (CNN) dizajnirane za prepoznavanje rukom pisanih cifara (MNIST stil). Omogućava korisnicima da prate proces treninga u realnom vremenu, ispituju aktivacije unutrašnjih slojeva i razumeju kako informacije protiču kroz mrežu.

## Osnovna Arhitektura

### 1. Frontend: React + Three.js
- **React**: Upravlja stanjem interfejsa, panelima i logikom aplikacije.
- **React Three Fiber (@react-three/fiber)**: Povezuje React sa Three.js za iscrtavanje 3D scene.
- **Tailwind CSS**: Obezbeđuje moderan, premium izgled kontrolnih panela.

### 2. Računanje: TensorFlow.js + Web Workers
- **Inference & Trening**: Sva teška računanja neuronske mreže su izbačena u **Web Worker** (`inferenceWorker.ts`). Ovo osigurava da interfejs ostane fluidan (60 FPS) čak i tokom aktivnog treninga.
- **Backend**: Koristi `cpu` backend unutar worker-a radi stabilnosti i kompatibilnosti na različitim pretraživačima i hardveru.

### 3. Inteligencija: Google Gemini AI
- **AI Mentor**: Integrisan `gemini-1.5-flash` model za real-time edukaciju.
- **Kontekstualna Svest**: AI prima trenutno stanje vizuelizacije (aktivni sloj, metrike treninga, epohu) kako bi pružio relevantne odgovore.
- **Komunikacija**: Implementiran asinhroni klijent u `utils/gemini.ts` koji komunicira sa Google GenAI API-jem.

## Struktura Neuronske Mreže
Model je sekvencijalni CNN sa sledećim slojevima:
1.  **Input Layer (Ulaz)**: 28x28 grayscale slika.
2.  **Conv2D Layer**: 8 filtera (3x3 kernel), ReLU aktivacija i Batch Normalizacija.
3.  **MaxPooling2D Layer**: 2x2 pooling, smanjuje prostorne dimenzije.
4.  **Fully Connected (FC) Layer**: 64 neurona sa ReLU aktivacijom, Batch Normalizacijom i Dropout-om (25%).
5.  **Output Layer (Izlaz)**: 10 neurona sa Softmax aktivacijom (predstavljaju cifre 0-9).

## Ključne Funkcionalnosti

### Vizuelizacija Treninga u Realnom Vremenu
- **Mape Aktivacija**: Izlazi konvolucionih i pooling slojeva se pretvaraju u teksture u realnom vremenu tokom treninga.
- **Svetlucanje Neurona (Glow)**: Neuroni FC i Output slojeva menjaju intenzitet sjaja na osnovu njihovih stvarnih vrednosti aktivacije ($0.0$ do $1.0$). FC sloj je vizuelizovan kao 8x8 matrica radi preglednosti.
- **Dinamičke Veze**: Debljina i boja linija veza između slojeva odražavaju snagu protoka informacija.

### Monitor Treninga (Training Monitor)
- **Grafik Gubitka (Loss)**: Prati `categoricalCrossentropy` gubitak. Preporučuje se ciljna vrednost ispod **0.1** za stabilna predviđanja.
- **Grafik Preciznosti (Acc)**: Prati procenat tačnih klasifikacija na trenutnom batch-u.
- **Praćenje Epoha**: Vizuelizuje napredak kroz ciklus od 100 epoha treninga.

### Generator Sintetičkih Podataka
- **Font-to-Tensor**: Umesto statičnih slika, sistem koristi `OffscreenCanvas` za generisanje 1000 visokokvalitetnih uzoraka koristeći sistemske fontove (Arial/Sans-serif).
- **Augmentacija**: Nasumična rotacija, translacija i šum se dodaju svakom uzorku kako bi se osiguralo da model dobro generalizuje na različite stilove unosa.

### AI Mentor (Gemini Integration)
- **Problem-Solving**: Pomaže korisnicima da razumeju visoke vrednosti Gubitka (Loss) i niske vrednosti Preciznosti (Acc).
- **Interaktivni Chat**: Omogućava direktno postavljanje pitanja o arhitekturi modela (npr. "Šta radi Conv sloj?").
- **Analiza Metrika**: Funkcija `analyzeTrainingState` automatski interpretira grafikone i daje savete za optimizaciju hiperparametara.

## Tehnički Detalji Implementacije

### Sinhronizacija Težina (Weights Synchronization)
Pošto vizuelizacija zahteva podatke iz međuslojeva (Conv, Pool), worker koristi pristup sa dva modela:
- **Trening Model**: Optimizovan za brzinu, vraća samo finalno predviđanje.
- **Vizuelizacioni Model**: Kompajliran sa više izlaza za ekstrakciju unutrašnjih stanja.
Težine se eksplicitno sinhronizuju koristeći `model.setWeights(trainModel.getWeights())` nakon svakog koraka treninga.

### Obrada Ulaza
Kada korisnik odabere cifru ili otpremi sliku:
1.  Slika se menja na veličinu 28x28.
2.  Vrši se **Grayscale konverzija** ($0.299R + 0.587G + 1.14B$).
3.  Primenjuje se **Pojačanje Contrasta** kako bi tanke linije fontova bile jasno vidljive konvolucionim filterima.

## Uputstvo za Korišćenje
1.  **Inicijalizacija**: Pri učitavanju, mreža ima nasumične težine.
2.  **Trening**: Otvorite **Trening Monitor** i kliknite na **Start**. Pratite pad crvene linije (Loss).
3.  **Testiranje**: Kada Loss padne ispod 0.1, koristite **MNIST Input** panel da odaberete cifru. 3D scena će se ažurirati i pokazati kako je mreža klasifikuje.
4.  **Inspekcija**: Kliknite na bilo koji 3D sloj da fokusirate kameru i vidite njegove specifične parametre.

---
*Kreirao Antigravity tim za Advanced Agentic Coding.*
