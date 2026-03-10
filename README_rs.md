<div align="center">
<img width="1200" alt="CNN 3D Vizuelizacija Hero" src="public/cnn-arvr.gif" />

# CNN 3D Visualization & LIVE Training (v1.1.0)
**CNN 3D Visualization** je interaktivna platforma otvorenog koda namenjena edukaciji i istraživanju konvolucionih neuronskih mreža. Omogućava korisnicima da u realnom vremenu prate trening modela direktno u brauzeru, vizuelizuju protok informacija kroz 5-slojnu arhitekturu i eksperimentišu sa sopstvenim setovima podataka.

[![CNN Live Demo](https://img.shields.io/badge/Live-Demo-brightgreen?style=for-the-badge&logo=vercel)](https://blagojevicboban.github.io/cnn-arvr/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
</div>

---

### 🌐 [Live Demo: blagojevicboban.github.io/cnn-arvr](https://blagojevicboban.github.io/cnn-arvr/)

Ova platforma implementira ključne stubove moderne ML vizuelizacije:

### ✅ Result 1: Interaktivni 3D Ekosistem
- **Sloj-po-Sloj Inspekcija**: Svaki sloj (Input, Conv, Pool, FC, Output) je prikazan kao fizički entitet u 3D prostoru.
- **Mape Aktivacija**: Izlazi konvolucionih filtera se renderuju kao dinamičke teksture koje se ažuriraju u realnom vremenu tokom inference i treninga.
- **Neuralni Sjaj (Glow)**: Intenzitet svetlosti neurona u FC slojevima direktno odražava njihovu aktivacionu vrednost ($0.0$ do $1.0$).
- **Dinamičke Veze**: Debljina i boja linija između slojeva vizuelizuju snagu i smer protoka informacija (Backpropagation).

### ✅ Result 2: In-Browser Trening (TF.js)
- **Klijentsko Računanje**: Kompletan trening i inference se izvršavaju unutar korisničkog pretraživača koristeći TensorFlow.js.
- **Web Worker Paralelizacija**: Sva teška ML računanja su izbačena u poseban worker thread, omogućavajući fluidnih 60 FPS za 3D vizuelizaciju čak i tokom intenzivnog treninga.
- **Dual-Model Sinhronizacija**: Sistem koristi dva modela - jedan optimizovan za brzinu treninga i drugi za ekstrakciju unutrašnjih aktivacija za vizuelizaciju.

### ✅ Result 3: 8x8 FC Prikaz Matrice
- **Strukturno Poravnanje**: Fully Connected (FC) sloj je predstavljen kao 8x8 matrica (64 neurona) za bolju prostornu organizaciju.
- **Vizuelizacija Potpune Povezanosti**: Optimizovani algoritmi uzorkovanja osiguravaju da svaki neuron u 8x8 matrici prikazuje vizuelni protok podataka, eliminišući "mrtve zone".

### ✅ Result 4: Multijezičnost i Vizuelni Kontrast
- **EN/RS Prebacivač**: Trenutna promena između engleskog (podrazumevani) i srpskog jezika za sve UI elemente i AI Mentora.
- **Visual Contrast Mod**: Toggle za visok kontrast koji pojačava vidljivost aktivnih neurona i veza.

### ✅ Result 5: Dinamičko Prikupljanje Podataka
- **Dataset Builder**: Korisnici mogu kreirati sopstvene trening setove otpremanjem slika ili korišćenjem ugrađenih MNIST primera.
- **Interaktivno Labeliranje**: Jednostavan interfejs za dodeljivanje labela (0-9) i trenutnu konverziju u tenzorske formate.
- **Augmentacija u Realnom Vremenu**: Sistem automatski vrši grayscale konverziju, promenu veličine (28x28) i pojačavanje kontrasta za optimalne rezultate.

### ✅ Result 6: Vizuelni Monitor Performansi
- **Real-time Recharts**: Integrisani grafikoni prate Gubitak (Loss) i Preciznost (Accuracy) kroz epohe.
- **Checkpoints**: Automatsko čuvanje najboljih modela u `localStorage` pretraživača, omogućavajući nastavak treninga nakon osvežavanja stranice.
- **Statusna Konzola**: Detaljan uvid u stanje Web Workera i progres treninga.

### ✅ Result 7: Gemini AI Mentor
- **Kontekstualna Pomoć**: Chat sa vestačkom inteligencijom koja poznaje vaše trenutne metrike i aktivni sloj.
- **Interaktivna Objašnjenja**: Postavljajte tehnička pitanja poput "Šta radi konvolucioni sloj?" i dobijte trenutne stručne odgovore.
- **Saveti za Optimizaciju**: Dobijte savete u realnom vremenu o tome kako da poboljšate preciznost modela i smanjite gubitak.

---

## 🚀 Ključne Karakteristike
- **3D Renderovanje**: Pokretano pomoću **React Three Fiber** i **Three.js** za vrhunske performanse.
- **Sintetički Generator**: Generisanje hiljada uzoraka koristeći sistemske fontove i `OffscreenCanvas`.
- **Responsive UI**: Moderan interfejs sa staklenim efektom (Glassmorphism) izgrađen pomoću **Tailwind CSS**.
- **Inicijalizacija Težina**: Vizuelna potvrda transformacije iz nasumičnog šuma u prepoznatljive filtere.
- **Gemini AI Integracija**: Mogućnost korišćenja Google GenAI za analizu rezultata i objašnjenje koncepata neuronskih mreža.

## 🛠 Tech Stack
- **Frontend**: React 19, Three.js, React Three Fiber, React Three Drei
- **ML Engine**: TensorFlow.js (CPU/Core backend u workeru)
- **Styling**: Tailwind CSS 4.0
- **Charts**: Recharts
- **Build Tool**: Vite 6.0
- **Icons**: Lucide React

## 💻 Lokalno Pokretanje

### 1. Preduslovi
- **Node.js** (v18+)
- **NPM** ili **Yarn**

### 2. Instalacija
Klonirajte spremište i instalirajte zavisnosti:
```bash
git clone https://github.com/blagojevicboban/cnn-arvr.git
cd cnn-arvr
npm install
```

### 3. Konfiguracija
Setujte `GEMINI_API_KEY` u `.env` fajlu ako planirate da koristite Google GenAI funkcionalnosti:
```env
VITE_GEMINI_API_KEY=vash_api_kljuch
```

### 4. Pokretanje razvojnog servera
```bash
npm run dev
```
Aplikacija će biti dostupna na `http://localhost:3000`.

---

## 🌍 Deployment

Projekat je optimizovan za **GitHub Pages**. Za implementaciju sopstvene verzije:
1. Podesite `base` u `vite.config.ts`.
2. Pokrenite:
   ```bash
   npm run deploy
   ```

## 🐛 Troubleshooting
- **Black Screen**: Proverite konzolu brauzera. Najčešće je uzrok neuspela inicijalizacija `WebGPU` ili `WebGL` konteksta. Aplikacija primarno koristi `CPU` backend unutar workera radi stabilnosti.
- **Gemini Error**: Proverite da li je vaš API ključ validan i da li imate podešen CORS ako pristupate sa neautorizovanih domena.
- **Performance Lag**: Zatvorite ostale tabove koji koriste GPU. Iako worker radi teška računanja, Three.js i dalje zahteva resurse za 60 FPS renderovanje.

## 🤝 Doprinos
Doprinosi su uvek dobrodošli! Ako imate ideju za poboljšanje vizuelizacije ili nove slojeve, slobodno otvorite **Issue** ili pošaljite **Pull Request**.

## 📄 Licenca
Ovaj projekat je licenciran pod **MIT** licencom - pogledajte [LICENSE](LICENSE) fajl za detalje.

---
<div align="center">
<i>Kreirano od strane Antigravity tima u okviru Advanced Agentic Coding projekta.</i>
</div>
