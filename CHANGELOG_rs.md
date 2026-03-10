# Changelog (Srpski)

Sve značajne promene na projektu **CNN 3D Visualization** biće dokumentovane u ovom fajlu.

---

## [1.1.0] - 2026-03-10

### 🎨 Vizuelna Poboljšanja
- **8x8 FC Matrica**: Neuroni Fully Connected (FC) sloja su reorganizovani iz 10x12 mreže u strukturiranu **8x8 matricu** (64 neurona) radi bolje prostorne organizacije.
- **Uniformne Veze**: Implementiran je "offset sampling" algoritam za Pool -> FC veze koji osigurava da cela 8x8 mreža vizuelno reaguje na aktivacije, eliminišući prazne kolone.
- **Vizuelni Kontrast**: Dodat je novi "toggle" prekidač za pojačavanje intenziteta svetlećih neurona i neprozirnosti aktivnih veza, poboljšavajući vidljivost tokom treninga.
- **Čišćenje Interfejsa**: Eksperimentalne ikonice su zamenjene standardnom `Contrast` ikonicom, a suvišni tekstualni natpisi u kontrolnoj traci su uklonjeni radi "premium" izgleda.

### 🌐 Podrška za više jezika
- **Potpuna Lokalizacija**: Integrisana sveobuhvatna podrška za **Engleski** (sada podrazumevani jezik) i **Srpski**.
- **Prekidač za jezik**: Dodat je globalni `EN/RS` switch sa ikonicom planete za prevođenje interfejsa u realnom vremenu.
- **Multijezičnost AI Mentora**: 
    - AI Mentor sada koristi trenutni jezički kontekst za početni pozdrav.
    - Prevedene su sve brze akcije ("Quick Actions") i UI elementi unutar chat interfejsa.
    - Unapređen sistemski prompt kako bi AI odgovarao na jeziku koji je korisnik izabrao.

### 🛠 Tehnička Poboljšanja
- **TypeScript & Build**: 
    - Rešena kritična TypeScript greška u `gemini.ts` ispravnim referenciranjem Vite tipova klijentske strane za `import.meta.env`.
    - Verifikovan proces produkcionog build-a (`npm run build`).
- **Konzistentnost Mreže**: 3D prikaz je usklađen sa stvarnom arhitekturom modela (64 FC neurona umesto prethodnih 80).
- **Podrazumevano Stanje**: Postavljen podrazumevani jezik aplikacije na engleski i inicijalni izvor inference-a na prvu MNIST cifru.

### 📝 Dokumentacija
- **Tehnička Dokumentacija**: Ažurirani `TECHNICAL_DETAILS.md` i `TECHNICAL_DETAILS_rs.md` da odražavaju trenutnu arhitekturu od 64 neurona i 8x8 raspored.
- **Sveobuhvatni README**: Unapređena glavna dokumentacija novim sekcijama "Results" koje ističu 8x8 matricu, podršku za više jezika i vizuelni kontrast.
- **Verzionisanje**: Globalno podizanje verzije na **v1.1.0**.

---
*Kreirano od strane Antigravity tima za Advanced Agentic Coding.*
