# Changelog

All notable changes to the **CNN 3D Visualization** project will be documented in this file.

---

## [1.1.0] - 2026-03-10

### 🎨 Visual Enhancements
- **FC Layer Matrix**: Reorganized the Fully Connected (FC) layer neurons from a linear list/10x12 grid into a structured **8x8 matrix** (64 neurons).
- **Uniform Connectivity**: Implemented an "offset sampling" algorithm for Pool -> FC connections to ensure the entire 8x8 grid visually responds to activations, eliminating empty columns.
- **Visual Contrast**: Added a new toggle to boost the emissive intensity of active neurons and the opacity of active connections, improving visibility in high-activity states.
- **UI Cleanup**: Replaced experimental icons with a standard `Contrast` icon and removed redundant text labels in the main control bar for a more premium look.

### 🌐 Multilingual Support
- **Full Localization**: Integrated comprehensive support for both **English** (now the default language) and **Serbian**.
- **Language Toggle**: Added a global `EN/RS` switch with a globe icon for real-time interface translation.
- **AI Mentor Multilingualism**: 
    - The AI Mentor now uses the current language context for its initial welcome message.
    - Translated all "Quick Actions" and UI elements within the chat interface.
    - Enhanced the system prompt to ensure the AI responds in the selected language.

### 🛠 Technical Improvements
- **TypeScript & Build**: 
    - Fixed a critical TypeScript error in `gemini.ts` by correctly referencing Vite's client-side types for `import.meta.env`.
    - Verified the production build process (`npm run build`).
- **Network Consistency**: Aligned the 3D representation to the actual model architecture (64 FC neurons instead of 80).
- **Default State**: Set the default application language to English and the initial inference source to the first MNIST digit.

### 📝 Documentation
- **Technical Docs**: Updated `TECHNICAL_DETAILS.md` and `TECHNICAL_DETAILS_rs.md` to reflect the current 64-neuron architecture and 8x8 layout.
- **Comprehensive README**: Enhanced the main documentation with new "Results" sections highlighting the 8x8 matrix, multilingual support, and visual contrast features.
- **Versioning**: Global version bump to **v1.1.0**.

---
*Created by the Antigravity team for Advanced Agentic Coding.*
