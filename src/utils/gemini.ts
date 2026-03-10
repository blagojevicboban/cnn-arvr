import { GoogleGenerativeAI } from "@google/generative-ai";

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY || "";
const genAI = new GoogleGenerativeAI(API_KEY);

export async function askGemini(prompt: string, context?: any) {
  if (!API_KEY || API_KEY === "MY_GEMINI_API_KEY") {
    return "Gemini API ključ nije podešen. Molimo podesite VITE_GEMINI_API_KEY u .env fajlu.";
  }

  const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

  const fullPrompt = `
    Kao stručnjak za Deep Learning i CNN (Convolutional Neural Networks), pomozi korisniku da razume vizuelizaciju.
    
    KONTEKST SISTEMA:
    - Arhitektura: MNIST CNN (Input -> Conv -> Pool -> FC -> Output)
    - Web vizuelizacija: React + Three.js
    
    TRENUTNI PODACI:
    ${context ? JSON.stringify(context, null, 2) : "Nema dodatnih podataka."}
    
    PITANJE KORISNIKA:
    ${prompt}
    
    Odgovori na jeziku na kojem je postavljeno pitanje (podrazumevano srpski). Budi kratak, stručan i motivišući.
  `;

  try {
    const result = await model.generateContent(fullPrompt);
    const response = await result.response;
    return response.text();
  } catch (error) {
    console.error("Gemini Error:", error);
    return "Došlo je do greške u komunikaciji sa Gemini AI servisom.";
  }
}

export async function analyzeTrainingState(history: any[]) {
  if (history.length === 0) return "Trening još nije počeo. Kliknite na 'Start' u monitoru.";
  
  const lastMetrics = history[history.length - 1];
  const prompt = `Analiziraj trenutne metrike treninga: Loss=${lastMetrics.loss}, Accuracy=${lastMetrics.accuracy}. Da li model dobro uči? Daj kratak savet.`;
  
  return askGemini(prompt, { history: history.slice(-5) });
}
