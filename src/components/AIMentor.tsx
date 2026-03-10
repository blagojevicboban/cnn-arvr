import React, { useState, useEffect, useRef } from 'react';
import { Sparkles, MessageSquare, Send, X, Bot, BrainCircuit, Activity, BookOpen, AlertCircle } from 'lucide-react';
import { askGemini } from '../utils/gemini';

interface AIMentorProps {
  isOpen: boolean;
  onClose: () => void;
  lang?: 'en' | 'sr';
  context?: {
    activeLayer?: string;
    trainingHistory?: any[];
    currentMetrics?: { loss: number; accuracy: number };
    epoch?: number;
    step?: number;
  };
}

export function AIMentor({ isOpen, onClose, context, lang = 'en' }: AIMentorProps) {
  const [messages, setMessages] = useState<{ role: 'ai' | 'user'; text: string }[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Initial welcome message
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([{
        role: 'ai',
        text: lang === 'sr' 
          ? 'Zdravo! Ja sam tvoj Gemini-powered AI Mentor. Tu sam da ti objasnim kako funkcioniše ova neuronska mreža. Šta te najviše zanima?'
          : 'Hi! I am your Gemini-powered AI Mentor. I am here to help you understand how this neural network works. What are you most interested in?'
      }]);
    }
  }, [lang]);

  // Scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', text: userMsg }]);
    setIsLoading(true);

    // Context for Gemini
    const geminiContext = {
        layer: context?.activeLayer,
        epoch: context?.epoch,
        metrics: context?.currentMetrics,
        lastSteps: context?.trainingHistory?.slice(-3)
    };

    const response = await askGemini(userMsg, geminiContext);
    setMessages(prev => [...prev, { role: 'ai', text: response }]);
    setIsLoading(false);
  };

  const handleQuickQuestion = async (q: string) => {
    setInput(q);
    // Auto-send
    setTimeout(() => {
        handleSend();
    }, 100);
  };

  if (!isOpen) return null;

  return (
    <div className="absolute top-4 left-4 sm:left-auto sm:right-[310px] z-50 w-full max-w-[340px] h-[550px] max-h-[85vh] bg-black/80 backdrop-blur-2xl border border-blue-500/20 rounded-3xl shadow-[0_0_50px_rgba(59,130,246,0.15)] flex flex-col overflow-hidden animate-in zoom-in-95 fade-in duration-300 pointer-events-auto">
      {/* Header */}
      <div className="p-4 border-b border-white/10 flex justify-between items-center bg-blue-500/10">
        <div className="flex items-center gap-3">
          <div className="bg-blue-500/20 p-2 rounded-xl">
            <Bot size={20} className="text-blue-400" />
          </div>
          <div>
            <h3 className="text-white font-bold text-sm tracking-tight flex items-center gap-2">
                Gemini AI Mentor
                <Sparkles size={14} className="text-yellow-400 animate-pulse" />
            </h3>
            <div className="flex items-center gap-1.5">
                <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
                <span className="text-[10px] text-gray-400 font-mono">System: Active</span>
            </div>
          </div>
        </div>
        <button 
          onClick={onClose}
          className="p-2 hover:bg-white/10 rounded-full transition-colors text-gray-400 hover:text-white"
        >
          <X size={18} />
        </button>
      </div>

      {/* Messages */}
      <div 
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-4 space-y-4 scroll-smooth custom-scrollbar"
      >
        {messages.map((msg, idx) => (
          <div key={idx} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] p-3 rounded-2xl text-xs sm:text-sm leading-relaxed ${
              msg.role === 'user' 
                ? 'bg-blue-600 text-white rounded-tr-none shadow-lg shadow-blue-600/10' 
                : 'bg-white/5 border border-white/5 text-gray-200 rounded-tl-none shadow-inner'
            }`}>
              {msg.text}
            </div>
          </div>
        ))}
        {isLoading && (
          <div className="flex justify-start">
            <div className="bg-white/5 p-3 rounded-2xl rounded-tl-none flex items-center gap-2">
              <div className="flex gap-1">
                <div className="w-1 h-1 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.3s]" />
                <div className="w-1 h-1 bg-blue-400 rounded-full animate-bounce [animation-delay:-0.15s]" />
                <div className="w-1 h-1 bg-blue-400 rounded-full animate-bounce" />
              </div>
              <span className="text-[10px] text-blue-400 font-medium">{lang === 'sr' ? 'Gemini razmišlja...' : 'Gemini is thinking...'}</span>
            </div>
          </div>
        )}
      </div>

      {/* Quick Actions */}
      {messages.length < 3 && !isLoading && (
        <div className="px-4 pb-2 flex flex-wrap gap-2">
            <button 
                onClick={() => handleQuickQuestion(lang === 'sr' ? "Objasni mi konvolucioni sloj." : "Explain the convolutional layer to me.")}
                className="text-[10px] bg-white/5 hover:bg-white/10 border border-white/10 px-2 py-1.5 rounded-lg text-gray-300 transition-all flex items-center gap-1.5"
            >
                <BrainCircuit size={12} className="text-blue-400" />
                {lang === 'sr' ? 'Šta je Conv?' : 'What is Conv?'}
            </button>
            <button 
                onClick={() => handleQuickQuestion(lang === 'sr' ? "Analiziraj metrike mog treninga." : "Analyze my training metrics.")}
                className="text-[10px] bg-white/5 hover:bg-white/10 border border-white/10 px-2 py-1.5 rounded-lg text-gray-300 transition-all flex items-center gap-1.5"
            >
                <Activity size={12} className="text-green-400" />
                {lang === 'sr' ? 'Analiziraj trening' : 'Analyze training'}
            </button>
            <button 
                onClick={() => handleQuickQuestion(lang === 'sr' ? "Kako da smanjim Loss?" : "How do I reduce Loss?")}
                className="text-[10px] bg-white/5 hover:bg-white/10 border border-white/10 px-2 py-1.5 rounded-lg text-gray-300 transition-all flex items-center gap-1.5"
            >
                <AlertCircle size={12} className="text-red-400" />
                {lang === 'sr' ? 'Smanji Loss' : 'Reduce Loss'}
            </button>
        </div>
      )}

      {/* Input */}
      <div className="p-4 border-t border-white/10 bg-white/5">
        <div className="relative group">
          <input 
            type="text" 
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder={lang === 'sr' ? "Pitaj mentora..." : "Ask mentor..."}
            className="w-full bg-black/40 border border-white/10 rounded-2xl py-3 pl-4 pr-12 text-sm text-white focus:outline-none focus:ring-1 focus:ring-blue-500 transition-all group-hover:border-white/20"
          />
          <button 
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className={`absolute right-2 top-1/2 -translate-y-1/2 p-2 rounded-xl transition-all ${
              input.trim() && !isLoading 
                ? 'bg-blue-600 text-white shadow-lg' 
                : 'text-gray-500 opacity-50'
            }`}
          >
            <Send size={16} />
          </button>
        </div>
        <p className="text-[9px] text-gray-500 mt-2 text-center flex items-center justify-center gap-1">
            <BookOpen size={10} />
            Powered by Google Gemini 1.5 Flash
        </p>
      </div>
    </div>
  );
}
