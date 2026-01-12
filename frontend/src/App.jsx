import { useState, useRef, useEffect } from "react";
import axios from "axios";
import { motion, AnimatePresence } from "framer-motion";
import { Send, Menu, Bot, LogOut, Sparkles, MessageSquare, Plus, Trash2, ArrowRight } from "lucide-react";
import { onAuthStateChanged, signInWithEmailAndPassword, createUserWithEmailAndPassword, signOut } from "firebase/auth";
import { collection, addDoc, query, where, orderBy, onSnapshot, serverTimestamp, doc, deleteDoc } from "firebase/firestore";
import { auth, db } from "./firebase";

// --- CONFIG ---
const API_URL = "https://bmsit-backend-9o1e.onrender.com/chat";
const MODES = ["Study Buddy", "The Professor", "The Bro", "ELI5"];
const YEARS = ["1", "2", "3", "4"];

// --- HELPER: Detect URLs and make them clickable ---
const renderMessageWithLinks = (text) => {
  const urlRegex = /(https?:\/\/[^\s]+)/g;
  const parts = text.split(urlRegex);

  return parts.map((part, index) => {
    if (part.match(urlRegex)) {
      return (
        <a 
          key={index} 
          href={part} 
          target="_blank" 
          rel="noopener noreferrer" 
          className="text-blue-400 hover:text-blue-300 underline break-all"
        >
          {part}
        </a>
      );
    }
    return part;
  });
};

function App() {
  // --- STATE ---
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);
  
  // Chat Data
  const [chats, setChats] = useState([]); 
  const [currentChatId, setCurrentChatId] = useState(null); 
  const [messages, setMessages] = useState([]); 
  
  // UI Inputs
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState("Study Buddy");
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const messagesEndRef = useRef(null);

  // --- NEW: PERSISTENT YEAR ---
  const [year, setYear] = useState(localStorage.getItem("bmsit_year") || "1");

  // Login Form
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");

  // --- 1. AUTH LISTENER ---
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, (currentUser) => {
      setUser(currentUser);
      setAuthLoading(false);
    });
    return () => unsubscribe();
  }, []);

  // --- 2. SAVE YEAR TO MEMORY ---
  useEffect(() => {
    localStorage.setItem("bmsit_year", year);
  }, [year]);

  // --- 3. LOAD SIDEBAR HISTORY ---
  useEffect(() => {
    if (!user) return;
    const q = query(
      collection(db, "chats"), 
      where("userId", "==", user.uid),
      orderBy("createdAt", "desc")
    );
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const chatList = snapshot.docs.map(doc => ({
        id: doc.id,
        ...doc.data()
      }));
      setChats(chatList);
    });
    return () => unsubscribe();
  }, [user]);

  // --- 4. LOAD MESSAGES ---
  useEffect(() => {
    if (!currentChatId) {
      setMessages([]); 
      return;
    }
    const messagesRef = collection(db, "chats", currentChatId, "messages");
    const q = query(messagesRef, orderBy("createdAt", "asc"));
    const unsubscribe = onSnapshot(q, (snapshot) => {
      const msgs = snapshot.docs.map(doc => doc.data());
      setMessages(msgs);
    });
    return () => unsubscribe();
  }, [currentChatId]);

  // Scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // --- ACTIONS ---

  const handleAuth = async (e) => {
    e.preventDefault();
    setError("");
    try {
      if (isLogin) {
        await signInWithEmailAndPassword(auth, email, password);
      } else {
        await createUserWithEmailAndPassword(auth, email, password);
      }
    } catch (err) {
      setError(err.message.replace("Firebase: ", ""));
    }
  };

  const createNewChat = () => {
    setCurrentChatId(null);
    setMessages([]);
  };

  const deleteChat = async (e, chatId) => {
    e.stopPropagation(); 
    if (window.confirm("Delete this chat?")) {
      await deleteDoc(doc(db, "chats", chatId));
      if (currentChatId === chatId) createNewChat();
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;
    const userText = input;
    setInput(""); 
    setLoading(true);

    try {
      let chatId = currentChatId;
      if (!chatId) {
        const docRef = await addDoc(collection(db, "chats"), {
          userId: user.uid,
          title: userText.slice(0, 30) + "...", 
          createdAt: serverTimestamp()
        });
        chatId = docRef.id;
        setCurrentChatId(chatId);
      }

      await addDoc(collection(db, "chats", chatId, "messages"), {
        role: "user",
        content: userText,
        createdAt: serverTimestamp()
      });

      const token = await user.getIdToken();
      const response = await axios.post(API_URL, {
        message: userText,
        year: year, 
        mode: mode,
        token: token
      });

      const aiResponse = response.data.response;

      await addDoc(collection(db, "chats", chatId, "messages"), {
        role: "assistant",
        content: aiResponse,
        createdAt: serverTimestamp()
      });
      
    } catch (error) {
      console.error("Error:", error);
    } finally {
      setLoading(false);
    }
  };

  // --- RENDER ---

  if (authLoading) return <div className="h-screen bg-black flex items-center justify-center text-white">Loading...</div>;

  // LOGIN SCREEN
  if (!user) {
    return (
      <div className="h-screen w-full bg-black flex flex-col items-center justify-center p-4 relative overflow-hidden">
        <div className="absolute top-[-20%] left-[-10%] w-[500px] h-[500px] bg-blue-900 rounded-full blur-[120px] opacity-20"></div>
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="w-full max-w-md bg-[#0f0f0f] border border-[#222] rounded-2xl p-8 shadow-2xl relative z-10">
          <div className="text-center mb-8">
            <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center mx-auto mb-4 shadow-lg shadow-blue-900/50">
              <Sparkles className="text-white" size={24} />
            </div>
            <h1 className="text-2xl font-bold text-white mb-2">BMSIT Assistant</h1>
            <p className="text-gray-400 text-sm">Log in to access your student portal</p>
          </div>
          <form onSubmit={handleAuth} className="space-y-4">
            <div>
              <label className="text-xs font-semibold text-gray-500 uppercase ml-1">Email</label>
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="w-full bg-[#1e1e1e] border border-[#333] text-white rounded-lg p-3 mt-1 focus:border-blue-500 outline-none transition-colors" placeholder="student@bmsit.in" />
            </div>
            <div>
              <label className="text-xs font-semibold text-gray-500 uppercase ml-1">Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} className="w-full bg-[#1e1e1e] border border-[#333] text-white rounded-lg p-3 mt-1 focus:border-blue-500 outline-none transition-colors" placeholder="••••••••" />
            </div>
            {error && <div className="text-red-500 text-sm text-center bg-red-900/20 p-2 rounded">{error}</div>}
            <button type="submit" className="w-full bg-white text-black font-bold py-3 rounded-lg hover:scale-[1.02] transition-transform flex items-center justify-center gap-2">
              {isLogin ? "Enter Portal" : "Create Account"} <ArrowRight size={18} />
            </button>
          </form>
          <div className="mt-6 text-center">
            <p className="text-gray-500 text-sm">
              {isLogin ? "New here?" : "Already have an ID?"} <button onClick={() => setIsLogin(!isLogin)} className="text-blue-400 font-semibold ml-2 hover:underline">{isLogin ? "Sign Up" : "Log In"}</button>
            </p>
          </div>
        </motion.div>
      </div>
    );
  }

  // APP INTERFACE
  return (
    <div className="flex h-screen w-full bg-black text-white">
      
      {/* SIDEBAR */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div initial={{ width: 0, opacity: 0 }} animate={{ width: 280, opacity: 1 }} exit={{ width: 0, opacity: 0 }} className="bg-[#0f0f0f] border-r border-[#222] flex flex-col h-full">
            <div className="p-4 flex items-center gap-3 border-b border-[#222]">
              <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                <Bot size={20} className="text-white" />
              </div>
              <span className="font-semibold">BMSIT AI</span>
            </div>

            {/* NEW CHAT BUTTON */}
            <div className="p-3">
              <button onClick={createNewChat} className="w-full flex items-center gap-2 bg-[#1e1e1e] hover:bg-[#2a2a2a] text-white p-3 rounded-lg transition-colors border border-[#333]">
                <Plus size={16} /> <span className="text-sm font-medium">New Chat</span>
              </button>
            </div>

            {/* HISTORY LIST */}
            <div className="flex-1 p-3 overflow-y-auto space-y-1">
              <div className="text-xs font-bold text-gray-500 mb-2 px-2 pt-2">RECENT CHATS</div>
              {chats.map((chat) => (
                <div 
                  key={chat.id} 
                  onClick={() => setCurrentChatId(chat.id)}
                  className={`group flex items-center justify-between p-3 rounded-lg cursor-pointer text-sm transition-colors ${currentChatId === chat.id ? "bg-[#1f1f1f] text-white border border-[#333]" : "text-gray-400 hover:bg-[#161616]"}`}
                >
                  <div className="flex items-center gap-2 truncate">
                    <MessageSquare size={14} />
                    <span className="truncate max-w-[140px]">{chat.title}</span>
                  </div>
                  <button onClick={(e) => deleteChat(e, chat.id)} className="opacity-0 group-hover:opacity-100 hover:text-red-400 transition-opacity">
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>

            {/* USER PROFILE */}
            <div className="p-4 border-t border-[#222]">
              <div className="flex items-center gap-3 p-3 bg-[#161616] rounded-xl border border-[#222]">
                <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-purple-500 to-blue-500 flex items-center justify-center text-xs font-bold">
                  {user.email[0].toUpperCase()}
                </div>
                <div className="flex-1 overflow-hidden">
                  <div className="text-sm font-medium truncate">Student</div>
                  <button onClick={() => signOut(auth)} className="text-[10px] text-gray-500 hover:text-red-400 transition-colors flex items-center gap-1">
                    <LogOut size={10} /> Log Out
                  </button>
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* MAIN CHAT AREA */}
      <div className="flex-1 flex flex-col h-full relative">
        {/* TOP BAR */}
        <div className="h-16 flex items-center justify-between px-6 border-b border-[#222]">
          <div className="flex items-center gap-4">
            <button onClick={() => setSidebarOpen(!sidebarOpen)} className="p-2 hover:bg-[#1f1f1f] rounded-lg text-gray-400 hover:text-white transition-colors">
              <Menu size={20} />
            </button>
            
            {/* MODE SELECTOR */}
            <div className="flex items-center gap-2 bg-[#1e1e1e] px-3 py-1.5 rounded-full border border-[#333]">
              <Sparkles size={14} className="text-yellow-400" />
              <select value={mode} onChange={(e) => setMode(e.target.value)} className="bg-transparent text-sm font-medium outline-none cursor-pointer text-gray-200">
                {MODES.map(m => <option key={m} value={m}>{m}</option>)}
              </select>
            </div>

            {/* --- SAVED YEAR SELECTOR --- */}
            <div className="flex items-center gap-2 bg-[#1e1e1e] px-3 py-1.5 rounded-full border border-[#333]">
              <span className="text-xs text-blue-400 font-bold">YEAR</span>
              <select value={year} onChange={(e) => setYear(e.target.value)} className="bg-transparent text-sm font-medium outline-none cursor-pointer text-gray-200">
                {YEARS.map(y => <option key={y} value={y}>{y}</option>)}
              </select>
            </div>

          </div>
        </div>

        {/* MESSAGES */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6 scroll-smooth">
          {messages.length === 0 && (
             <div className="flex h-full flex-col items-center justify-center text-gray-500 opacity-50">
                <Bot size={48} className="mb-4" />
                <p>Start a new conversation...</p>
             </div>
          )}
          {messages.map((msg, index) => (
            <motion.div key={index} initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className={`flex w-full ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              <div className={`max-w-[75%] rounded-2xl p-4 shadow-sm ${msg.role === "user" ? "bg-[#252525] text-white rounded-br-none border border-[#333]" : "bg-transparent text-gray-100"}`}>
                {msg.role === "assistant" && (
                  <div className="flex items-center gap-2 mb-2 text-blue-400 text-xs font-bold uppercase tracking-wide">
                    <Bot size={14} /> BMSIT Assistant
                  </div>
                )}
                <div className="leading-relaxed whitespace-pre-wrap">
                  {/* THIS IS THE FIX: RENDERS LINKS AS CLICKABLE */}
                  {renderMessageWithLinks(msg.content)}
                </div>
              </div>
            </motion.div>
          ))}
          {loading && (
            <div className="flex items-center gap-2 text-gray-500 text-sm ml-4 animate-pulse">
              <Bot size={16} /> Thinking...
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* INPUT */}
        <div className="p-6">
          <div className="max-w-4xl mx-auto bg-[#1e1e1e] rounded-full border border-[#333] flex items-center px-4 py-3 shadow-2xl focus-within:border-blue-600 transition-colors">
            <input type="text" value={input} onChange={(e) => setInput(e.target.value)} onKeyDown={(e) => e.key === "Enter" && handleSend()} placeholder={`Ask Year ${year} question...`} className="flex-1 bg-transparent outline-none text-white placeholder-gray-500 ml-2" />
            <button onClick={handleSend} className={`p-2 rounded-full transition-all ${input.trim() ? "bg-white text-black hover:scale-105" : "bg-[#333] text-gray-500"}`}>
              <Send size={18} />
            </button>
          </div>
          <div className="text-center text-[10px] text-gray-600 mt-3">
            BMSIT AI can make mistakes. Double check important info.
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;