"use client";


import React, { useState, useRef, useEffect } from "react";

export default function Home() {
  // Placeholder for chat messages
  const [messages, setMessages] = useState<{role: string, content: string}[]>([]);
  const [input, setInput] = useState("");

  // Ref for chat scroll
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  // Scroll to bottom on new message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const sendMessage = (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    // Add user message
    setMessages(prev => [...prev, { role: "user", content: input }]);
    setInput("");

    // Simulate bot response
    setTimeout(() => {
      setMessages(prev => [...prev, { role: "bot", content: "This is a placeholder response." }]);
    }, 1000);
  };

  return (
    <main className="h-screen flex flex-col bg-[var(--unh-white)] overflow-hidden">
      <header className="bg-[var(--unh-blue)] px-8 py-4 text-center shadow-md" style={{ color: '#fff' }}>
  <div className="flex items-center">
    <img src="/unh.svg" alt="UNH Logo" className="my-6 mr-4" style={{ maxWidth: '500px', height: 'auto', width: 'auto', marginTop: '24px', marginBottom: '24px' }} />
    <span className="text-3xl font-bold">Catalog Chatbot</span>
  </div>
      </header>
      <div className="flex-1 flex flex-col items-center overflow-hidden">
        <div className="w-2/3 flex flex-col h-full">
          <div className="flex-1 overflow-y-auto overflow-x-hidden px-4 py-2 scrollbar-hide" style={{wordBreak: 'break-word'}}>
            {messages.map((msg, i) => {
              // Add extra margin if previous message is a different role
              const isRoleChange = i > 0 && messages[i - 1].role !== msg.role;
        if (msg.role === "user") {
                return (
                  <div
                    key={i}
                    className={`flex justify-end mb-2${i === messages.length - 1 ? ' mb-6' : ''}`}
                    style={isRoleChange ? { marginTop: '1rem' } : {}}
                  >
          <div className="bg-[var(--unh-blue)] text-[var(--unh-white)] rounded-2xl break-words whitespace-pre-line m-1 px-6 py-4 text-lg md:text-xl max-w-[800px] w-fit block box-border">
                      {msg.content}
          </div>
                  </div>
                );
              } else {
                return (
                  <div
                    key={i}
                    className={`flex justify-start mb-2${i === messages.length - 1 ? ' mb-6' : ''}`}
                    style={isRoleChange ? { marginTop: '1rem' } : {}}
                  >
          <div className="bg-[var(--unh-light-gray)] text-black rounded-2xl break-words whitespace-pre-line m-1 px-6 py-4 text-lg md:text-xl max-w-[800px] w-fit block box-border">
                      {msg.content}
          </div>
                  </div>
                );
              }
            })}
            <div ref={chatEndRef} />
          </div>
          <form onSubmit={sendMessage} className="flex gap-2 bg-white py-2 px-4">
            <div className="flex-1 flex items-center">
              <div className="relative w-full flex items-center">
                <input
                  className="w-full rounded-full border-2 border-gray-400 text-black pr-14 px-8 py-6 text-lg md:text-xl placeholder:text-gray-400 bg-transparent box-border focus:outline-none"
                  type="text"
                  placeholder="Ask questions about programs, courses, and policies"
                  value={input}
                  onChange={e => setInput(e.target.value)}
                />
                <button
                  type="submit"
                  className="absolute right-3 top-1/2 -translate-y-1/2 rounded-full bg-[var(--unh-blue)] p-3 flex items-center justify-center shadow hover:bg-[var(--unh-accent-blue)] transition-colors"
                  aria-label="Send"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" strokeWidth={2} stroke="white" className="w-6 h-6">
                    <path strokeLinecap="round" strokeLinejoin="round" d="M5 12h14M12 5l7 7-7 7" />
                  </svg>
                </button>
              </div>
            </div>
          </form>
          <div className="w-full px-4 pb-2 text-center text-sm text-gray-500">
            <p className="text-lg">Not answering your question? Contact us at grad.school@unh.edu.</p>
          </div>
        </div>
      </div>
    </main>
  );
}
