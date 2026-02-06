'use client';

import { useState, useRef, useEffect } from 'react';
import { Send, Loader2, Sparkles, User, Bot, Trash2 } from 'lucide-react';
import Button from './Button';

interface Message {
    role: 'user' | 'assistant';
    content: string;
}

export default function AIConsultant() {
    const [messages, setMessages] = useState<Message[]>([]);
    const [input, setInput] = useState('');
    const [loading, setLoading] = useState(false);
    const scrollRef = useRef<HTMLDivElement>(null);

    // Auto-scroll to bottom directly without smooth behavior for instant feedback
    useEffect(() => {
        if (scrollRef.current) {
            scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
        }
    }, [messages, loading]);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!input.trim() || loading) return;

        const userMessage: Message = { role: 'user', content: input };
        const newMessages = [...messages, userMessage];

        setMessages(newMessages);
        setInput('');
        setLoading(true);

        try {
            const res = await fetch('/api/consult', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ messages: newMessages }),
            });

            const data = await res.json();

            if (!res.ok) {
                throw new Error(data.error || 'Something went wrong');
            }

            const aiMessage: Message = { role: 'assistant', content: data.reply };
            setMessages((prev) => [...prev, aiMessage]);
        } catch (err: any) {
            console.error(err);
            // Optionally handle error state here
            const errorMessage: Message = { role: 'assistant', content: "죄송합니다. 오류가 발생했습니다. 잠시 후 다시 시도해주세요." };
            setMessages((prev) => [...prev, errorMessage]);
        } finally {
            setLoading(false);
        }
    };

    const clearChat = () => {
        if (confirm('대화 내용을 모두 지우시겠습니까?')) {
            setMessages([]);
        }
    };

    return (
        <div className="w-full max-w-4xl mx-auto bg-card rounded-3xl border border-border shadow-2xl overflow-hidden flex flex-col h-[600px] relative">
            {/* Background Decoration */}
            <div className="absolute top-0 right-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -z-10" />

            {/* Header */}
            <div className="p-4 border-b border-border bg-card/50 backdrop-blur-sm flex items-center justify-between z-10">
                <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/10 rounded-full text-primary">
                        <Sparkles size={20} />
                    </div>
                    <div>
                        <h2 className="font-bold text-lg">AI 커리어 멘토</h2>
                        <p className="text-xs text-muted-foreground">당신의 고민을 들어주고 로드맵을 제안합니다.</p>
                    </div>
                </div>
                {messages.length > 0 && (
                    <button onClick={clearChat} className="p-2 text-muted-foreground hover:text-destructive transition-colors">
                        <Trash2 size={18} />
                    </button>
                )}
            </div>

            {/* Chat Area */}
            <div className="flex-1 overflow-y-auto p-4 space-y-4" ref={scrollRef}>
                {messages.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center text-center text-muted-foreground space-y-4 opacity-60">
                        <div className="w-16 h-16 bg-secondary rounded-full flex items-center justify-center mb-2">
                            <Bot size={32} />
                        </div>
                        <div>
                            <p className="text-lg font-medium text-foreground">안녕하세요! 무엇을 도와드릴까요?</p>
                            <p className="text-sm">"비전공자인데 개발자가 되고 싶어요"<br />"3개월 차인데 슬럼프가 왔어요"<br />등 자유롭게 이야기해주세요.</p>
                        </div>
                    </div>
                ) : (
                    messages.map((msg, idx) => (
                        <div key={idx} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                            <div className={`flex max-w-[80%] md:max-w-[70%] gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                                {/* Avatar */}
                                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 mt-1 ${msg.role === 'user' ? 'bg-primary text-primary-foreground' : 'bg-secondary text-secondary-foreground'}`}>
                                    {msg.role === 'user' ? <User size={16} /> : <Bot size={16} />}
                                </div>

                                {/* Bubble */}
                                <div className={`p-4 rounded-2xl text-sm leading-relaxed whitespace-pre-wrap shadow-sm ${msg.role === 'user'
                                        ? 'bg-primary text-primary-foreground rounded-tr-none'
                                        : 'bg-secondary/50 text-foreground border border-border/50 rounded-tl-none'
                                    }`}>
                                    {msg.content}
                                </div>
                            </div>
                        </div>
                    ))
                )}
                {loading && (
                    <div className="flex w-full justify-start">
                        <div className="flex gap-3">
                            <div className="w-8 h-8 rounded-full bg-secondary text-secondary-foreground flex items-center justify-center flex-shrink-0 mt-1">
                                <Bot size={16} />
                            </div>
                            <div className="bg-secondary/50 p-4 rounded-2xl rounded-tl-none border border-border/50 flex items-center gap-2">
                                <Loader2 className="animate-spin h-4 w-4 text-primary" />
                                <span className="text-xs text-muted-foreground">답변을 작성 중입니다...</span>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-border bg-card">
                <form onSubmit={handleSubmit} className="flex gap-2">
                    <input
                        type="text"
                        value={input}
                        onChange={(e) => setInput(e.target.value)}
                        placeholder="메시지를 입력하세요..."
                        className="flex-1 p-3 rounded-xl border border-input bg-background focus:ring-2 focus:ring-primary/50 focus:border-primary transition-all outline-none"
                    />
                    <Button
                        type="submit"
                        disabled={loading || !input.trim()}
                        className="px-4 rounded-xl shadow-md"
                    >
                        <Send size={18} />
                    </Button>
                </form>
            </div>
        </div>
    );
}
