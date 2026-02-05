'use client';

import { useState, useEffect } from 'react';
import Button from './Button';
import { Sparkles, Bot, Loader2 } from 'lucide-react';

interface AIAdviserProps {
    jobTitle: string;
}

export default function AIAdviser({ jobTitle }: AIAdviserProps) {
    const [status, setStatus] = useState<'idle' | 'loading' | 'completed'>('idle');
    const [message, setMessage] = useState('');

    // API call logic
    const handleAnalyze = async () => {
        setStatus('loading');
        setMessage('');

        try {
            const res = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ jobTitle }),
            });

            const data = await res.json();

            if (!res.ok) throw new Error(data.error || 'Something went wrong');

            setStatus('completed');
            startTypingEffect(data.result);
        } catch (error) {
            console.error('API Error:', error);
            // Fallback to mock data on error (e.g. Quota exceeded)
            setStatus('completed');
            const fallbackMessage = `(API 연결 불안정으로 예시 답변을 표시합니다)\n\n회원님의 성향을 분석해보니, **${jobTitle}** 직무가 정말 잘 어울립니다.\n\n단순히 개발을 좋아하는 것을 넘어, 시스템의 원리를 이해하고 최적화하는 데 강점이 있으시네요. 특히 최신 AI 기술을 활용하여 실질적인 가치를 만들어내는 능력은 현업에서 가장 필요로 하는 역량입니다.\n\n지금 바로 관련 포트폴리오를 준비해보세요! 🚀`;
            startTypingEffect(fallbackMessage);
        }
    };

    const startTypingEffect = (text: string) => {
        let i = 0;
        const interval = setInterval(() => {
            setMessage((prev) => text.slice(0, i + 1));
            i++;
            if (i >= text.length) clearInterval(interval);
        }, 30);
    };

    return (
        <div className="w-full max-w-3xl mx-auto mt-16 p-1 rounded-3xl bg-gradient-to-r from-primary via-purple-500 to-secondary animate-in fade-in slide-in-from-bottom-8 duration-1000">
            <div className="bg-card rounded-[22px] p-6 md:p-8">
                <div className="flex items-start gap-4">
                    <div className="p-3 bg-primary/10 rounded-2xl shrink-0">
                        <Bot className="w-8 h-8 text-primary" />
                    </div>

                    <div className="space-y-4 w-full">
                        <div>
                            <h3 className="text-xl font-bold flex items-center gap-2">
                                AI 커리어 멘토의 한마디
                                <span className="text-xs font-normal px-2 py-0.5 rounded-full bg-primary/10 text-primary uppercase tracking-wide">Beta</span>
                            </h3>
                            <p className="text-muted-foreground mt-1">
                                선택한 답변을 바탕으로 AI가 분석한 커리어 조언을 확인해보세요.
                            </p>
                        </div>

                        {status === 'idle' && (
                            <div className="pt-2">
                                <Button
                                    onClick={handleAnalyze}
                                    className="bg-gradient-to-r from-primary to-secondary hover:opacity-90 transition-opacity text-white border-0"
                                >
                                    <Sparkles className="w-4 h-4 mr-2" />
                                    AI 분석 요청하기
                                </Button>
                            </div>
                        )}

                        {status === 'loading' && (
                            <div className="flex items-center gap-2 text-primary font-medium p-4 bg-primary/5 rounded-xl">
                                <Loader2 className="w-5 h-5 animate-spin" />
                                답변을 분석하여 조언을 생성하고 있습니다...
                            </div>
                        )}

                        {status === 'completed' && (
                            <div className="bg-muted/50 p-6 rounded-xl border border-border/50">
                                <p className="text-lg leading-relaxed whitespace-pre-wrap">
                                    {message}
                                    <span className="inline-block w-2 h-5 bg-primary/50 ml-1 animate-pulse align-middle" />
                                </p>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </div>
    );
}
