'use client';

import { Suspense, useEffect, useState } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { JOBS } from '@/lib/data';
import JobCard from '@/components/JobCard';
import Button from '@/components/Button';
import AIAdviser from '@/components/AIAdviser';
import { useAuth } from '@/context/AuthContext';
import SignupPromptModal from '@/components/SignupPromptModal';

// Separate component to wrap in Suspense for useSearchParams
function ResultContent() {
    const searchParams = useSearchParams();
    const bestId = searchParams.get('best');
    const secondId = searchParams.get('second');
    const { user } = useAuth();
    const [showSignupPrompt, setShowSignupPrompt] = useState(false);

    useEffect(() => {
        // Show prompt after 30 seconds if user is not logged in
        if (!user) {
            const timer = setTimeout(() => {
                setShowSignupPrompt(true);
            }, 30000);

            return () => clearTimeout(timer);
        }
    }, [user]);

    const bestJob = JOBS.find((j) => j.id === bestId);
    const secondJob = JOBS.find((j) => j.id === secondId);

    if (!bestJob) {
        return (
            <div className="text-center space-y-4">
                <h1 className="text-2xl font-bold">결과를 찾을 수 없습니다.</h1>
                <Link href="/test">
                    <Button>다시하기</Button>
                </Link>
            </div>
        );
    }

    return (
        <div className="max-w-4xl mx-auto space-y-12 animate-in fade-in slide-in-from-bottom-8 duration-700">
            <div className="text-center space-y-4">
                <h1 className="text-3xl md:text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-primary to-secondary">
                    당신에게 추천하는 AI 직무
                </h1>
                <p className="text-lg text-muted-foreground">
                    답변을 기반으로 가장 잘 어울리는 포지션을 선정했습니다.
                </p>
            </div>

            <div className="grid grid-cols-1 gap-8">
                <JobCard job={bestJob} rank={1} />

                {secondJob && (
                    <div className="mt-8">
                        <h2 className="text-2xl font-bold mb-6 text-center text-muted-foreground">
                            이런 직무도 어울려요
                        </h2>
                        <JobCard job={secondJob} rank={2} />
                    </div>
                )}
            </div>

            {/* Focus Areas & Roadmap Section */}
            <div className="space-y-6">
                {/* Focus Areas */}
                <div className="bg-white/50 dark:bg-black/20 rounded-2xl p-6 border border-border/50">
                    <h3 className="text-lg font-bold mb-4 flex items-center gap-2">
                        🔥 3개월 집중 공략 분야
                        <span className="text-xs font-normal text-muted-foreground">(선택과 집중)</span>
                    </h3>
                    <div className="grid gap-3 sm:grid-cols-3">
                        {bestJob.focus_areas?.map((area, idx) => (
                            <div key={idx} className="bg-background/80 p-3 rounded-xl text-sm font-medium border border-primary/10 text-center shadow-sm">
                                {area}
                            </div>
                        ))}
                    </div>
                </div>

                {/* 3-Month Roadmap */}
                <div className="bg-white/50 dark:bg-black/20 rounded-2xl p-6 border border-border/50">
                    <h3 className="text-lg font-bold mb-4">📅 월별 학습 로드맵</h3>
                    <div className="space-y-4">
                        {bestJob.roadmap?.map((item, idx) => (
                            <div key={idx} className="relative pl-6 border-l-2 border-primary/20 pb-1 last:pb-0">
                                <div className="absolute -left-[9px] top-0 w-4 h-4 rounded-full bg-primary border-4 border-background" />
                                <div className="mb-1 text-sm font-bold text-primary">{item.step}</div>
                                <p className="text-sm text-muted-foreground">{item.action}</p>
                            </div>
                        ))}
                    </div>
                </div>
            </div>

            {/* AI Adviser Section */}
            <div className="pt-2">
                <AIAdviser jobTitle={bestJob.title} />
            </div>

            <div className="text-center pt-8 pb-16">
                <div className="flex flex-col md:flex-row gap-4 justify-center items-center">
                    <Link href="/">
                        <Button variant="outline" size="lg">
                            처음으로 돌아가기
                        </Button>
                    </Link>
                    <Button
                        className="bg-zinc-800 text-white hover:bg-zinc-700"
                        onClick={() => {
                            if (navigator.share) {
                                navigator.share({
                                    title: 'AI 커리어 가이드 결과',
                                    text: `나에게 맞는 AI 직무는 ${bestJob.title}입니다!`,
                                    url: window.location.href,
                                }).catch(() => { });
                            } else {
                                navigator.clipboard.writeText(window.location.href);
                                alert('링크가 복사되었습니다!');
                            }
                        }}
                    >
                        결과 공유하기 🔗
                    </Button>
                </div>
            </div>

            <SignupPromptModal
                isOpen={showSignupPrompt}
                onClose={() => setShowSignupPrompt(false)}
            />
        </div>
    );
}

export default function ResultPage() {
    return (
        <div className="min-h-screen bg-background p-6 md:p-12">
            <Suspense fallback={<div className="text-center p-12">분석 중...</div>}>
                <ResultContent />
            </Suspense>
        </div>
    );
}
