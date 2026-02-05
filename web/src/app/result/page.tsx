'use client';

import { Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import Link from 'next/link';
import { JOBS } from '@/lib/data';
import JobCard from '@/components/JobCard';
import Button from '@/components/Button';

// Separate component to wrap in Suspense for useSearchParams
function ResultContent() {
    const searchParams = useSearchParams();
    const bestId = searchParams.get('best');
    const secondId = searchParams.get('second');

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
