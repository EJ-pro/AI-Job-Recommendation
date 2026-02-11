'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { QUESTIONS, QUESTIONS_BEGINNER, Question } from '@/lib/data';
import { calculateRecommendation } from '@/lib/utils';
import QuizCard from '@/components/QuizCard';
import Button from '@/components/Button';
import { Sparkles, GraduationCap, Code } from 'lucide-react';

type Difficulty = 'beginner' | 'advanced' | null;

export default function TestPage() {
    const router = useRouter();
    const [difficulty, setDifficulty] = useState<Difficulty>(null);
    const [currentIndex, setCurrentIndex] = useState(0);
    const [answers, setAnswers] = useState<Record<number, number>>({});

    // Get current questions based on difficulty
    const currentQuestions: Question[] = difficulty === 'beginner' ? QUESTIONS_BEGINNER : QUESTIONS;

    const handleAnswer = (optionIndex: number) => {
        // Save answer
        const currentQuestion = currentQuestions[currentIndex];
        const newAnswers = { ...answers, [currentQuestion.id]: optionIndex };
        setAnswers(newAnswers);

        // Navigate or Finish
        if (currentIndex < currentQuestions.length - 1) {
            setTimeout(() => {
                setCurrentIndex((prev) => prev + 1);
            }, 300); // Slight delay for visual feedback
        } else {
            finishTest(newAnswers);
        }
    };

    const handleSkip = () => {
        // Just move to next question without saving answer
        if (currentIndex < currentQuestions.length - 1) {
            setCurrentIndex((prev) => prev + 1);
        } else {
            finishTest(answers);
        }
    };

    const finishTest = (finalAnswers: Record<number, number>) => {
        const recommendations = calculateRecommendation(finalAnswers, currentQuestions);
        const bestJobId = recommendations[0]?.id;
        const secondJobId = recommendations[1]?.id;

        // Save to localStorage for ResultPage to pick up
        const scores = recommendations.reduce((acc, job) => {
            acc[job.id] = (job as any).score || 0;
            return acc;
        }, {} as Record<string, number>);

        localStorage.setItem('test_result_scores', JSON.stringify(scores));
        localStorage.setItem('test_result_best', bestJobId);
        localStorage.setItem('test_result_second', secondJobId);

        // Redirect to results
        router.push(`/result?best=${bestJobId}&second=${secondJobId}`);
    };

    // Difficulty Selection Screen
    if (!difficulty) {
        return (
            <div className="min-h-screen bg-background flex flex-col items-center justify-center p-6 space-y-12 animate-in fade-in duration-700">
                <div className="text-center space-y-4 max-w-2xl">
                    <h1 className="text-3xl md:text-5xl font-bold">당신의 레벨을 선택해주세요</h1>
                    <p className="text-xl text-muted-foreground">현재 학습 상황에 맞춰서 질문을 구성해 드립니다.</p>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-8 w-full max-w-4xl">
                    {/* Beginner Card */}
                    <button
                        onClick={() => setDifficulty('beginner')}
                        className="group relative p-8 rounded-3xl bg-card border-2 border-border hover:border-primary/50 transition-all hover:shadow-xl hover:-translate-y-2 text-left space-y-6"
                    >
                        <div className="w-16 h-16 rounded-2xl bg-yellow-100 dark:bg-yellow-900/30 flex items-center justify-center text-3xl">
                            🐣
                        </div>
                        <div className="space-y-2">
                            <h2 className="text-2xl font-bold group-hover:text-primary transition-colors">입문자 모드</h2>
                            <p className="text-sm font-semibold text-primary">부트캠프 1~2개월차</p>
                            <p className="text-muted-foreground leading-relaxed">
                                아직 전문 용어가 낯설고,<br />
                                개발 공부를 막 시작한 단계입니다.
                            </p>
                        </div>
                        <div className="absolute inset-x-0 bottom-0 h-1 bg-gradient-to-r from-yellow-400 to-orange-400 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300" />
                    </button>

                    {/* Advanced Card */}
                    <button
                        onClick={() => setDifficulty('advanced')}
                        className="group relative p-8 rounded-3xl bg-card border-2 border-border hover:border-primary/50 transition-all hover:shadow-xl hover:-translate-y-2 text-left space-y-6"
                    >
                        <div className="w-16 h-16 rounded-2xl bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center text-3xl">
                            🔥
                        </div>
                        <div className="space-y-2">
                            <h2 className="text-2xl font-bold group-hover:text-primary transition-colors">실전 모드</h2>
                            <p className="text-sm font-semibold text-primary">부트캠프 3~4개월차 이상</p>
                            <p className="text-muted-foreground leading-relaxed">
                                프로젝트 경험이 있고,<br />
                                구체적인 진로를 고민하는 단계입니다.
                            </p>
                        </div>
                        <div className="absolute inset-x-0 bottom-0 h-1 bg-gradient-to-r from-blue-400 to-purple-400 transform scale-x-0 group-hover:scale-x-100 transition-transform duration-300" />
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center p-4">
            <QuizCard
                question={currentQuestions[currentIndex]}
                currentIndex={currentIndex}
                totalQuestions={currentQuestions.length}
                onAnswer={handleAnswer}
                onSkip={handleSkip}
            />
        </div>
    );
}
