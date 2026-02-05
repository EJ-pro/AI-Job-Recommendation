'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { QUESTIONS } from '@/lib/data';
import { calculateRecommendation } from '@/lib/utils';
import QuizCard from '@/components/QuizCard';

export default function TestPage() {
    const router = useRouter();
    const [currentIndex, setCurrentIndex] = useState(0);
    const [answers, setAnswers] = useState<Record<number, number>>({});

    const handleAnswer = (optionIndex: number) => {
        // Save answer
        const currentQuestion = QUESTIONS[currentIndex];
        const newAnswers = { ...answers, [currentQuestion.id]: optionIndex };
        setAnswers(newAnswers);

        // Navigate or Finish
        if (currentIndex < QUESTIONS.length - 1) {
            setTimeout(() => {
                setCurrentIndex((prev) => prev + 1);
            }, 300); // Slight delay for visual feedback
        } else {
            finishTest(newAnswers);
        }
    };

    const finishTest = (finalAnswers: Record<number, number>) => {
        const recommendations = calculateRecommendation(finalAnswers);
        const bestJobId = recommendations[0]?.id;
        const secondJobId = recommendations[1]?.id;

        // Redirect to results
        router.push(`/result?best=${bestJobId}&second=${secondJobId}`);
    };

    return (
        <div className="min-h-screen bg-background text-foreground flex flex-col items-center justify-center p-4">
            <QuizCard
                question={QUESTIONS[currentIndex]}
                currentIndex={currentIndex}
                totalQuestions={QUESTIONS.length}
                onAnswer={handleAnswer}
            />
        </div>
    );
}
