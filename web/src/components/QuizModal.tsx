import React, { useState, useEffect } from 'react';
import Button from '@/components/Button';
import { CheckCircle2, XCircle, X } from 'lucide-react';
import clsx from 'clsx';

interface QuizModalProps {
    isOpen: boolean;
    onClose: () => void;
    question: string;
    options: string[];
    correctAnswer: number;
    onComplete: (success: boolean) => void;
}

export function QuizModal({ isOpen, onClose, question, options, correctAnswer, onComplete }: QuizModalProps) {
    const [selectedOption, setSelectedOption] = useState<number | null>(null);
    const [isSubmitted, setIsSubmitted] = useState(false);
    const [isCorrect, setIsCorrect] = useState(false);

    useEffect(() => {
        if (isOpen) {
            // Prevent background scrolling when modal is open
            document.body.style.overflow = 'hidden';
            resetState();
        } else {
            document.body.style.overflow = 'unset';
        }
        return () => {
            document.body.style.overflow = 'unset';
        };
    }, [isOpen]);

    const handleSubmit = () => {
        if (selectedOption === null) return;

        const correct = selectedOption === correctAnswer;
        setIsCorrect(correct);
        setIsSubmitted(true);

        if (correct) {
            setTimeout(() => {
                onComplete(true);
                onClose();
            }, 1500);
        }
    };

    const resetState = () => {
        setSelectedOption(null);
        setIsSubmitted(false);
        setIsCorrect(false);
    };

    if (!isOpen) return null;

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm animate-in fade-in duration-200">
            <div
                className="bg-background w-full max-w-md rounded-2xl shadow-2xl overflow-hidden border border-border animate-in zoom-in-95 duration-200"
                role="dialog"
                aria-modal="true"
            >
                {/* Header */}
                <div className="flex items-center justify-between p-6 border-b border-border">
                    <div>
                        <h2 className="text-xl font-bold">Knowledge Check</h2>
                        <p className="text-sm text-muted-foreground mt-1">
                            다음 단계로 넘어가기 위해 퀴즈를 풀어보세요!
                        </p>
                    </div>
                    <button onClick={onClose} className="text-muted-foreground hover:text-foreground transition-colors">
                        <X size={24} />
                    </button>
                </div>

                {/* Body */}
                <div className="p-6">
                    <h3 className="mb-6 text-lg font-medium leading-relaxed">
                        Q. {question}
                    </h3>

                    <div className="space-y-3">
                        {options.map((option, index) => (
                            <label
                                key={index}
                                className={clsx(
                                    "flex items-center space-x-3 border rounded-xl p-4 cursor-pointer transition-all duration-200",
                                    selectedOption === index ? "border-primary bg-primary/5 ring-1 ring-primary" : "border-border hover:bg-muted/50 hover:border-muted-foreground/30",
                                    isSubmitted && index === correctAnswer && "!border-green-500 !bg-green-500/10 !ring-green-500",
                                    isSubmitted && selectedOption === index && index !== correctAnswer && "!border-red-500 !bg-red-500/10 !ring-red-500"
                                )}
                            >
                                <input
                                    type="radio"
                                    name="quiz-option"
                                    value={index}
                                    checked={selectedOption === index}
                                    onChange={() => !isSubmitted && setSelectedOption(index)}
                                    className="sr-only"
                                    disabled={isSubmitted}
                                />
                                <div className={clsx(
                                    "w-5 h-5 rounded-full border flex items-center justify-center shrink-0",
                                    selectedOption === index ? "border-primary" : "border-muted-foreground/50",
                                    isSubmitted && index === correctAnswer && "border-green-500 bg-green-500 text-white",
                                    isSubmitted && selectedOption === index && index !== correctAnswer && "border-red-500 bg-red-500 text-white"
                                )}>
                                    {isSubmitted && index === correctAnswer && <CheckCircle2 size={12} />}
                                    {isSubmitted && selectedOption === index && index !== correctAnswer && <XCircle size={12} />}
                                    {selectedOption === index && !isSubmitted && <div className="w-2.5 h-2.5 rounded-full bg-primary" />}
                                </div>
                                <span className={clsx("flex-1 text-sm font-medium", isSubmitted && index === correctAnswer && "text-green-600 dark:text-green-400")}>
                                    {option}
                                </span>
                            </label>
                        ))}
                    </div>
                </div>

                {/* Footer */}
                <div className="p-6 border-t border-border bg-muted/20 flex justify-end">
                    {!isSubmitted ? (
                        <Button
                            onClick={handleSubmit}
                            disabled={selectedOption === null}
                            className="w-full sm:w-auto"
                        >
                            제출하기
                        </Button>
                    ) : (
                        <Button
                            onClick={isCorrect ? () => { } : resetState}
                            className={clsx("w-full sm:w-auto", isCorrect ? "bg-green-600 hover:bg-green-700" : "")}
                        >
                            {isCorrect ? "정답입니다! 🎉" : "다시 시도하기"}
                        </Button>
                    )}
                </div>
            </div>
        </div>
    );
}
