import { Question } from '../lib/data';
import Button from './Button';
import ProgressBar from './ProgressBar';
import { twMerge } from 'tailwind-merge';

interface QuizCardProps {
    question: Question;
    currentIndex: number;
    totalQuestions: number;
    onAnswer: (optionIndex: number) => void;
    onSkip?: () => void;
    className?: string;
}

export default function QuizCard({
    question,
    currentIndex,
    totalQuestions,
    onAnswer,
    onSkip,
    className,
}: QuizCardProps) {
    return (
        <div className={twMerge('w-full max-w-2xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500', className)}>
            {/* Progress */}
            <div className="space-y-2">
                <div className="flex justify-between text-sm font-medium text-muted-foreground">
                    <span>Question {currentIndex + 1}</span>
                    <span>{totalQuestions}</span>
                </div>
                <ProgressBar current={currentIndex + 1} total={totalQuestions} />
            </div>

            {/* Question Card */}
            <div className="bg-card text-card-foreground rounded-2xl shadow-xl border border-border p-8 md:p-12 space-y-8 transition-all hover:shadow-2xl hover:shadow-primary/5">
                <h2 className="text-2xl md:text-3xl font-bold leading-tight break-keep text-center">
                    {question.question}
                </h2>

                <div className="space-y-4">
                    {question.options.map((option, index) => (
                        <Button
                            key={index}
                            variant="outline"
                            fullWidth
                            size="lg"
                            onClick={() => onAnswer(index)}
                            className="group relative overflow-hidden border-2 hover:border-primary hover:bg-primary/5 text-left justify-start h-auto py-6 transition-all duration-300"
                        >
                            <span className="relative z-10 text-lg group-hover:text-primary transition-colors">
                                {option.text}
                            </span>
                        </Button>
                    ))}

                    {/* Skip Button */}
                    {onSkip && (
                        <div className="pt-4 flex justify-center">
                            <button
                                onClick={onSkip}
                                className="text-muted-foreground hover:text-foreground underline-offset-4 hover:underline text-sm font-medium transition-colors"
                            >
                                이 질문 건너뛰기
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
}
