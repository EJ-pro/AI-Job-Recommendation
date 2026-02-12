'use client';

import React, { useState, useEffect } from 'react';
import { RoadmapStep } from '@/lib/data';
import { ProgressHero } from './ProgressHero';
import { QuizModal } from './QuizModal';
import { RoadmapOverview } from './RoadmapOverview';
import { CheckCircle2, Lock, ChevronRight, BookOpen, ExternalLink, PlayCircle } from 'lucide-react';
import clsx from 'clsx';
import Button from '@/components/Button';

interface JobReadinessTrackerProps {
    jobId: string;
    roadmap: RoadmapStep[];
}

export function JobReadinessTracker({ jobId, roadmap }: JobReadinessTrackerProps) {
    const [completedSteps, setCompletedSteps] = useState<number[]>([]);
    const [activeStep, setActiveStep] = useState<number | null>(null);
    const [quizStepIndex, setQuizStepIndex] = useState<number | null>(null);

    // Load progress from localStorage
    useEffect(() => {
        const saved = localStorage.getItem(`job-progress-${jobId}`);
        if (saved) {
            setCompletedSteps(JSON.parse(saved));
        }
    }, [jobId]);

    // Save progress
    useEffect(() => {
        localStorage.setItem(`job-progress-${jobId}`, JSON.stringify(completedSteps));
    }, [jobId, completedSteps]);

    const progress = Math.round((completedSteps.length / roadmap.length) * 100);

    // Sort roadmap: Incomplete steps first, Completed steps last.
    // Within each group, maintain original order (index).
    const sortedRoadmapWithIndex = roadmap
        .map((step, index) => ({ step, index }))
        .sort((a, b) => {
            const aCompleted = completedSteps.includes(a.index);
            const bCompleted = completedSteps.includes(b.index);

            if (aCompleted === bCompleted) {
                return a.index - b.index; // Maintain original order
            }
            return aCompleted ? 1 : -1; // Move completed to bottom
        });

    const handleStepClick = (originalIndex: number) => {
        if (completedSteps.includes(originalIndex)) return; // Already done

        // Check if previous step is completed (optional enforcement)
        // if (index > 0 && !completedSteps.includes(index - 1)) return; 

        if (roadmap[originalIndex].quiz) {
            setQuizStepIndex(originalIndex);
        } else {
            // No quiz, mark as done immediately (or keep as just content view)
            // For now, let's just mark it done to be friendly if no quiz exists
            markStepComplete(originalIndex);
        }
    };

    const markStepComplete = (originalIndex: number) => {
        if (!completedSteps.includes(originalIndex)) {
            setCompletedSteps([...completedSteps, originalIndex]);
        }
        setQuizStepIndex(null);
    };

    return (
        <div className="w-full max-w-4xl mx-auto">
            <ProgressHero progress={progress} />

            <RoadmapOverview
                roadmap={roadmap}
                completedSteps={completedSteps}
                onStepClick={(index) => {
                    // Optional: Scroll to the step or just view details
                    const element = document.getElementById(`step-${index}`);
                    if (element) {
                        element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    }
                }}
            />

            <div className="relative border-l-2 border-slate-200 dark:border-slate-800 ml-4 md:ml-6 space-y-12">
                {sortedRoadmapWithIndex.map(({ step, index }) => {
                    const isCompleted = completedSteps.includes(index);
                    const isLocked = index > 0 && !completedSteps.includes(index - 1) && !isCompleted;
                    const hasQuiz = !!step.quiz;

                    return (
                        <div key={index} id={`step-${index}`} className="relative pl-8 md:pl-12">
                            {/* Timeline Node */}
                            <div
                                className={clsx(
                                    "absolute -left-[9px] top-0 w-5 h-5 rounded-full border-4 transition-colors",
                                    isCompleted ? "bg-green-500 border-green-100" :
                                        (isLocked ? "bg-slate-300 border-slate-100" : "bg-blue-500 border-blue-100")
                                )}
                            />

                            <div className={clsx(
                                "group rounded-xl border p-6 transition-all duration-300",
                                isCompleted ? "bg-green-50/50 border-green-200" :
                                    (isLocked ? "bg-slate-50 border-slate-200 opacity-70" : "bg-white border-slate-200 shadow-sm hover:shadow-md hover:border-blue-300")
                            )}>
                                <div className="flex flex-col md:flex-row md:items-start justify-between gap-4 mb-4">
                                    <div>
                                        <span className="text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1 block">
                                            {step.step}
                                        </span>
                                        <h3 className="text-xl font-bold text-slate-900 flex items-center gap-2">
                                            {step.title}
                                            {isCompleted && <CheckCircle2 className="w-5 h-5 text-green-500" />}
                                            {isLocked && <Lock className="w-4 h-4 text-slate-400" />}
                                        </h3>
                                        <p className="text-slate-600 mt-2 leading-relaxed">
                                            {step.description}
                                        </p>
                                    </div>

                                    <div className="flex-shrink-0">
                                        {!isCompleted && !isLocked && (
                                            <Button
                                                onClick={() => handleStepClick(index)}
                                                className={clsx(
                                                    "gap-2",
                                                    hasQuiz ? "bg-blue-600 hover:bg-blue-700" : "bg-slate-900"
                                                )}
                                            >
                                                {hasQuiz ? <PlayCircle className="w-4 h-4" /> : <CheckCircle2 className="w-4 h-4" />}
                                                {hasQuiz ? "퀴즈 풀고 완료하기" : "완료 표시하기"}
                                            </Button>
                                        )}
                                        {isCompleted && (
                                            <span className="inline-flex items-center px-3 py-1 rounded-full bg-green-100 text-green-700 text-sm font-medium">
                                                완료됨
                                            </span>
                                        )}
                                    </div>
                                </div>

                                <div className="grid md:grid-cols-2 gap-6 mt-6">
                                    <div>
                                        <h4 className="text-sm font-semibold text-slate-900 mb-3 flex items-center gap-2">
                                            <BookOpen className="w-4 h-4 text-blue-500" />
                                            Key Topics
                                        </h4>
                                        <ul className="space-y-2">
                                            {step.topics.map((topic, i) => (
                                                <li key={i} className="flex items-center text-sm text-slate-600">
                                                    <span className="w-1.5 h-1.5 bg-blue-400 rounded-full mr-2" />
                                                    {topic}
                                                </li>
                                            ))}
                                        </ul>
                                    </div>

                                    <div>
                                        <h4 className="text-sm font-semibold text-slate-900 mb-3 flex items-center gap-2">
                                            <ExternalLink className="w-4 h-4 text-purple-500" />
                                            Recommended Resources
                                        </h4>
                                        <ul className="space-y-2">
                                            {step.resources.map((res, i) => (
                                                <li key={i}>
                                                    <a
                                                        href={res.url}
                                                        target="_blank"
                                                        rel="noopener noreferrer"
                                                        className="text-sm text-blue-600 hover:text-blue-800 hover:underline flex items-center gap-1 group/link"
                                                    >
                                                        {res.name}
                                                        <ChevronRight className="w-3 h-3 opacity-0 group-hover/link:opacity-100 transition-opacity" />
                                                    </a>
                                                </li>
                                            ))}
                                        </ul>
                                    </div>
                                </div>
                            </div>
                        </div>
                    );
                })}
            </div>

            {quizStepIndex !== null && roadmap[quizStepIndex].quiz && (
                <QuizModal
                    isOpen={quizStepIndex !== null}
                    onClose={() => setQuizStepIndex(null)}
                    question={roadmap[quizStepIndex].quiz!.question}
                    options={roadmap[quizStepIndex].quiz!.options}
                    correctAnswer={roadmap[quizStepIndex].quiz!.correctAnswer}
                    onComplete={(success) => {
                        if (success) markStepComplete(quizStepIndex);
                    }}
                />
            )}
        </div>
    );
}
