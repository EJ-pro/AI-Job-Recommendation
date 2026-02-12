'use client';

import React, { useRef } from 'react';
import { RoadmapStep } from '@/lib/data';
import { CheckCircle2, Lock, Map, ChevronLeft, ChevronRight } from 'lucide-react';
import clsx from 'clsx';

interface RoadmapOverviewProps {
    roadmap: RoadmapStep[];
    completedSteps: number[];
    onStepClick?: (index: number) => void;
}

export function RoadmapOverview({ roadmap, completedSteps, onStepClick }: RoadmapOverviewProps) {
    const scrollContainerRef = useRef<HTMLDivElement>(null);

    const scroll = (direction: 'left' | 'right') => {
        if (scrollContainerRef.current) {
            const container = scrollContainerRef.current;
            const scrollAmount = 300; // Adjust scroll amount as needed
            const newScrollLeft = direction === 'left'
                ? container.scrollLeft - scrollAmount
                : container.scrollLeft + scrollAmount;

            container.scrollTo({
                left: newScrollLeft,
                behavior: 'smooth'
            });
        }
    };

    return (
        <div className="w-full bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl p-6 mb-8 shadow-sm">
            <div className="flex items-center gap-2 mb-6">
                <Map className="w-5 h-5 text-primary" />
                <h3 className="text-lg font-bold text-slate-900 dark:text-white">로드맵 한눈에 보기</h3>
            </div>

            <div className="relative group/container">
                {/* Navigation Buttons */}
                <button
                    onClick={() => scroll('left')}
                    className="absolute left-0 top-1/2 -translate-y-1/2 -translate-x-4 z-20 w-10 h-10 bg-white dark:bg-slate-800 rounded-full shadow-lg border border-slate-200 dark:border-slate-700 flex items-center justify-center text-slate-600 dark:text-slate-300 hover:scale-110 transition-transform opacity-0 group-hover/container:opacity-100 disabled:opacity-0"
                    aria-label="Previous steps"
                >
                    <ChevronLeft size={20} />
                </button>

                <button
                    onClick={() => scroll('right')}
                    className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-4 z-20 w-10 h-10 bg-white dark:bg-slate-800 rounded-full shadow-lg border border-slate-200 dark:border-slate-700 flex items-center justify-center text-slate-600 dark:text-slate-300 hover:scale-110 transition-transform opacity-0 group-hover/container:opacity-100 disabled:opacity-0"
                    aria-label="Next steps"
                >
                    <ChevronRight size={20} />
                </button>

                <div
                    ref={scrollContainerRef}
                    className="relative overflow-x-auto pb-4 pt-2 -mx-2 px-2 no-scrollbar snap-x scroll-smooth"
                >
                    {/* Horizontal Connection Line */}
                    <div className="absolute top-[34px] left-0 w-[200%] h-1 bg-slate-100 dark:bg-slate-800 rounded-full -z-10" />

                    {/* Progress Line - simplified for scrolling */}
                    <div
                        className="absolute top-[34px] left-0 h-1 bg-primary/30 rounded-full transition-all duration-500 -z-10"
                        style={{
                            width: `${Math.min(100, (completedSteps.length / (roadmap.length - 1)) * 100)}%`
                        }}
                    />

                    <div className="flex gap-4">
                        {roadmap.map((step, index) => {
                            const isCompleted = completedSteps.includes(index);
                            const isLocked = index > 0 && !completedSteps.includes(index - 1) && !isCompleted;

                            return (
                                <div
                                    key={index}
                                    className="flex flex-col items-center min-w-[120px] snap-center cursor-pointer group"
                                    onClick={() => onStepClick && onStepClick(index)}
                                >
                                    {/* Node Circle */}
                                    <div className={clsx(
                                        "w-12 h-12 rounded-full flex items-center justify-center border-4 transition-all duration-300 mb-3 bg-white dark:bg-slate-900 z-10 relative",
                                        isCompleted ? "border-green-500 text-green-500" :
                                            (isLocked ? "border-slate-200 text-slate-300 dark:border-slate-700 dark:text-slate-600" : "border-blue-500 text-blue-500 shadow-lg shadow-blue-500/20 scale-110")
                                    )}>
                                        {isCompleted ? (
                                            <CheckCircle2 size={20} strokeWidth={3} />
                                        ) : isLocked ? (
                                            <Lock size={18} />
                                        ) : (
                                            <span className="text-lg font-bold">{index + 1}</span>
                                        )}
                                    </div>

                                    {/* Text Content */}
                                    <div className="text-center space-y-1 w-full px-1">
                                        <div className={clsx(
                                            "text-xs font-bold uppercase tracking-wider",
                                            isCompleted ? "text-green-600" :
                                                (isLocked ? "text-slate-400" : "text-blue-600")
                                        )}>
                                            Phase {index + 1}
                                        </div>
                                        <div className={clsx(
                                            "text-sm font-medium leading-tight line-clamp-2 min-h-[2.5rem] flex items-center justify-center px-1 transition-colors",
                                            isCompleted ? "text-slate-700 dark:text-slate-300" :
                                                (isLocked ? "text-slate-400 dark:text-slate-600" : "text-slate-900 dark:text-white font-bold")
                                        )}>
                                            {step.title}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            </div>

            <style jsx>{`
                .no-scrollbar::-webkit-scrollbar {
                    display: none;
                }
                .no-scrollbar {
                    -ms-overflow-style: none;
                    scrollbar-width: none;
                }
            `}</style>
        </div>
    );
}
